#!/usr/bin/env python3
"""Build temporal substrate index from video and possession config."""
import argparse
import logging
import json
import os
import time
from pathlib import Path
import yaml
import jsonlines
import cv2
import numpy as np
from dotenv import load_dotenv

# Import faiss (with try/except for optional dependency)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

from src.data_models import Possession, Segment
from src.segment_generator import SegmentGenerator
from src.possession_detector import PossessionDetector
from src.mllm_possession_detector import MLLMPossessionDetector
from src.clip_extractor import ClipExtractor
from src.segment_enricher import SegmentEnricher
from src.embedding_generator import EmbeddingGenerator
from src.trajectory_embedding_generator import TrajectoryEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_possessions(config_path: Path, game_id: str) -> list[Possession]:
    """Load possession definitions from YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    possessions = []
    for i, poss_data in enumerate(config['possessions']):
        possession = Possession(
            game_id=game_id,
            possession_id=i,
            start=poss_data['start'],
            end=poss_data['end'],
            metadata=poss_data.get('metadata', {})
        )
        possessions.append(possession)

    logger.info(f"Loaded {len(possessions)} possessions from {config_path}")
    return possessions


def validate_video(video_path: Path) -> dict:
    """Validate video file and extract metadata."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    metadata = {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

    logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, {width}x{height}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Build temporal substrate index')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--config', help='Path to possession config (YAML). Optional if using --auto-detect-possessions')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--window-sizes', nargs='+', type=float, default=[2.0, 4.0, 8.0],
                       help='Window sizes in seconds')
    parser.add_argument('--stride', type=float, default=1.0, help='Stride in seconds')

    # Auto-detection options
    parser.add_argument('--auto-detect-possessions', action='store_true',
                       help='Automatically detect possessions from video')
    parser.add_argument('--min-possession', type=float, default=5.0,
                       help='Minimum possession duration (seconds)')
    parser.add_argument('--max-possession', type=float, default=30.0,
                       help='Maximum possession duration (seconds)')
    parser.add_argument('--activity-threshold', type=float, default=2.0,
                       help='Activity detection threshold (lower = more sensitive)')
    parser.add_argument('--scene-threshold', type=float, default=30.0,
                       help='Scene change threshold (higher = fewer cuts)')

    # MLLM-based detection options
    parser.add_argument('--use-mllm', action='store_true',
                       help='Use MLLM-based possession detection (GPT-4V)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--mllm-model', default='gpt-4o',
                       help='MLLM model to use (gpt-4o, gpt-4-turbo, gpt-4o-mini)')
    parser.add_argument('--mllm-sample-rate', type=float, default=2.5,
                       help='Sample rate for MLLM detection (seconds)')
    parser.add_argument('--mllm-batch-size', type=int, default=6,
                       help='Frames per API request')

    # Clip extraction options
    parser.add_argument('--extract-clips', action='store_true',
                       help='Extract video clips for all segments')
    parser.add_argument('--clips-dir', default='outputs/clips',
                       help='Directory to save extracted clips')
    parser.add_argument('--max-clips', type=int,
                       help='Maximum number of clips to extract (for testing)')

    # M2: Enrichment options
    parser.add_argument('--enrich-segments', action='store_true',
                       help='Enrich segments with GPT-4V metadata extraction')
    parser.add_argument('--generate-embeddings', action='store_true',
                       help='Generate CLIP embeddings for segments')
    parser.add_argument('--enrichment-batch-size', type=int, default=10,
                       help='Segments per enrichment batch (default: 10)')
    parser.add_argument('--device', default='cpu',
                       help='Device for CLIP model (cpu, cuda)')

    args = parser.parse_args()

    # Validate arguments
    if not args.auto_detect_possessions and not args.use_mllm and not args.config:
        parser.error("Either --config, --auto-detect-possessions, or --use-mllm must be provided")

    if args.enrich_segments and not args.extract_clips:
        parser.error("--enrich-segments requires --extract-clips (need video clips for enrichment)")

    if args.enrich_segments:
        api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            parser.error("--enrich-segments requires OpenAI API key. Set OPENAI_API_KEY env var or use --openai-api-key")

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract game_id from video filename
    game_id = video_path.stem

    # Validate video
    logger.info(f"Processing game: {game_id}")
    video_metadata = validate_video(video_path)

    # Load or detect possessions
    if args.use_mllm:
        logger.info("Using MLLM-based possession detection...")
        api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            parser.error("OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-api-key")

        detector = MLLMPossessionDetector(
            api_key=api_key,
            model=args.mllm_model,
            sample_rate=args.mllm_sample_rate,
            batch_size=args.mllm_batch_size,
            min_possession_duration=args.min_possession,
            max_possession_duration=args.max_possession
        )
        possessions = detector.detect_possessions(video_path, game_id)
    elif args.auto_detect_possessions:
        logger.info("Auto-detecting possessions from video (basic OpenCV)...")
        detector = PossessionDetector(
            min_possession_duration=args.min_possession,
            max_possession_duration=args.max_possession,
            activity_threshold=args.activity_threshold,
            scene_threshold=args.scene_threshold
        )
        possessions = detector.detect_possessions(video_path, game_id)
    else:
        config_path = Path(args.config)
        possessions = load_possessions(config_path, game_id)

    # Validate possession timestamps against video duration
    for poss in possessions:
        if poss.end > video_metadata['duration']:
            logger.warning(
                f"Possession {poss.possession_id} end time ({poss.end}s) "
                f"exceeds video duration ({video_metadata['duration']:.1f}s)"
            )

    # Generate segments
    generator = SegmentGenerator(window_sizes=args.window_sizes, stride=args.stride)
    all_segments = []
    for possession in possessions:
        segments = generator.generate_segments(possession)
        all_segments.extend(segments)

    # Extract clips if requested
    clips_extracted = False
    clips_dir = None
    if args.extract_clips:
        clips_dir = Path(args.clips_dir)
        extractor = ClipExtractor(clips_dir)
        logger.info("Extracting video clips...")
        extractor.extract_clips_for_segments(
            video_path,
            all_segments,
            max_clips=args.max_clips
        )
        clips_extracted = True

    # M2: Enrich segments and generate embeddings
    segments_enriched = False
    embeddings_generated = False

    if args.generate_embeddings or args.enrich_segments:
        logger.info("=" * 80)
        logger.info("M2: Generating embeddings and enriching segments")
        logger.info("=" * 80)

        # Initialize generators
        embedding_gen = None
        traj_gen = None
        enricher = None

        if args.generate_embeddings:
            logger.info(f"Initializing CLIP embedding generator (device: {args.device})...")
            embedding_gen = EmbeddingGenerator(device=args.device)

        if args.enrich_segments:
            logger.info("Initializing GPT-4V segment enricher...")
            api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
            enricher = SegmentEnricher(api_key=api_key, model=args.mllm_model)

            logger.info("Initializing trajectory embedding generator...")
            traj_gen = TrajectoryEmbeddingGenerator(device=args.device)

        # Process segments
        total_segments = len(all_segments)
        for i, segment in enumerate(all_segments):
            clip_path = clips_dir / f"{segment.segment_id}.mp4"

            if not clip_path.exists():
                logger.warning(f"Clip not found for {segment.segment_id}, skipping")
                continue

            # Generate visual embedding
            if embedding_gen:
                visual_emb = embedding_gen.encode_video_clip(clip_path, num_frames=4)
                segment.video_embedding = visual_emb.tolist()

            # Enrich with GPT-4V
            if enricher:
                metadata = enricher.enrich_segment(clip_path)

                # Generate trajectory embedding from motion description
                if traj_gen:
                    traj_emb = traj_gen.generate_trajectory_embedding(
                        metadata.player_movements
                    )
                    segment.trajectory_embedding = traj_emb.tolist()

                # Store enriched metadata
                if segment.metadata is None:
                    segment.metadata = {}

                segment.metadata.update({
                    'derived': {
                        'ball_speed_estimate': metadata.ball_speed_estimate,
                        'player_ball_proximity': metadata.player_ball_proximity,
                        'offensive_spacing': metadata.offensive_spacing,
                        'screen_angle_est': metadata.screen_angle_est,
                        'motion_intensity': metadata.motion_intensity
                    },
                    'court_semantics': {
                        'ball_zone': metadata.ball_zone,
                        'primary_zone': metadata.primary_zone,
                        'paint_occupied': metadata.paint_occupied
                    },
                    'weak_events': {
                        'possible_screen': metadata.possible_screen,
                        'possible_drive': metadata.possible_drive,
                        'possible_shot': metadata.possible_shot,
                        'possible_pass': metadata.possible_pass,
                        'possible_rebound': metadata.possible_rebound
                    }
                })

            # Progress logging
            if (i + 1) % args.enrichment_batch_size == 0 or (i + 1) == total_segments:
                logger.info(f"Processed {i+1}/{total_segments} segments")

                # Rate limiting for API
                if enricher and (i + 1) < total_segments:
                    time.sleep(0.5)

        segments_enriched = args.enrich_segments
        embeddings_generated = args.generate_embeddings

        # Print API usage if enrichment was performed
        if enricher:
            usage = enricher.get_api_usage()
            logger.info(f"API usage: {usage['total_calls']} calls, {usage['total_tokens']} tokens")
            logger.info(f"Estimated cost: ${usage['estimated_cost']:.2f}")

        # Build Faiss indices if embeddings were generated
        if embeddings_generated and FAISS_AVAILABLE:
            logger.info("=" * 80)
            logger.info("Building Faiss indices for multi-modal retrieval")
            logger.info("=" * 80)

            # Collect visual embeddings
            visual_embeddings = []
            trajectory_embeddings = []
            valid_segment_ids = []

            for segment in all_segments:
                if segment.video_embedding is not None:
                    visual_embeddings.append(segment.video_embedding)
                    valid_segment_ids.append(segment.segment_id)

                    if segment.trajectory_embedding is not None:
                        trajectory_embeddings.append(segment.trajectory_embedding)
                    else:
                        # Placeholder for segments without trajectory
                        trajectory_embeddings.append([0.0] * 512)

            if visual_embeddings:
                # Convert to numpy arrays
                visual_embs = np.array(visual_embeddings, dtype='float32')
                traj_embs = np.array(trajectory_embeddings, dtype='float32')

                # L2 normalize
                visual_embs = visual_embs / np.linalg.norm(visual_embs, axis=1, keepdims=True)
                traj_embs = traj_embs / np.linalg.norm(traj_embs, axis=1, keepdims=True)

                # Build indices
                d = visual_embs.shape[1]  # 512 for ViT-B-32

                # Visual index
                visual_index = faiss.IndexFlatIP(d)
                visual_index.add(visual_embs)
                visual_index_path = output_dir / f"{game_id}_visual_index.bin"
                faiss.write_index(visual_index, str(visual_index_path))
                logger.info(f"✓ Built visual index: {len(visual_embs)} vectors, {d} dims")

                # Trajectory index
                trajectory_index = faiss.IndexFlatIP(d)
                trajectory_index.add(traj_embs)
                trajectory_index_path = output_dir / f"{game_id}_trajectory_index.bin"
                faiss.write_index(trajectory_index, str(trajectory_index_path))
                logger.info(f"✓ Built trajectory index: {len(traj_embs)} vectors, {d} dims")

                # Save index mapping
                index_mapping = {
                    "index_to_segment_id": {i: seg_id for i, seg_id in enumerate(valid_segment_ids)},
                    "indices": {
                        "visual": f"{game_id}_visual_index.bin",
                        "trajectory": f"{game_id}_trajectory_index.bin"
                    },
                    "dimension": int(d),
                    "num_vectors": len(visual_embs)
                }

                mapping_path = output_dir / f"{game_id}_index_mapping.json"
                with open(mapping_path, 'w') as f:
                    json.dump(index_mapping, f, indent=2)

                logger.info(f"✓ Saved index mapping: {mapping_path}")
            else:
                logger.warning("No embeddings found, skipping Faiss index building")

        elif embeddings_generated and not FAISS_AVAILABLE:
            logger.warning("Faiss not available, skipping index building. Install with: pip install faiss-cpu")

    # Save outputs
    possessions_file = output_dir / f"{game_id}_possessions.jsonl"
    segments_file = output_dir / f"{game_id}_segments.jsonl"

    with jsonlines.open(possessions_file, 'w') as writer:
        for poss in possessions:
            writer.write(poss.to_dict())

    with jsonlines.open(segments_file, 'w') as writer:
        for seg in all_segments:
            writer.write(seg.to_dict())

    # Save index metadata
    index_metadata = {
        'game_id': game_id,
        'video_path': str(video_path.absolute()),
        'video_metadata': video_metadata,
        'num_possessions': len(possessions),
        'num_segments': len(all_segments),
        'window_sizes': args.window_sizes,
        'stride': args.stride,
        'auto_detected': args.auto_detect_possessions or args.use_mllm,
        'detection_method': 'mllm' if args.use_mllm else ('opencv' if args.auto_detect_possessions else 'manual'),
        'clips_extracted': clips_extracted,
        'segments_enriched': segments_enriched,
        'embeddings_generated': embeddings_generated
    }

    if clips_extracted:
        index_metadata['clips_dir'] = str(clips_dir.absolute())

    if args.use_mllm:
        index_metadata['mllm_model'] = args.mllm_model

    if segments_enriched:
        index_metadata['enrichment_model'] = args.mllm_model

    if embeddings_generated:
        index_metadata['embedding_model'] = 'ViT-B-32'
        index_metadata['embedding_dimension'] = 512

    metadata_file = output_dir / f"{game_id}_index_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(index_metadata, f, indent=2)

    logger.info(f"✓ Generated {len(all_segments)} segments from {len(possessions)} possessions")
    logger.info(f"✓ Outputs written to {output_dir}")
    logger.info(f"  - Possessions: {possessions_file}")
    logger.info(f"  - Segments: {segments_file}")
    logger.info(f"  - Metadata: {metadata_file}")


if __name__ == '__main__':
    main()
