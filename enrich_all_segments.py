#!/usr/bin/env python3
"""Enrich all 364 segments with expanded event taxonomy."""
import os
from pathlib import Path
from dotenv import load_dotenv
import jsonlines
import numpy as np
import faiss
import json
from tqdm import tqdm

load_dotenv()

from src.segment_enricher import SegmentEnricher
from src.embedding_generator import EmbeddingGenerator
from src.trajectory_embedding_generator import TrajectoryEmbeddingGenerator
from src.data_models import Segment

print("=" * 80)
print("FULL SEGMENT ENRICHMENT - 364 SEGMENTS")
print("=" * 80)

# Configuration
api_key = os.getenv('OPENAI_API_KEY')
clips_dir = Path("outputs/clips")
output_dir = Path("outputs")
segments_file = output_dir / "nba_final_segments.jsonl"

# Initialize generators
print("\nInitializing generators...")
embedding_gen = EmbeddingGenerator(device='cpu')
enricher = SegmentEnricher(api_key=api_key, model='gpt-4o')
traj_gen = TrajectoryEmbeddingGenerator(device='cpu')
print("✓ Generators ready")

# Load segments
print(f"\nLoading segments from {segments_file}...")
all_segments = []
with jsonlines.open(segments_file) as reader:
    for seg_dict in reader:
        seg = Segment.from_dict(seg_dict)
        all_segments.append(seg)

print(f"✓ Loaded {len(all_segments)} segments")

# Enrich segments
enriched_segments = []
visual_embeddings = []
trajectory_embeddings = []

print(f"\n{'='*80}")
print(f"Enriching {len(all_segments)} segments...")
print(f"Estimated time: ~{len(all_segments) * 2.5 / 60:.1f} minutes")
print(f"Estimated cost: ~${len(all_segments) * 0.004:.2f}")
print(f"{'='*80}\n")

for i, segment in enumerate(tqdm(all_segments, desc="Enriching")):
    clip_path = clips_dir / f"{segment.segment_id}.mp4"

    if not clip_path.exists():
        print(f"\n⚠ Skipping {segment.segment_id} - clip not found")
        continue

    try:
        # Generate visual embedding
        visual_emb = embedding_gen.encode_video_clip(clip_path, num_frames=4)

        # Enrich with GPT-4V (expanded events)
        metadata = enricher.enrich_segment(clip_path)

        # Generate trajectory embedding
        traj_emb = traj_gen.generate_trajectory_embedding(metadata.player_movements)

        # Store embeddings
        segment.video_embedding = visual_emb.tolist()
        segment.trajectory_embedding = traj_emb.tolist()

        # Store metadata with expanded events
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
                # Original events
                'possible_screen': metadata.possible_screen,
                'possible_drive': metadata.possible_drive,
                'possible_shot': metadata.possible_shot,
                'possible_pass': metadata.possible_pass,
                'possible_rebound': metadata.possible_rebound,
                # Expanded events
                'possible_steal': metadata.possible_steal,
                'possible_block': metadata.possible_block,
                'possible_cut': metadata.possible_cut,
                'possible_post_up': metadata.possible_post_up,
                'possible_pick_and_roll': metadata.possible_pick_and_roll,
                'possible_fast_break': metadata.possible_fast_break,
                'possible_defensive_rotation': metadata.possible_defensive_rotation,
                'possible_assist': metadata.possible_assist,
                'possible_turnover': metadata.possible_turnover,
                'possible_dribble_move': metadata.possible_dribble_move,
                'rebound_type': metadata.rebound_type
            },
            'player_movements': metadata.player_movements
        })

        enriched_segments.append(segment)
        visual_embeddings.append(visual_emb)
        trajectory_embeddings.append(traj_emb)

    except Exception as e:
        print(f"\n❌ Error enriching {segment.segment_id}: {e}")
        continue

    # Checkpoint every 50 segments
    if (i + 1) % 50 == 0:
        print(f"\n💾 Checkpoint: {i+1}/{len(all_segments)} segments enriched")

print(f"\n✓ Enriched {len(enriched_segments)} segments")

# Save enriched segments
enriched_file = output_dir / "nba_final_segments_enriched.jsonl"
with jsonlines.open(enriched_file, 'w') as writer:
    for seg in enriched_segments:
        writer.write(seg.to_dict())

print(f"✓ Saved to {enriched_file}")

# Build Faiss indices
print("\nBuilding Faiss indices...")

visual_embs = np.array(visual_embeddings, dtype='float32')
traj_embs = np.array(trajectory_embeddings, dtype='float32')

# Normalize
visual_embs = visual_embs / np.linalg.norm(visual_embs, axis=1, keepdims=True)
traj_embs = traj_embs / np.linalg.norm(traj_embs, axis=1, keepdims=True)

# Visual index
visual_index = faiss.IndexFlatIP(512)
visual_index.add(visual_embs)
visual_index_path = output_dir / "nba_final_visual_index.bin"
faiss.write_index(visual_index, str(visual_index_path))
print(f"✓ Visual index: {len(visual_embs)} vectors")

# Trajectory index
trajectory_index = faiss.IndexFlatIP(512)
trajectory_index.add(traj_embs)
trajectory_index_path = output_dir / "nba_final_trajectory_index.bin"
faiss.write_index(trajectory_index, str(trajectory_index_path))
print(f"✓ Trajectory index: {len(traj_embs)} vectors")

# Save index mapping
index_mapping = {
    "index_to_segment_id": {i: seg.segment_id for i, seg in enumerate(enriched_segments)},
    "indices": {
        "visual": "nba_final_visual_index.bin",
        "trajectory": "nba_final_trajectory_index.bin"
    },
    "dimension": 512,
    "num_vectors": len(enriched_segments)
}

mapping_path = output_dir / "nba_final_index_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(index_mapping, f, indent=2)

print(f"✓ Index mapping saved")

# Show API usage
usage = enricher.get_api_usage()
print(f"\n{'='*80}")
print(f"📊 API USAGE")
print(f"{'='*80}")
print(f"Total API calls: {usage['total_calls']}")
print(f"Total tokens: {usage['total_tokens']:,}")
print(f"Estimated cost: ${usage['estimated_cost']:.2f}")
print(f"{'='*80}\n")

print("✅ FULL ENRICHMENT COMPLETE!")
print(f"   - {len(enriched_segments)} segments enriched")
print(f"   - 15 event types detected per segment")
print(f"   - Ready for search in viewer")
print("\n🚀 Restart viewer: streamlit run viewer/app.py")
