#!/usr/bin/env python3
"""Create demo Faiss indices from our enriched test segments."""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
import jsonlines

load_dotenv()

from src.segment_enricher import SegmentEnricher
from src.embedding_generator import EmbeddingGenerator
from src.trajectory_embedding_generator import TrajectoryEmbeddingGenerator
from src.data_models import Segment

print("Creating demo indices for viewer testing...")

# Configuration
api_key = os.getenv('OPENAI_API_KEY')
clips_dir = Path("outputs/clips")
output_dir = Path("outputs")

# Demo clips to enrich
demo_clips = [
    "nba_final_p0_s0",
    "nba_final_p0_s1",
    "nba_final_p0_s2",
    "nba_final_p0_s3",
    "nba_final_p0_s4",
    "nba_final_p8_s5",
    "nba_final_p8_s6",
    "nba_final_p15_s2",
    "nba_final_p15_s3",
    "nba_final_p0_s15",
]

# Load original segments
segments_file = output_dir / "nba_final_segments.jsonl"
all_segments = {}
with jsonlines.open(segments_file) as reader:
    for seg_dict in reader:
        seg = Segment.from_dict(seg_dict)
        all_segments[seg.segment_id] = seg

# Initialize generators
print("Initializing generators...")
embedding_gen = EmbeddingGenerator(device='cpu')
enricher = SegmentEnricher(api_key=api_key, model='gpt-4o')
traj_gen = TrajectoryEmbeddingGenerator(device='cpu')

# Enrich demo segments
enriched_segments = []
visual_embeddings = []
trajectory_embeddings = []

print(f"\nEnriching {len(demo_clips)} segments...")
for i, seg_id in enumerate(demo_clips):
    clip_path = clips_dir / f"{seg_id}.mp4"

    if not clip_path.exists():
        print(f"  ⚠ Skipping {seg_id} - clip not found")
        continue

    print(f"  [{i+1}/{len(demo_clips)}] {seg_id}...")

    # Get segment data
    segment = all_segments[seg_id]

    # Generate visual embedding
    visual_emb = embedding_gen.encode_video_clip(clip_path, num_frames=4)

    # Enrich with GPT-4V
    metadata = enricher.enrich_segment(clip_path)

    # Generate trajectory embedding
    traj_emb = traj_gen.generate_trajectory_embedding(metadata.player_movements)

    # Store embeddings
    segment.video_embedding = visual_emb.tolist()
    segment.trajectory_embedding = traj_emb.tolist()

    # Store metadata
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

    enriched_segments.append(segment)
    visual_embeddings.append(visual_emb)
    trajectory_embeddings.append(traj_emb)

print(f"\n✓ Enriched {len(enriched_segments)} segments")

# Save enriched segments
demo_segments_file = output_dir / "nba_final_segments_demo.jsonl"
with jsonlines.open(demo_segments_file, 'w') as writer:
    for seg in enriched_segments:
        writer.write(seg.to_dict())

print(f"✓ Saved enriched segments: {demo_segments_file}")

# Build Faiss indices
print("\nBuilding Faiss indices...")

# Convert to numpy arrays and normalize
visual_embs = np.array(visual_embeddings, dtype='float32')
traj_embs = np.array(trajectory_embeddings, dtype='float32')

visual_embs = visual_embs / np.linalg.norm(visual_embs, axis=1, keepdims=True)
traj_embs = traj_embs / np.linalg.norm(traj_embs, axis=1, keepdims=True)

# Build indices
d = 512

# Visual index
visual_index = faiss.IndexFlatIP(d)
visual_index.add(visual_embs)
visual_index_path = output_dir / "nba_final_visual_index.bin"
faiss.write_index(visual_index, str(visual_index_path))
print(f"✓ Visual index: {len(visual_embs)} vectors → {visual_index_path}")

# Trajectory index
trajectory_index = faiss.IndexFlatIP(d)
trajectory_index.add(traj_embs)
trajectory_index_path = output_dir / "nba_final_trajectory_index.bin"
faiss.write_index(trajectory_index, str(trajectory_index_path))
print(f"✓ Trajectory index: {len(traj_embs)} vectors → {trajectory_index_path}")

# Save index mapping
index_mapping = {
    "index_to_segment_id": {i: seg.segment_id for i, seg in enumerate(enriched_segments)},
    "indices": {
        "visual": "nba_final_visual_index.bin",
        "trajectory": "nba_final_trajectory_index.bin"
    },
    "dimension": int(d),
    "num_vectors": len(enriched_segments)
}

mapping_path = output_dir / "nba_final_index_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(index_mapping, f, indent=2)

print(f"✓ Index mapping: {mapping_path}")

# Show API usage
usage = enricher.get_api_usage()
print(f"\n📊 API Usage: {usage['total_calls']} calls, {usage['total_tokens']} tokens, ${usage['estimated_cost']:.4f}")

print("\n" + "=" * 80)
print("✅ DEMO INDICES READY!")
print("=" * 80)
print("\nYou can now test the search in the viewer:")
print("  1. Run: streamlit run viewer/app.py")
print("  2. Go to 'Search Segments' tab")
print("  3. Try queries like:")
print("     - 'pick and roll'")
print("     - 'drive to basket'")
print("     - 'screen action'")
print("\nNote: This is a demo with 10 segments. Run full enrichment for all 364 segments.")
print("=" * 80 + "\n")
