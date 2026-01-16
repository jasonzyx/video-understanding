#!/usr/bin/env python3
"""Create Faiss indices from already enriched results."""
import json
import numpy as np
import faiss
from pathlib import Path
import jsonlines
from src.data_models import Segment

print("Creating indices from enriched results...")

output_dir = Path("outputs")

# Load our enriched quality results
results_file = output_dir / "enrichment_quality_results.json"
with open(results_file) as f:
    enriched_results = json.load(f)

# Load original segments to get full data
segments_file = output_dir / "nba_final_segments.jsonl"
all_segments = {}
with jsonlines.open(segments_file) as reader:
    for seg_dict in reader:
        seg = Segment.from_dict(seg_dict)
        all_segments[seg.segment_id] = seg

# Map clip names to segment IDs
clip_to_segment = {}
for seg_id, seg in all_segments.items():
    clip_name = f"{seg_id}.mp4"
    clip_to_segment[clip_name] = seg

# Process enriched results and create enriched segments
enriched_segments = []
visual_embeddings = []
trajectory_embeddings = []

print(f"\nProcessing {len(enriched_results)} enriched segments...")

for result in enriched_results:
    clip_name = result['clip']

    if clip_name not in clip_to_segment:
        print(f"  ⚠ Skipping {clip_name} - segment not found")
        continue

    segment = clip_to_segment[clip_name]
    meta = result['metadata']

    # Create fake embeddings for demo (normally from CLIP)
    # In real system, these would be actual CLIP embeddings
    np.random.seed(hash(segment.segment_id) % 2**32)
    visual_emb = np.random.randn(512).astype('float32')
    visual_emb = visual_emb / np.linalg.norm(visual_emb)

    # Trajectory embedding - make it correlated with motion intensity
    traj_emb = np.random.randn(512).astype('float32')
    if 'drive' in meta['player_movements'].lower():
        traj_emb[0:100] += 2.0  # Boost drive-related features
    if 'screen' in meta['player_movements'].lower():
        traj_emb[100:200] += 2.0  # Boost screen-related features
    traj_emb = traj_emb / np.linalg.norm(traj_emb)

    # Store in segment
    segment.video_embedding = visual_emb.tolist()
    segment.trajectory_embedding = traj_emb.tolist()
    segment.metadata = {
        'derived': {
            'ball_speed_estimate': meta['ball_speed_estimate'],
            'player_ball_proximity': meta['player_ball_proximity'],
            'offensive_spacing': meta['offensive_spacing'],
            'screen_angle_est': meta['screen_angle_est'],
            'motion_intensity': meta['motion_intensity']
        },
        'court_semantics': {
            'ball_zone': meta['ball_zone'],
            'primary_zone': meta['primary_zone'],
            'paint_occupied': meta['paint_occupied']
        },
        'weak_events': {
            'possible_screen': meta['possible_screen'],
            'possible_drive': meta['possible_drive'],
            'possible_shot': meta['possible_shot'],
            'possible_pass': meta['possible_pass'],
            'possible_rebound': meta['possible_rebound']
        }
    }

    enriched_segments.append(segment)
    visual_embeddings.append(visual_emb)
    trajectory_embeddings.append(traj_emb)

    print(f"  ✓ {segment.segment_id}")

# Save enriched segments
segments_output = output_dir / "nba_final_segments_demo.jsonl"
with jsonlines.open(segments_output, 'w') as writer:
    for seg in enriched_segments:
        writer.write(seg.to_dict())

print(f"\n✓ Saved {len(enriched_segments)} enriched segments: {segments_output}")

# Build Faiss indices
print("\nBuilding Faiss indices...")

visual_embs = np.array(visual_embeddings, dtype='float32')
traj_embs = np.array(trajectory_embeddings, dtype='float32')

d = 512

# Visual index
visual_index = faiss.IndexFlatIP(d)
visual_index.add(visual_embs)
visual_index_path = output_dir / "nba_final_visual_index.bin"
faiss.write_index(visual_index, str(visual_index_path))
print(f"✓ Visual index: {len(visual_embs)} vectors")

# Trajectory index
trajectory_index = faiss.IndexFlatIP(d)
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
    "dimension": int(d),
    "num_vectors": len(enriched_segments)
}

mapping_path = output_dir / "nba_final_index_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(index_mapping, f, indent=2)

print(f"✓ Index mapping: {mapping_path}")

print("\n" + "=" * 80)
print("✅ DEMO INDICES READY FOR VIEWER!")
print("=" * 80)
print("\n🚀 Launch the viewer:")
print("   streamlit run viewer/app.py")
print("\n📍 Go to 'Search Segments' tab and try:")
print("   - 'pick and roll' (should find screen segment)")
print("   - 'drive to basket' (should find drive segment)")
print("   - 'screen action'")
print("\n⚠️  Note: Using simplified embeddings for demo")
print("   Run full enrichment for production-quality embeddings")
print("=" * 80 + "\n")
