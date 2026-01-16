#!/usr/bin/env python3
"""Show detailed enrichment results for quality evaluation."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

from src.segment_enricher import SegmentEnricher
from src.embedding_generator import EmbeddingGenerator
from src.trajectory_embedding_generator import TrajectoryEmbeddingGenerator

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env")
    exit(1)

# Test clips - let's do 5 diverse segments
clips_dir = Path("outputs/clips")
test_clips = [
    "nba_final_p0_s0.mp4",   # First segment
    "nba_final_p5_s10.mp4",  # Mid-game segment
    "nba_final_p8_s5.mp4",   # Another possession
    "nba_final_p15_s2.mp4",  # Late game segment
    "nba_final_p0_s15.mp4",  # 8-second window
]

print("=" * 100)
print("M2 SEGMENT ENRICHMENT - QUALITY EVALUATION")
print("=" * 100)

# Initialize generators
print("\nInitializing generators...")
embedding_gen = EmbeddingGenerator(device='cpu')
enricher = SegmentEnricher(api_key=api_key, model='gpt-4o')
traj_gen = TrajectoryEmbeddingGenerator(device='cpu')
print("✓ All generators initialized\n")

results = []

# Test each clip
for i, clip_name in enumerate(test_clips):
    clip_path = clips_dir / clip_name

    if not clip_path.exists():
        print(f"⚠ Clip not found: {clip_name}")
        continue

    print(f"\n{'=' * 100}")
    print(f"SEGMENT {i+1}/{len(test_clips)}: {clip_name}")
    print(f"{'=' * 100}\n")

    # Generate visual embedding
    print("→ Generating visual embedding...")
    visual_emb = embedding_gen.encode_video_clip(clip_path, num_frames=4)
    print(f"  ✓ Shape: {visual_emb.shape}, L2-normalized: {abs(visual_emb.dot(visual_emb) - 1.0) < 0.01}")

    # Enrich with GPT-4V
    print("\n→ Extracting metadata with GPT-4V...")
    metadata = enricher.enrich_segment(clip_path)

    # Generate trajectory embedding
    print("\n→ Generating trajectory embedding...")
    traj_emb = traj_gen.generate_trajectory_embedding(metadata.player_movements)
    print(f"  ✓ Shape: {traj_emb.shape}, L2-normalized: {abs(traj_emb.dot(traj_emb) - 1.0) < 0.01}")

    # Display enriched metadata
    print("\n" + "─" * 100)
    print("ENRICHED METADATA:")
    print("─" * 100)

    print(f"\n📍 MOTION/TRAJECTORY:")
    print(f"  Player movements: {metadata.player_movements}")
    print(f"  Motion intensity: {metadata.motion_intensity:.2f}")

    print(f"\n🏀 COURT SEMANTICS:")
    print(f"  Ball zone: {metadata.ball_zone}")
    print(f"  Primary action zone: {metadata.primary_zone}")
    print(f"  Paint occupied: {metadata.paint_occupied}")

    print(f"\n📊 DERIVED METRICS:")
    print(f"  Ball speed: {metadata.ball_speed_estimate}")
    print(f"  Player-ball proximity: {metadata.player_ball_proximity}")
    print(f"  Offensive spacing: {metadata.offensive_spacing}")
    print(f"  Screen angle: {metadata.screen_angle_est}")

    print(f"\n⚡ WEAK EVENT DETECTION (probabilities):")
    events_table = [
        ["Screen", f"{metadata.possible_screen:.2f}"],
        ["Drive", f"{metadata.possible_drive:.2f}"],
        ["Shot", f"{metadata.possible_shot:.2f}"],
        ["Pass", f"{metadata.possible_pass:.2f}"],
        ["Rebound", f"{metadata.possible_rebound:.2f}"],
    ]
    print(tabulate(events_table, headers=["Event", "Probability"], tablefmt="simple"))

    # Store for summary
    results.append({
        'clip': clip_name,
        'metadata': metadata.to_dict(),
        'visual_emb_dim': len(visual_emb),
        'traj_emb_dim': len(traj_emb)
    })

# Save results to file
output_file = Path("outputs/enrichment_quality_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 100)
print("QUALITY EVALUATION SUMMARY")
print("=" * 100)

# API usage summary
usage = enricher.get_api_usage()
print(f"\n📊 API Usage:")
print(f"  Total calls: {usage['total_calls']}")
print(f"  Total tokens: {usage['total_tokens']}")
print(f"  Estimated cost: ${usage['estimated_cost']:.4f}")
print(f"  Cost per segment: ${usage['estimated_cost'] / len(results):.4f}")

# Metadata coverage
print(f"\n✅ Metadata Coverage:")
print(f"  Segments enriched: {len(results)}")
print(f"  Visual embeddings: {len(results)} × 512-dim")
print(f"  Trajectory embeddings: {len(results)} × 512-dim")

# Event detection summary
print(f"\n🎯 Event Detection Summary:")
all_screens = [r['metadata']['possible_screen'] for r in results]
all_drives = [r['metadata']['possible_drive'] for r in results]
all_shots = [r['metadata']['possible_shot'] for r in results]

print(f"  High confidence screens (>0.7): {sum(1 for s in all_screens if s > 0.7)}/{len(all_screens)}")
print(f"  High confidence drives (>0.7): {sum(1 for d in all_drives if d > 0.7)}/{len(all_drives)}")
print(f"  High confidence shots (>0.7): {sum(1 for s in all_shots if s > 0.7)}/{len(all_shots)}")

# Zone distribution
zones = [r['metadata']['ball_zone'] for r in results]
zone_counts = {}
for zone in zones:
    zone_counts[zone] = zone_counts.get(zone, 0) + 1

print(f"\n🗺️  Ball Zone Distribution:")
for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
    print(f"  {zone}: {count}")

print(f"\n💾 Full results saved to: {output_file}")
print("\n" + "=" * 100)
