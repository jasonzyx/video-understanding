#!/usr/bin/env python3
"""Test segment enrichment on sample clips."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.segment_enricher import SegmentEnricher
from src.embedding_generator import EmbeddingGenerator
from src.trajectory_embedding_generator import TrajectoryEmbeddingGenerator

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env")
    exit(1)

# Test clips
clips_dir = Path("outputs/clips")
test_clips = [
    "nba_final_p0_s0.mp4",
    "nba_final_p0_s1.mp4",
    "nba_final_p0_s2.mp4"
]

print("=" * 80)
print("Testing M2 Segment Enrichment")
print("=" * 80)

# Initialize generators
print("\nInitializing generators...")
embedding_gen = EmbeddingGenerator(device='cpu')
enricher = SegmentEnricher(api_key=api_key, model='gpt-4o')
traj_gen = TrajectoryEmbeddingGenerator(device='cpu')

print("✓ All generators initialized\n")

# Test each clip
for i, clip_name in enumerate(test_clips):
    clip_path = clips_dir / clip_name

    if not clip_path.exists():
        print(f"⚠ Clip not found: {clip_name}")
        continue

    print(f"\n--- Testing clip {i+1}/{len(test_clips)}: {clip_name} ---")

    # Generate visual embedding
    print("  Generating visual embedding...")
    visual_emb = embedding_gen.encode_video_clip(clip_path, num_frames=4)
    print(f"  ✓ Visual embedding: shape={visual_emb.shape}, norm={visual_emb.dot(visual_emb):.4f}")

    # Enrich with GPT-4V
    print("  Enriching with GPT-4V...")
    metadata = enricher.enrich_segment(clip_path)
    print(f"  ✓ Metadata extracted:")
    print(f"     Motion: {metadata.player_movements}")
    print(f"     Ball zone: {metadata.ball_zone}")
    print(f"     Possible screen: {metadata.possible_screen:.2f}")
    print(f"     Possible drive: {metadata.possible_drive:.2f}")
    print(f"     Possible shot: {metadata.possible_shot:.2f}")

    # Generate trajectory embedding
    print("  Generating trajectory embedding...")
    traj_emb = traj_gen.generate_trajectory_embedding(metadata.player_movements)
    print(f"  ✓ Trajectory embedding: shape={traj_emb.shape}, norm={traj_emb.dot(traj_emb):.4f}")

print("\n" + "=" * 80)
print("Enrichment Test Complete!")
print("=" * 80)

# Print API usage
usage = enricher.get_api_usage()
print(f"\nAPI Usage:")
print(f"  Calls: {usage['total_calls']}")
print(f"  Tokens: {usage['total_tokens']}")
print(f"  Estimated cost: ${usage['estimated_cost']:.4f}")
