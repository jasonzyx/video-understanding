#!/usr/bin/env python3
"""Test just the reranker without loading full engine."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ ERROR: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in .env file")
    exit(1)

print(f"✓ Found API key: {api_key[:10]}...")

# Check for clips directory
clips_dir = Path('outputs/clips')
if not clips_dir.exists():
    print(f"\n❌ ERROR: Clips directory not found: {clips_dir}")
    print("Please extract clips first with:")
    print("  python build_index.py --video data/nba_final.mp4 --extract-clips")
    exit(1)

# Get first few clips
clip_files = sorted(clips_dir.glob('*.mp4'))[:3]
if not clip_files:
    print(f"\n❌ ERROR: No clips found in {clips_dir}")
    exit(1)

print(f"✓ Found {len(clip_files)} test clips")

print("\n" + "="*70)
print("Testing Gemini Reranker")
print("="*70)

try:
    from src.gemini_flash_reranker import GeminiFlashReranker
    from src.data_models import Segment

    print("\n[1/2] Initializing Gemini reranker...")
    reranker = GeminiFlashReranker(api_key=api_key)
    print("    ✓ Reranker loaded")

    print("\n[2/2] Testing reranking with query 'corner three pointer'...")

    # Create dummy segments for the test clips
    test_segments = []
    for i, clip_path in enumerate(clip_files):
        segment = Segment(
            game_id="nba_final",
            segment_id=clip_path.stem,
            start=float(i * 10),
            end=float(i * 10 + 5),
            duration=5.0,
            possession_id=i,
            metadata={
                'weak_events': {'possible_shot': 0.5},
                'court_semantics': {'ball_zone': 'corner'}
            }
        )
        test_segments.append(segment)

    print(f"    Testing with {len(test_segments)} clips...")

    # Rerank
    results = reranker.rerank(
        query="corner three pointer",
        segments=test_segments,
        clip_dir=clips_dir,
        top_k=3
    )

    print(f"\n    ✓ Reranking complete! Found {len(results)} relevant results")

    print("\n    Reranked Results:")
    for i, r in enumerate(results, 1):
        print(f"\n      {i}. {r.segment_id}")
        print(f"         Relevance Score: {r.relevance_score:.1f}/10")
        print(f"         Is Relevant: {r.is_relevant}")
        print(f"         Explanation: {r.explanation}")

    # Get API usage stats
    usage = reranker.get_api_usage()
    print(f"\n    API Usage:")
    print(f"      Total calls: {usage['total_calls']}")
    print(f"      Estimated cost: ${usage['estimated_cost']:.4f}")

    print("\n" + "="*70)
    print("✅ SUCCESS: Gemini reranker is working with your GEMINI_API_KEY!")
    print("="*70)
    print("\nYou can now use the viewer with reranking enabled:")
    print("  streamlit run viewer/app.py")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
