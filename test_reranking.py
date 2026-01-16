#!/usr/bin/env python3
"""Quick test of reranking functionality."""
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

# Check for required files
output_dir = Path('outputs')
enriched_segments = output_dir / 'nba_final_segments_enriched.jsonl'
visual_index = output_dir / 'nba_final_visual_index.bin'
trajectory_index = output_dir / 'nba_final_trajectory_index.bin'
mapping = output_dir / 'nba_final_index_mapping.json'
clips_dir = output_dir / 'clips'

missing = []
if not enriched_segments.exists():
    missing.append(str(enriched_segments))
if not visual_index.exists():
    missing.append(str(visual_index))
if not trajectory_index.exists():
    missing.append(str(trajectory_index))
if not mapping.exists():
    missing.append(str(mapping))
if not clips_dir.exists():
    missing.append(str(clips_dir))

if missing:
    print("\n❌ ERROR: Missing required files:")
    for f in missing:
        print(f"  - {f}")
    print("\nPlease run enrichment first:")
    print("  python build_index.py --video data/nba_final.mp4 --extract-clips --enrich-segments --generate-embeddings")
    exit(1)

print("✓ All required files found")

# Test loading the retrieval engine
print("\n" + "="*70)
print("Testing Reranking System")
print("="*70)

try:
    from src.multimodal_retrieval_engine import MultiModalRetrievalEngine

    print("\n[1/3] Loading retrieval engine WITHOUT reranking...")
    engine_base = MultiModalRetrievalEngine(
        segments_path=enriched_segments,
        visual_index_path=visual_index,
        trajectory_index_path=trajectory_index,
        mapping_path=mapping,
        device='cpu',
        enable_reranking=False
    )
    print("    ✓ Base engine loaded")

    print("\n[2/3] Testing search WITHOUT reranking...")
    query = "corner three pointer"
    results_base = engine_base.search(query, top_k=5)
    print(f"    ✓ Found {len(results_base)} results")

    print("\n    Top 3 results (Stage 1 only):")
    for i, r in enumerate(results_base[:3], 1):
        shot_prob = r.segment.metadata.get('weak_events', {}).get('possible_shot', 0) if r.segment.metadata else 0
        zone = r.segment.metadata.get('court_semantics', {}).get('ball_zone', 'unknown') if r.segment.metadata else 'unknown'
        print(f"      {i}. {r.segment_id}")
        print(f"         Combined Score: {r.combined_score:.3f}")
        print(f"         Shot Probability: {shot_prob:.2f}")
        print(f"         Ball Zone: {zone}")

    print("\n[3/3] Loading retrieval engine WITH Gemini reranking...")
    engine_rerank = MultiModalRetrievalEngine(
        segments_path=enriched_segments,
        visual_index_path=visual_index,
        trajectory_index_path=trajectory_index,
        mapping_path=mapping,
        device='cpu',
        enable_reranking=True,
        reranker_type='gemini',
        reranker_api_key=api_key,
        clip_dir=clips_dir
    )
    print("    ✓ Reranking engine loaded")

    print(f"\n[4/4] Testing search WITH Gemini reranking (this will take ~10-15s)...")
    print(f"    Query: '{query}'")
    results_rerank = engine_rerank.search(query, top_k=5)
    print(f"    ✓ Found {len(results_rerank)} relevant results")

    print("\n    Top 3 results (Stage 1 + Stage 2 Gemini):")
    for i, r in enumerate(results_rerank[:3], 1):
        shot_prob = r.segment.metadata.get('weak_events', {}).get('possible_shot', 0) if r.segment.metadata else 0
        zone = r.segment.metadata.get('court_semantics', {}).get('ball_zone', 'unknown') if r.segment.metadata else 'unknown'
        print(f"      {i}. {r.segment_id}")
        print(f"         Rerank Score: {r.rerank_score:.1f}/10")
        print(f"         Stage 1 Score: {r.combined_score:.3f}")
        print(f"         Shot Probability: {shot_prob:.2f}")
        print(f"         Ball Zone: {zone}")
        explanation = r.segment.metadata.get('rerank_explanation', 'No explanation') if r.segment.metadata else 'No explanation'
        print(f"         AI Explanation: {explanation}")

    # Get API usage stats
    usage = engine_rerank.reranker.get_api_usage()
    print(f"\n    API Usage:")
    print(f"      Total calls: {usage['total_calls']}")
    print(f"      Estimated cost: ${usage['estimated_cost']:.4f}")

    print("\n" + "="*70)
    print("✅ SUCCESS: Reranking system is working!")
    print("="*70)
    print("\nYou can now use the viewer with reranking enabled:")
    print("  streamlit run viewer/app.py")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
