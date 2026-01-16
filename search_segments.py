#!/usr/bin/env python3
"""CLI tool for multi-modal segment search with score breakdown."""
import argparse
import logging
from pathlib import Path
from tabulate import tabulate

from src.multimodal_retrieval_engine import MultiModalRetrievalEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description='Multi-modal segment search')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--game-id', default='nba_final', help='Game ID')
    parser.add_argument('--output-dir', default='outputs', help='Output directory with indices')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--show-scores', action='store_true',
                       help='Show detailed score breakdown')
    parser.add_argument('--show-metadata', action='store_true',
                       help='Show enriched metadata for each result')

    # Custom fusion weights
    parser.add_argument('--visual-weight', type=float, help='Visual score weight')
    parser.add_argument('--trajectory-weight', type=float, help='Trajectory score weight')
    parser.add_argument('--events-weight', type=float, help='Events score weight')
    parser.add_argument('--metadata-weight', type=float, help='Metadata score weight')

    # Filters
    parser.add_argument('--min-duration', type=float, help='Minimum segment duration')
    parser.add_argument('--max-duration', type=float, help='Maximum segment duration')
    parser.add_argument('--min-event-score', type=float, help='Minimum event score threshold')

    parser.add_argument('--device', default='cpu', help='Device for CLIP (cpu, cuda)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    game_id = args.game_id

    # Check if indices exist
    segments_file = output_dir / f"{game_id}_segments.jsonl"
    visual_index = output_dir / f"{game_id}_visual_index.bin"
    trajectory_index = output_dir / f"{game_id}_trajectory_index.bin"
    mapping_file = output_dir / f"{game_id}_index_mapping.json"

    if not all([segments_file.exists(), visual_index.exists(), trajectory_index.exists(), mapping_file.exists()]):
        logger.error("Indices not found. Run build_index.py with --enrich-segments --generate-embeddings first.")
        return 1

    # Build custom weights if provided
    custom_weights = None
    if any([args.visual_weight, args.trajectory_weight, args.events_weight, args.metadata_weight]):
        custom_weights = {
            'visual': args.visual_weight or 0.4,
            'trajectory': args.trajectory_weight or 0.3,
            'events': args.events_weight or 0.2,
            'metadata': args.metadata_weight or 0.1
        }

    # Build filters
    filters = {}
    if args.min_duration:
        filters['min_duration'] = args.min_duration
    if args.max_duration:
        filters['max_duration'] = args.max_duration
    if args.min_event_score:
        filters['min_event_score'] = args.min_event_score

    # Initialize retrieval engine
    logger.info("Initializing multi-modal retrieval engine...")
    engine = MultiModalRetrievalEngine(
        segments_path=segments_file,
        visual_index_path=visual_index,
        trajectory_index_path=trajectory_index,
        mapping_path=mapping_file,
        weights=custom_weights,
        device=args.device
    )

    # Get engine stats
    stats = engine.get_stats()
    logger.info(f"Loaded {stats['num_segments']} segments, {stats['num_indexed']} indexed vectors")
    logger.info(f"Fusion weights: {stats['weights']}")

    # Execute search
    logger.info(f"Searching for: '{args.query}'")
    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        filters=filters if filters else None
    )

    if not results:
        print("\n❌ No results found")
        return 0

    # Display results
    print("\n" + "=" * 100)
    print(f"SEARCH RESULTS: '{args.query}'")
    print("=" * 100)

    if args.show_scores:
        # Detailed view with score breakdown
        table_data = []
        for result in results:
            table_data.append([
                result.rank,
                result.segment_id,
                f"{result.combined_score:.3f}",
                f"{result.visual_score:.2f}",
                f"{result.trajectory_score:.2f}",
                f"{result.events_score:.2f}",
                f"{result.metadata_score:.2f}",
                format_time(result.segment.start),
                f"{result.segment.duration:.1f}s"
            ])

        headers = ["Rank", "Segment ID", "Combined", "Visual", "Traj", "Events", "Meta", "Time", "Dur"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    else:
        # Simple view
        table_data = []
        for result in results:
            table_data.append([
                result.rank,
                result.segment_id,
                f"{result.combined_score:.3f}",
                format_time(result.segment.start),
                f"{result.segment.duration:.1f}s"
            ])

        headers = ["Rank", "Segment ID", "Score", "Time", "Duration"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))

    # Show metadata for top results
    if args.show_metadata:
        print("\n" + "=" * 100)
        print("METADATA FOR TOP RESULTS")
        print("=" * 100)

        for result in results[:3]:  # Show top 3
            segment = result.segment
            metadata = segment.metadata

            if not metadata:
                continue

            print(f"\n{'─' * 100}")
            print(f"Rank #{result.rank}: {segment.segment_id} (Score: {result.combined_score:.3f})")
            print(f"Time: {format_time(segment.start)} - {format_time(segment.end)} ({segment.duration:.1f}s)")
            print(f"{'─' * 100}")

            # Motion
            derived = metadata.get('derived', {})
            if 'motion_intensity' in derived:
                print(f"\n📍 Motion Intensity: {derived['motion_intensity']:.2f}")

            # Court semantics
            semantics = metadata.get('court_semantics', {})
            if semantics:
                print(f"\n🏀 Court Semantics:")
                print(f"  Ball zone: {semantics.get('ball_zone', 'N/A')}")
                print(f"  Primary zone: {semantics.get('primary_zone', 'N/A')}")
                print(f"  Paint occupied: {semantics.get('paint_occupied', 'N/A')}")

            # Derived metrics
            if derived:
                print(f"\n📊 Metrics:")
                print(f"  Ball speed: {derived.get('ball_speed_estimate', 'N/A')}")
                print(f"  Spacing: {derived.get('offensive_spacing', 'N/A')}")
                if derived.get('screen_angle_est') and derived.get('screen_angle_est') != 'none':
                    print(f"  Screen angle: {derived.get('screen_angle_est')}")

            # Events
            events = metadata.get('weak_events', {})
            if events:
                high_confidence = [(k, v) for k, v in events.items() if v > 0.5]
                if high_confidence:
                    print(f"\n⚡ High Confidence Events:")
                    for event, prob in sorted(high_confidence, key=lambda x: -x[1]):
                        event_name = event.replace('possible_', '').title()
                        print(f"  {event_name}: {prob:.2f}")

    # Summary
    print("\n" + "=" * 100)
    print(f"Found {len(results)} results")
    if filters:
        print(f"Filters applied: {filters}")
    print("=" * 100 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
