#!/usr/bin/env python3
"""Test expanded event taxonomy on sample segments."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.segment_enricher import SegmentEnricher

def test_expanded_events():
    """Test enrichment with expanded event taxonomy."""
    print("=" * 80)
    print("TESTING EXPANDED EVENT TAXONOMY")
    print("=" * 80)

    # Initialize enricher
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    enricher = SegmentEnricher(api_key=api_key, model='gpt-4o')

    # Test on 3 sample clips
    test_clips = [
        "outputs/clips/nba_final_p0_s0.mp4",
        "outputs/clips/nba_final_p0_s1.mp4",
        "outputs/clips/nba_final_p0_s2.mp4"
    ]

    for clip_path_str in test_clips:
        clip_path = Path(clip_path_str)

        if not clip_path.exists():
            print(f"⚠ Skipping {clip_path.name} - not found")
            continue

        print(f"\n{'='*80}")
        print(f"Testing: {clip_path.name}")
        print(f"{'='*80}")

        try:
            metadata = enricher.enrich_segment(clip_path)

            print("\n📊 ENRICHMENT RESULTS:")
            print(f"\nMotion: {metadata.player_movements}")
            print(f"Intensity: {metadata.motion_intensity:.2f}")

            print("\n🏀 COURT SEMANTICS:")
            print(f"  Ball zone: {metadata.ball_zone}")
            print(f"  Primary zone: {metadata.primary_zone}")
            print(f"  Paint occupied: {metadata.paint_occupied}")

            print("\n📈 DERIVED METRICS:")
            print(f"  Ball speed: {metadata.ball_speed_estimate}")
            print(f"  Player proximity: {metadata.player_ball_proximity}")
            print(f"  Spacing: {metadata.offensive_spacing}")
            print(f"  Screen angle: {metadata.screen_angle_est}")

            # Basic events
            print("\n⚡ BASIC EVENTS:")
            basic_events = [
                ('Screen', metadata.possible_screen),
                ('Drive', metadata.possible_drive),
                ('Shot', metadata.possible_shot),
                ('Pass', metadata.possible_pass),
                ('Rebound', metadata.possible_rebound)
            ]
            for name, prob in basic_events:
                bar = '█' * int(prob * 20)
                print(f"  {name:10s} [{bar:20s}] {prob:.2f}")

            # Advanced events
            print("\n🎯 ADVANCED EVENTS:")
            advanced_events = [
                ('Steal', metadata.possible_steal),
                ('Block', metadata.possible_block),
                ('Cut', metadata.possible_cut),
                ('Post Up', metadata.possible_post_up),
                ('Pick & Roll', metadata.possible_pick_and_roll),
                ('Fast Break', metadata.possible_fast_break),
                ('Def. Rotation', metadata.possible_defensive_rotation),
                ('Assist', metadata.possible_assist),
                ('Turnover', metadata.possible_turnover),
                ('Dribble Move', metadata.possible_dribble_move)
            ]
            for name, prob in advanced_events:
                bar = '█' * int(prob * 20)
                print(f"  {name:14s} [{bar:20s}] {prob:.2f}")

            print(f"\n  Rebound type: {metadata.rebound_type}")

            # Highlight high confidence events (>0.5)
            all_events = basic_events + advanced_events
            high_conf = [(name, prob) for name, prob in all_events if prob > 0.5]
            if high_conf:
                print("\n✅ HIGH CONFIDENCE DETECTIONS (>0.5):")
                for name, prob in sorted(high_conf, key=lambda x: -x[1]):
                    print(f"  • {name}: {prob:.2f}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Show API usage
    usage = enricher.get_api_usage()
    print(f"\n{'='*80}")
    print("📊 API USAGE")
    print(f"{'='*80}")
    print(f"Total API calls: {usage['total_calls']}")
    print(f"Total tokens: {usage['total_tokens']:,}")
    print(f"Estimated cost: ${usage['estimated_cost']:.2f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_expanded_events()
