#!/usr/bin/env python3
"""Quick test script for possession detection."""
import argparse
import logging
import os
from pathlib import Path
from src.possession_detector import PossessionDetector
from src.mllm_possession_detector import MLLMPossessionDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test possession detection on a video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--min-possession', type=float, default=5.0)
    parser.add_argument('--max-possession', type=float, default=30.0)
    parser.add_argument('--activity-threshold', type=float, default=2.0)
    parser.add_argument('--scene-threshold', type=float, default=30.0)

    # MLLM options
    parser.add_argument('--use-mllm', action='store_true',
                       help='Use MLLM-based detection (GPT-4V)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--mllm-model', default='gpt-4o',
                       help='MLLM model to use')
    parser.add_argument('--mllm-sample-rate', type=float, default=2.5,
                       help='Sample rate for MLLM (seconds)')
    parser.add_argument('--mllm-batch-size', type=int, default=6,
                       help='Frames per API request')

    args = parser.parse_args()

    video_path = Path(args.video)
    game_id = video_path.stem

    logger.info(f"Testing possession detection on: {video_path}")

    if args.use_mllm:
        logger.info("Using MLLM-based detection")
        logger.info(f"Parameters:")
        logger.info(f"  Model: {args.mllm_model}")
        logger.info(f"  Sample rate: {args.mllm_sample_rate}s")
        logger.info(f"  Batch size: {args.mllm_batch_size}")
        logger.info(f"  Min possession: {args.min_possession}s")
        logger.info(f"  Max possession: {args.max_possession}s")

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
    else:
        logger.info("Using basic OpenCV detection")
        logger.info(f"Parameters:")
        logger.info(f"  Min possession: {args.min_possession}s")
        logger.info(f"  Max possession: {args.max_possession}s")
        logger.info(f"  Activity threshold: {args.activity_threshold}")
        logger.info(f"  Scene threshold: {args.scene_threshold}")

        detector = PossessionDetector(
            min_possession_duration=args.min_possession,
            max_possession_duration=args.max_possession,
            activity_threshold=args.activity_threshold,
            scene_threshold=args.scene_threshold
        )

    possessions = detector.detect_possessions(video_path, game_id)

    logger.info("\n" + "="*60)
    logger.info(f"RESULTS: Detected {len(possessions)} possessions")
    logger.info("="*60)

    for poss in possessions:
        duration = poss.end - poss.start
        logger.info(f"  [{poss.possession_id:2d}] {poss.start:7.2f}s - {poss.end:7.2f}s  (duration: {duration:5.2f}s)")

    total_coverage = sum(p.end - p.start for p in possessions)
    logger.info("="*60)
    logger.info(f"Total coverage: {total_coverage:.1f}s")


if __name__ == '__main__':
    main()
