#!/usr/bin/env python3
"""Extract video clips for existing segments."""
import argparse
import logging
import json
from pathlib import Path
import jsonlines

from src.data_models import Segment
from src.clip_extractor import ClipExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Extract video clips for segments')
    parser.add_argument('--game-id', required=True, help='Game ID')
    parser.add_argument('--output-dir', default='outputs', help='Directory containing segments')
    parser.add_argument('--clips-dir', default='outputs/clips', help='Directory to save clips')
    parser.add_argument('--max-clips', type=int, help='Maximum number of clips to extract (for testing)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    game_id = args.game_id

    # Load metadata to get video path
    metadata_file = output_dir / f"{game_id}_index_metadata.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return 1

    with open(metadata_file) as f:
        metadata = json.load(f)

    video_path = Path(metadata['video_path'])
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    # Load segments
    segments_file = output_dir / f"{game_id}_segments.jsonl"
    if not segments_file.exists():
        logger.error(f"Segments file not found: {segments_file}")
        return 1

    with jsonlines.open(segments_file) as reader:
        segments = [Segment.from_dict(seg) for seg in reader]

    logger.info(f"Loaded {len(segments)} segments")

    # Extract clips
    clips_dir = Path(args.clips_dir)
    extractor = ClipExtractor(clips_dir)

    num_extracted = extractor.extract_clips_for_segments(
        video_path,
        segments,
        max_clips=args.max_clips
    )

    # Update metadata
    metadata['clips_extracted'] = True
    metadata['clips_dir'] = str(clips_dir.absolute())

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Extracted {num_extracted} clips")
    logger.info(f"✓ Updated metadata: {metadata_file}")

    return 0


if __name__ == '__main__':
    exit(main())
