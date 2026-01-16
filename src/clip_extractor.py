"""Extract video clips for segments."""
import logging
import subprocess
from pathlib import Path
from typing import List
from src.data_models import Segment

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Extracts video clips for segments using ffmpeg."""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save extracted clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_clip(self, video_path: Path, segment: Segment) -> Path:
        """Extract a single video clip for a segment.

        Args:
            video_path: Source video file
            segment: Segment to extract

        Returns:
            Path to extracted clip
        """
        clip_filename = f"{segment.segment_id}.mp4"
        clip_path = self.output_dir / clip_filename

        # Skip if already exists
        if clip_path.exists():
            return clip_path

        duration = segment.end - segment.start

        # Use ffmpeg to extract clip
        # -ss: start time
        # -t: duration
        # -c copy: copy codec (fast, no re-encoding)
        # -avoid_negative_ts make_zero: ensure timestamps start at 0
        cmd = [
            "ffmpeg",
            "-ss", str(segment.start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y",  # Overwrite output file
            str(clip_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return clip_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract clip {segment.segment_id}: {e.stderr}")
            raise

    def extract_clips_for_segments(
        self,
        video_path: Path,
        segments: List[Segment],
        max_clips: int = None
    ) -> int:
        """Extract clips for multiple segments.

        Args:
            video_path: Source video file
            segments: List of segments to extract
            max_clips: Maximum number of clips to extract (for testing)

        Returns:
            Number of clips extracted
        """
        segments_to_process = segments[:max_clips] if max_clips else segments
        extracted = 0

        logger.info(f"Extracting {len(segments_to_process)} clips to {self.output_dir}")

        for i, segment in enumerate(segments_to_process):
            if (i + 1) % 50 == 0:
                logger.info(f"Extracted {i + 1}/{len(segments_to_process)} clips")

            try:
                self.extract_clip(video_path, segment)
                extracted += 1
            except Exception as e:
                logger.error(f"Failed to extract {segment.segment_id}: {e}")

        logger.info(f"✓ Extracted {extracted} clips")
        return extracted
