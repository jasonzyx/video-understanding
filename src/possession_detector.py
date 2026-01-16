"""Automated possession detection using scene change + activity detection."""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from src.data_models import Possession

logger = logging.getLogger(__name__)


class PossessionDetector:
    """Detects possession-like segments from video using scene and activity analysis."""

    def __init__(
        self,
        scene_threshold: float = 30.0,
        activity_threshold: float = 2.0,
        min_possession_duration: float = 5.0,
        max_possession_duration: float = 30.0,
        merge_gap: float = 3.0,
        sample_rate: float = 0.5
    ):
        """
        Args:
            scene_threshold: Threshold for scene change detection (higher = fewer cuts)
            activity_threshold: Threshold for activity detection (higher = only high motion)
            min_possession_duration: Minimum possession length in seconds
            max_possession_duration: Maximum possession length in seconds
            merge_gap: Merge active periods separated by less than this (seconds)
            sample_rate: Sample every N seconds for analysis
        """
        self.scene_threshold = scene_threshold
        self.activity_threshold = activity_threshold
        self.min_possession_duration = min_possession_duration
        self.max_possession_duration = max_possession_duration
        self.merge_gap = merge_gap
        self.sample_rate = sample_rate

    def detect_possessions(self, video_path: Path, game_id: str) -> List[Possession]:
        """Detect possession boundaries from video.

        Args:
            video_path: Path to video file
            game_id: Game identifier

        Returns:
            List of detected possessions
        """
        logger.info(f"Analyzing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")

        # Step 1: Detect activity periods
        activity_periods = self._detect_activity_periods(cap, fps)
        cap.release()

        logger.info(f"Detected {len(activity_periods)} raw activity periods")

        # Step 2: Merge close periods
        merged_periods = self._merge_periods(activity_periods, self.merge_gap)
        logger.info(f"Merged into {len(merged_periods)} periods")

        # Step 3: Filter by duration
        filtered_periods = [
            (start, end) for start, end in merged_periods
            if (end - start) >= self.min_possession_duration
        ]
        logger.info(f"Filtered to {len(filtered_periods)} periods (min {self.min_possession_duration}s)")

        # Step 4: Split long periods
        possessions = []
        possession_id = 0

        for start, end in filtered_periods:
            duration = end - start

            if duration <= self.max_possession_duration:
                # Single possession
                possessions.append(Possession(
                    game_id=game_id,
                    possession_id=possession_id,
                    start=start,
                    end=end,
                    metadata={'duration': duration, 'auto_detected': True}
                ))
                possession_id += 1
            else:
                # Split into multiple possessions
                num_splits = int(np.ceil(duration / self.max_possession_duration))
                split_duration = duration / num_splits

                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = min(start + (i + 1) * split_duration, end)

                    possessions.append(Possession(
                        game_id=game_id,
                        possession_id=possession_id,
                        start=split_start,
                        end=split_end,
                        metadata={
                            'duration': split_end - split_start,
                            'auto_detected': True,
                            'split_from_long_period': True
                        }
                    ))
                    possession_id += 1

        logger.info(f"✓ Generated {len(possessions)} possessions")
        return possessions

    def _detect_activity_periods(
        self,
        cap: cv2.VideoCapture,
        fps: float
    ) -> List[Tuple[float, float]]:
        """Detect periods of activity in video.

        Returns:
            List of (start_time, end_time) tuples
        """
        frame_skip = int(fps * self.sample_rate)
        if frame_skip < 1:
            frame_skip = 1

        prev_frame = None
        prev_gray = None
        activity_scores = []
        timestamps = []

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / fps

                # Resize for faster processing
                small_frame = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Compute scene change score (frame difference)
                    scene_diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))

                    # Compute activity score (motion energy)
                    flow_score = 0.0
                    if scene_diff < self.scene_threshold:  # Not a scene cut
                        # Simple motion estimate using frame difference
                        motion = cv2.absdiff(gray, prev_gray)
                        flow_score = np.mean(motion)

                    activity_scores.append(flow_score)
                    timestamps.append(timestamp)

                prev_gray = gray
                prev_frame = frame

            frame_idx += 1

        # Find active periods (above threshold)
        activity_scores = np.array(activity_scores)
        timestamps = np.array(timestamps)

        is_active = activity_scores > self.activity_threshold

        # Find contiguous active regions
        periods = []
        start = None

        for i, active in enumerate(is_active):
            if active and start is None:
                start = timestamps[i]
            elif not active and start is not None:
                periods.append((start, timestamps[i]))
                start = None

        # Close final period if needed
        if start is not None:
            periods.append((start, timestamps[-1]))

        return periods

    def _merge_periods(
        self,
        periods: List[Tuple[float, float]],
        gap_threshold: float
    ) -> List[Tuple[float, float]]:
        """Merge periods separated by small gaps.

        Args:
            periods: List of (start, end) tuples
            gap_threshold: Merge if gap is less than this

        Returns:
            Merged periods
        """
        if not periods:
            return []

        # Sort by start time
        periods = sorted(periods, key=lambda x: x[0])

        merged = [periods[0]]

        for start, end in periods[1:]:
            prev_start, prev_end = merged[-1]

            if start - prev_end <= gap_threshold:
                # Merge with previous
                merged[-1] = (prev_start, max(end, prev_end))
            else:
                merged.append((start, end))

        return merged
