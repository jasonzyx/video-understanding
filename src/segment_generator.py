"""Generate candidate segments from possessions."""
import logging
from typing import List
from src.data_models import Possession, Segment

logger = logging.getLogger(__name__)


class SegmentGenerator:
    """Generates multi-scale sliding window segments within possessions."""

    def __init__(self, window_sizes: List[float] = None, stride: float = 1.0):
        """
        Args:
            window_sizes: List of window durations in seconds (e.g., [2, 4, 8])
            stride: Stride between windows in seconds
        """
        self.window_sizes = window_sizes or [2.0, 4.0, 8.0]
        self.stride = stride

    def generate_segments(self, possession: Possession) -> List[Segment]:
        """Generate all candidate segments for a possession.

        Args:
            possession: The possession to segment

        Returns:
            List of candidate segments
        """
        segments = []
        segment_counter = 0
        possession_duration = possession.end - possession.start

        for window_size in self.window_sizes:
            if window_size > possession_duration:
                # Skip windows larger than possession
                continue

            # Generate sliding windows
            current_start = possession.start
            while current_start + window_size <= possession.end:
                segment_id = f"{possession.game_id}_p{possession.possession_id}_s{segment_counter}"

                segment = Segment(
                    game_id=possession.game_id,
                    possession_id=possession.possession_id,
                    segment_id=segment_id,
                    start=current_start,
                    end=current_start + window_size,
                    duration=window_size,
                    metadata={
                        'window_size': window_size,
                        'possession_start': possession.start,
                        'possession_end': possession.end
                    }
                )
                segments.append(segment)
                segment_counter += 1
                current_start += self.stride

        logger.info(
            f"Generated {len(segments)} segments for possession {possession.possession_id} "
            f"(duration: {possession_duration:.1f}s)"
        )
        return segments
