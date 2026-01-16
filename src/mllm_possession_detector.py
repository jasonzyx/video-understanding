"""MLLM-based possession detection using GPT-4 Vision."""
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional
import time
from openai import OpenAI

from src.data_models import Possession
from src.frame_extractor import FrameExtractor, Frame

logger = logging.getLogger(__name__)


class MLLMPossessionDetector:
    """Detects possessions using multimodal LLM (GPT-4 Vision)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        sample_rate: float = 2.5,
        batch_size: int = 6,
        min_possession_duration: float = 5.0,
        max_possession_duration: float = 30.0,
        merge_gap: float = 3.0,
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o, gpt-4-turbo, gpt-4o-mini)
            sample_rate: Sample one frame every N seconds
            batch_size: Number of frames per API request
            min_possession_duration: Minimum possession length in seconds
            max_possession_duration: Maximum possession length in seconds
            merge_gap: Merge active periods separated by less than this (seconds)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.min_possession_duration = min_possession_duration
        self.max_possession_duration = max_possession_duration
        self.merge_gap = merge_gap

        self.frame_extractor = FrameExtractor(sample_rate=sample_rate)

        # Track API usage
        self.total_api_calls = 0
        self.total_tokens = 0

    def detect_possessions(self, video_path: Path, game_id: str) -> List[Possession]:
        """Detect possession boundaries using MLLM classification.

        Args:
            video_path: Path to video file
            game_id: Game identifier

        Returns:
            List of detected possessions
        """
        logger.info(f"MLLM-based possession detection: {video_path}")
        logger.info(f"Model: {self.model}, Sample rate: {self.sample_rate}s, Batch size: {self.batch_size}")

        # Step 1: Extract frames
        frames = self.frame_extractor.extract_frames(video_path)
        logger.info(f"Extracted {len(frames)} frames for classification")

        # Step 2: Classify frames with MLLM
        classifications = self._classify_frames(frames)
        logger.info(f"Classified {len(classifications)} frames")

        # Step 3: Stitch classifications into possession boundaries
        timestamps = [f.timestamp for f in frames]
        possessions = self._stitch_to_possessions(
            classifications, timestamps, game_id
        )

        logger.info(f"✓ Generated {len(possessions)} possessions")
        logger.info(f"API usage: {self.total_api_calls} calls, ~{self.total_tokens} tokens")

        return possessions

    def _classify_frames(self, frames: List[Frame]) -> List[str]:
        """Classify each frame (A=active, B=transition, C=inactive).

        Args:
            frames: List of frames to classify

        Returns:
            List of classifications ("A", "B", or "C")
        """
        batches = self.frame_extractor.batch_frames(frames, self.batch_size)
        logger.info(f"Processing {len(batches)} batches")

        all_classifications = []

        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} frames)")

            try:
                batch_results = self._classify_frames_batch(batch)
                all_classifications.extend(batch_results)
            except Exception as e:
                logger.error(f"Error classifying batch {batch_idx}: {e}")
                # Fallback: assume inactive on error
                all_classifications.extend(["C"] * len(batch))

            # Rate limiting courtesy delay
            if batch_idx < len(batches) - 1:
                time.sleep(0.5)

        return all_classifications

    def _classify_frames_batch(self, frames: List[Frame]) -> List[str]:
        """Classify a batch of frames using GPT-4 Vision.

        Args:
            frames: Batch of frames to classify

        Returns:
            List of classifications ("A", "B", or "C")
        """
        # Build message content with multiple images
        content = [
            {
                "type": "text",
                "text": self._get_classification_prompt(len(frames))
            }
        ]

        # Add images to content
        for idx, frame in enumerate(frames):
            base64_img = self.frame_extractor.encode_frame_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "low"  # Use low detail for cost optimization
                }
            })

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=300,
            temperature=0.0
        )

        self.total_api_calls += 1
        self.total_tokens += response.usage.total_tokens

        # Parse response
        response_text = response.choices[0].message.content
        classifications = self._parse_classifications(response_text, len(frames))

        return classifications

    def _get_classification_prompt(self, num_frames: int) -> str:
        """Generate prompt for frame classification.

        Args:
            num_frames: Number of frames in batch

        Returns:
            Prompt string
        """
        prompt = f"""You are analyzing basketball game footage. I will show you {num_frames} frames IN SEQUENCE.

For each frame, classify it into ONE of these categories:

**A) ACTIVE** - Active gameplay, SAME TEAM still in possession
  - Team has ball and is playing (passing, dribbling, setting up offense)
  - Offensive possession continuing
  - No change in which team has the ball

**B) TRANSITION** - Possession change happening or just happened
  - Shot attempt followed by defensive rebound (opponent gets ball)
  - Turnover visible (steal, bad pass intercepted, ball going to other team)
  - Fast break starting after defensive rebound
  - Players clearly transitioning between offense and defense
  - One team scores and opponent inbounds the ball

**C) INACTIVE** - Not active gameplay
  - Timeout, stoppage, whistle blown
  - Replay or slow-motion replay
  - Commentary shot, crowd shot
  - Halftime, pre-game, post-game
  - Transition graphics or commercials

IMPORTANT: If you see a shot attempt followed by a rebound, or any clear change in which team has the ball, classify as TRANSITION (B).

Respond in JSON format:
{{
  "frame_0": "A" or "B" or "C",
  "frame_1": "A" or "B" or "C",
  ...
  "frame_{num_frames - 1}": "A" or "B" or "C"
}}

Only output JSON, no additional text."""

        return prompt

    def _parse_classifications(self, response_text: str, expected_count: int) -> List[str]:
        """Parse JSON response from MLLM into classification list.

        Args:
            response_text: JSON response from API
            expected_count: Expected number of classifications

        Returns:
            List of classifications ("A", "B", or "C")
        """
        try:
            # Try to extract JSON from response
            # Sometimes the model adds markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            # Convert to classification list
            classifications = []
            for i in range(expected_count):
                key = f"frame_{i}"
                value = data.get(key, "C").upper()
                # Ensure valid values
                if value not in ["A", "B", "C"]:
                    value = "C"  # Default to inactive
                classifications.append(value)

            return classifications

        except Exception as e:
            logger.error(f"Error parsing classifications: {e}")
            logger.error(f"Response text: {response_text}")
            # Fallback: assume inactive
            return ["C"] * expected_count

    def _stitch_to_possessions(
        self,
        classifications: List[str],
        timestamps: List[float],
        game_id: str
    ) -> List[Possession]:
        """Stitch frame classifications into possession boundaries.

        Args:
            classifications: List of classifications ("A", "B", or "C")
            timestamps: Corresponding timestamps for each classification
            game_id: Game identifier

        Returns:
            List of Possession objects
        """
        # Find contiguous active periods, split at transitions
        periods = []
        start = None

        for i, classification in enumerate(classifications):
            if classification == "A":  # Active gameplay
                if start is None:
                    start = timestamps[i]
            elif classification == "B":  # Transition - end current possession
                if start is not None:
                    periods.append((start, timestamps[i]))
                    start = None
            else:  # "C" - Inactive
                if start is not None:
                    periods.append((start, timestamps[i]))
                    start = None

        # Close final period if needed
        if start is not None:
            periods.append((start, timestamps[-1]))

        logger.info(f"Found {len(periods)} raw active periods")

        # Merge close periods
        merged_periods = self._merge_periods(periods)
        logger.info(f"Merged into {len(merged_periods)} periods")

        # Filter by duration
        filtered_periods = [
            (start, end) for start, end in merged_periods
            if (end - start) >= self.min_possession_duration
        ]
        logger.info(f"Filtered to {len(filtered_periods)} periods (min {self.min_possession_duration}s)")

        # Convert to Possession objects
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
                    metadata={
                        'duration': duration,
                        'auto_detected': True,
                        'method': 'mllm',
                        'model': self.model
                    }
                ))
                possession_id += 1
            else:
                # Split long periods
                import numpy as np
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
                            'method': 'mllm',
                            'model': self.model,
                            'split_from_long_period': True
                        }
                    ))
                    possession_id += 1

        return possessions

    def _merge_periods(
        self,
        periods: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge periods separated by small gaps.

        Args:
            periods: List of (start, end) tuples

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

            if start - prev_end <= self.merge_gap:
                # Merge with previous
                merged[-1] = (prev_start, max(end, prev_end))
            else:
                merged.append((start, end))

        return merged
