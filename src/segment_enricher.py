"""Segment enrichment using GPT-4V for metadata extraction."""
import logging
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import cv2
from openai import OpenAI

from src.frame_extractor import FrameExtractor, Frame

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetadata:
    """Enriched metadata for a segment."""
    # Trajectory/motion
    player_movements: str  # GPT-4V description
    motion_intensity: float  # 0-1

    # Court semantics
    ball_zone: str  # "top_of_key", "paint", "corner", etc.
    primary_zone: str  # Where action happens
    paint_occupied: bool

    # Derived metrics
    ball_speed_estimate: str  # "slow", "medium", "fast"
    player_ball_proximity: str  # "close", "medium", "far"
    offensive_spacing: str  # "tight", "spread"
    screen_angle_est: Optional[str]  # "high", "side", "back", "none"

    # Weak event detection (0-1 probabilities)
    # Basic events
    possible_screen: float
    possible_drive: float
    possible_shot: float
    possible_pass: float
    possible_rebound: float

    # Advanced events
    possible_steal: float
    possible_block: float
    possible_cut: float  # Backdoor, baseline cuts
    possible_post_up: float
    possible_pick_and_roll: float  # Combined screen + roll action
    possible_fast_break: float  # Transition play
    possible_defensive_rotation: float
    possible_assist: float  # Distinguished from general pass
    possible_turnover: float
    possible_dribble_move: float  # Crossovers, behind-back

    # Rebound classification
    rebound_type: Optional[str]  # "offensive", "defensive", "none"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentMetadata':
        """Create from dictionary."""
        return cls(**data)


class SegmentEnricher:
    """Extract rich metadata using GPT-4V multi-frame analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        sample_rate: float = 0.5
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o, gpt-4-turbo, gpt-4o-mini)
            sample_rate: Extract one frame every N seconds from clip
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.sample_rate = sample_rate
        self.frame_extractor = FrameExtractor(sample_rate=sample_rate)

        # Track API usage
        self.total_api_calls = 0
        self.total_tokens = 0

    def enrich_segment(self, clip_path: Path) -> SegmentMetadata:
        """
        Extract metadata from segment clip.

        Strategy:
        - Extract 3-6 frames from clip (based on duration)
        - Send frames to GPT-4V with structured prompt
        - Parse JSON response into SegmentMetadata

        Args:
            clip_path: Path to video clip file

        Returns:
            SegmentMetadata with enriched information
        """
        # Extract frames from clip
        frames = self._extract_frames_from_clip(clip_path)

        if not frames:
            logger.warning(f"No frames extracted from {clip_path}")
            return self._get_default_metadata()

        # Query GPT-4V for metadata
        try:
            metadata_dict = self._query_gpt4v(frames)
            metadata = SegmentMetadata.from_dict(metadata_dict)
            return metadata
        except Exception as e:
            logger.error(f"Error enriching segment {clip_path}: {e}")
            return self._get_default_metadata()

    def _extract_frames_from_clip(self, clip_path: Path) -> List[Frame]:
        """
        Extract frames from clip based on duration.

        Frame sampling strategy:
        - 2s clips: 3 frames (0s, 1s, 2s)
        - 4s clips: 4 frames (0s, 1.3s, 2.7s, 4s)
        - 8s clips: 6 frames (0s, 1.6s, 3.2s, 4.8s, 6.4s, 8s)

        Args:
            clip_path: Path to video clip

        Returns:
            List of Frame objects
        """
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.error(f"Cannot open clip: {clip_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Determine target number of frames based on duration
        if duration <= 0:
            cap.release()
            return []
        elif duration <= 2.5:
            target_frames = 3
        elif duration <= 5.0:
            target_frames = 4
        else:
            target_frames = 6

        # Calculate frame indices to extract (evenly spaced)
        if target_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [
                int(i * (total_frames - 1) / (target_frames - 1))
                for i in range(target_frames)
            ]

        # Extract frames
        frames = []
        current_idx = 0

        while True:
            ret, frame_img = cap.read()
            if not ret:
                break

            if current_idx in frame_indices:
                timestamp = current_idx / fps if fps > 0 else 0
                frames.append(Frame(
                    timestamp=timestamp,
                    frame_index=current_idx,
                    image=frame_img
                ))

            current_idx += 1

        cap.release()
        return frames

    def _query_gpt4v(self, frames: List[Frame]) -> dict:
        """
        Query GPT-4V for metadata extraction.

        Args:
            frames: List of frames to analyze

        Returns:
            Dictionary with metadata fields
        """
        # Build message content with multiple images
        content = [
            {
                "type": "text",
                "text": self._get_enrichment_prompt(len(frames))
            }
        ]

        # Add frames
        for frame in frames:
            base64_img = self.frame_extractor.encode_frame_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "low"  # Cost optimization
                }
            })

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=500,
            temperature=0.0
        )

        self.total_api_calls += 1
        self.total_tokens += response.usage.total_tokens

        # Parse response
        response_text = response.choices[0].message.content
        metadata = self._parse_metadata(response_text)

        return metadata

    def _get_enrichment_prompt(self, num_frames: int) -> str:
        """Generate structured metadata extraction prompt."""
        return f"""You are analyzing a basketball video segment. I will show you {num_frames} frames IN SEQUENCE.

Extract the following metadata:

**1. MOTION/TRAJECTORY**
- Describe player movements (e.g., "player cuts to basket", "screen set at high post", "defender rotates")
- Motion intensity: slow (0.2), moderate (0.5), fast (0.8), very fast (1.0)

**2. COURT SEMANTICS**
- Ball zone: top_of_key, paint, left_wing, right_wing, left_corner, right_corner, mid_range
- Primary action zone: where main action occurs
- Paint occupied: true/false (are players in the paint?)

**3. DERIVED METRICS** (best estimates)
- Ball speed: slow, medium, fast
- Player-ball proximity: close (<3ft), medium (3-10ft), far (>10ft)
- Offensive spacing: tight (<12ft between players), spread (>12ft)
- Screen angle (if screen visible): high, side, back, none

**4. COMPREHENSIVE EVENT DETECTION** (probability 0.0-1.0)
Rate likelihood of these actions occurring:

*Basic Actions:*
- possible_screen: Is a screen being set?
- possible_drive: Is a player driving to basket?
- possible_shot: Is a shot attempt happening?
- possible_pass: Is a pass happening?
- possible_rebound: Is a rebound happening?

*Advanced Actions:*
- possible_steal: Is a steal/turnover occurring?
- possible_block: Is a shot being blocked?
- possible_cut: Is a player cutting (backdoor, baseline)?
- possible_post_up: Is a post-up move happening?
- possible_pick_and_roll: Is a pick-and-roll action (screen + roll)?
- possible_fast_break: Is this a fast break/transition play?
- possible_defensive_rotation: Is a defensive rotation happening?
- possible_assist: Is this a pass leading to a score (not just any pass)?
- possible_turnover: Is a turnover happening?
- possible_dribble_move: Is a dribble move (crossover, behind-back) happening?

*Rebound Classification:*
- rebound_type: If rebound detected, classify as "offensive", "defensive", or "none"

Respond in JSON format:
{{
  "player_movements": "<description>",
  "motion_intensity": <0.0-1.0>,
  "ball_zone": "<zone>",
  "primary_zone": "<zone>",
  "paint_occupied": <true/false>,
  "ball_speed_estimate": "<slow/medium/fast>",
  "player_ball_proximity": "<close/medium/far>",
  "offensive_spacing": "<tight/spread>",
  "screen_angle_est": "<high/side/back/none>",
  "possible_screen": <0.0-1.0>,
  "possible_drive": <0.0-1.0>,
  "possible_shot": <0.0-1.0>,
  "possible_pass": <0.0-1.0>,
  "possible_rebound": <0.0-1.0>,
  "possible_steal": <0.0-1.0>,
  "possible_block": <0.0-1.0>,
  "possible_cut": <0.0-1.0>,
  "possible_post_up": <0.0-1.0>,
  "possible_pick_and_roll": <0.0-1.0>,
  "possible_fast_break": <0.0-1.0>,
  "possible_defensive_rotation": <0.0-1.0>,
  "possible_assist": <0.0-1.0>,
  "possible_turnover": <0.0-1.0>,
  "possible_dribble_move": <0.0-1.0>,
  "rebound_type": "<offensive/defensive/none>"
}}

Only output JSON."""

    def _parse_metadata(self, response_text: str) -> dict:
        """
        Parse metadata from GPT-4V response.

        Args:
            response_text: Raw response from API

        Returns:
            Dictionary with metadata fields
        """
        try:
            # Try to extract JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            response_text = response_text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            metadata = json.loads(response_text)

            # Validate required fields
            required_fields = [
                'player_movements', 'motion_intensity', 'ball_zone',
                'primary_zone', 'paint_occupied', 'ball_speed_estimate',
                'player_ball_proximity', 'offensive_spacing', 'screen_angle_est',
                'possible_screen', 'possible_drive', 'possible_shot',
                'possible_pass', 'possible_rebound',
                'possible_steal', 'possible_block', 'possible_cut',
                'possible_post_up', 'possible_pick_and_roll', 'possible_fast_break',
                'possible_defensive_rotation', 'possible_assist', 'possible_turnover',
                'possible_dribble_move', 'rebound_type'
            ]

            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Missing field in metadata: {field}")
                    # Add default value
                    if field == 'player_movements':
                        metadata[field] = "unknown"
                    elif field == 'motion_intensity':
                        metadata[field] = 0.5
                    elif field in ['ball_zone', 'primary_zone']:
                        metadata[field] = "unknown"
                    elif field == 'paint_occupied':
                        metadata[field] = False
                    elif field in ['ball_speed_estimate', 'player_ball_proximity', 'offensive_spacing']:
                        metadata[field] = "medium"
                    elif field in ['screen_angle_est', 'rebound_type']:
                        metadata[field] = "none"
                    else:
                        # All event probabilities default to 0.0
                        metadata[field] = 0.0

            return metadata

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            raise

    def _get_default_metadata(self) -> SegmentMetadata:
        """Get default metadata when enrichment fails."""
        return SegmentMetadata(
            player_movements="unknown",
            motion_intensity=0.5,
            ball_zone="unknown",
            primary_zone="unknown",
            paint_occupied=False,
            ball_speed_estimate="medium",
            player_ball_proximity="medium",
            offensive_spacing="medium",
            screen_angle_est="none",
            # Basic events
            possible_screen=0.0,
            possible_drive=0.0,
            possible_shot=0.0,
            possible_pass=0.0,
            possible_rebound=0.0,
            # Advanced events
            possible_steal=0.0,
            possible_block=0.0,
            possible_cut=0.0,
            possible_post_up=0.0,
            possible_pick_and_roll=0.0,
            possible_fast_break=0.0,
            possible_defensive_rotation=0.0,
            possible_assist=0.0,
            possible_turnover=0.0,
            possible_dribble_move=0.0,
            rebound_type="none"
        )

    def get_api_usage(self) -> dict:
        """Get API usage statistics."""
        return {
            'total_calls': self.total_api_calls,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.total_tokens * 0.001 / 1000  # Rough estimate
        }
