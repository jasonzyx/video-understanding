"""Data models for temporal substrate."""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Possession:
    """Represents a single possession in a game."""
    game_id: str
    possession_id: int
    start: float  # seconds
    end: float  # seconds
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class Segment:
    """Represents a candidate temporal segment."""
    game_id: str
    possession_id: int
    segment_id: str
    start: float  # seconds
    end: float  # seconds
    duration: float  # seconds

    # Placeholders for future features
    video_embedding: Optional[Any] = None
    trajectory_embedding: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        data = asdict(self)
        # Remove None embeddings for cleaner JSON
        if data['video_embedding'] is None:
            del data['video_embedding']
        if data['trajectory_embedding'] is None:
            del data['trajectory_embedding']
        return data

    @classmethod
    def from_dict(cls, data: dict):
        # Handle optional fields that might not be in saved data
        return cls(
            game_id=data['game_id'],
            possession_id=data['possession_id'],
            segment_id=data['segment_id'],
            start=data['start'],
            end=data['end'],
            duration=data['duration'],
            video_embedding=data.get('video_embedding'),
            trajectory_embedding=data.get('trajectory_embedding'),
            metadata=data.get('metadata')
        )
