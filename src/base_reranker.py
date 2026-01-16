"""Base class for video segment rerankers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from dataclasses import dataclass

from src.data_models import Segment


@dataclass
class RerankResult:
    """Result of reranking a single segment."""
    segment_id: str
    relevance_score: float  # 0-10
    explanation: str
    is_relevant: bool  # True if score >= 7


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        segments: List[Segment],
        clip_dir: Path,
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank segments using model-specific verification.

        Args:
            query: Original search query
            segments: Candidate segments from Stage 1
            clip_dir: Directory containing extracted clips
            top_k: Number of results to return after reranking

        Returns:
            List of RerankResult, sorted by relevance_score descending
        """
        pass

    @abstractmethod
    def get_api_usage(self) -> dict:
        """Get API usage statistics."""
        pass
