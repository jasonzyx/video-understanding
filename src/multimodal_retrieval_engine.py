"""Multi-modal retrieval engine with score fusion."""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np
import faiss
import jsonlines

from src.data_models import Segment
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with multi-modal scores."""
    segment_id: str
    segment: Segment
    visual_score: float
    trajectory_score: float
    events_score: float
    metadata_score: float
    combined_score: float
    rank: int
    reranked: bool = False
    rerank_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'segment': self.segment.to_dict(),
            'visual_score': float(self.visual_score),
            'trajectory_score': float(self.trajectory_score),
            'events_score': float(self.events_score),
            'metadata_score': float(self.metadata_score),
            'combined_score': float(self.combined_score),
            'rank': self.rank,
            'reranked': self.reranked,
            'rerank_score': float(self.rerank_score) if self.rerank_score else None
        }


class MultiModalRetrievalEngine:
    """Hybrid retrieval with visual + trajectory + metadata + events."""

    def __init__(
        self,
        segments_path: Path,
        visual_index_path: Path,
        trajectory_index_path: Path,
        mapping_path: Path,
        weights: Optional[Dict[str, float]] = None,
        device: str = 'cpu',
        enable_reranking: bool = False,
        reranker_type: str = 'gemini',
        reranker_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        clip_dir: Optional[Path] = None
    ):
        """
        Initialize multi-modal retrieval engine.

        Args:
            segments_path: Path to segments JSONL file
            visual_index_path: Path to visual Faiss index
            trajectory_index_path: Path to trajectory Faiss index
            mapping_path: Path to index mapping JSON
            weights: Score fusion weights (default: visual=0.4, traj=0.3, events=0.2, meta=0.1)
            device: Device for CLIP model
            enable_reranking: If True, use reranking (Stage 2)
            reranker_type: Which reranker to use ('gemini' or 'gpt4v')
            reranker_api_key: OpenAI API key (for gpt4v) or Gemini key (auto-selected based on type)
            gemini_api_key: Alternative way to provide Gemini API key
            clip_dir: Directory containing extracted clips (required if enable_reranking=True)
        """
        logger.info("Loading multi-modal retrieval engine...")

        # Load Faiss indices
        self.visual_index = faiss.read_index(str(visual_index_path))
        self.trajectory_index = faiss.read_index(str(trajectory_index_path))
        logger.info(f"✓ Loaded Faiss indices: {self.visual_index.ntotal} vectors")

        # Load index mapping
        with open(mapping_path) as f:
            mapping = json.load(f)
        self.index_to_segment_id = {int(k): v for k, v in mapping['index_to_segment_id'].items()}
        self.segment_id_to_index = {v: int(k) for k, v in self.index_to_segment_id.items()}

        # Load segments
        self.segments = {}
        with jsonlines.open(segments_path) as reader:
            for seg_dict in reader:
                seg = Segment.from_dict(seg_dict)
                self.segments[seg.segment_id] = seg
        logger.info(f"✓ Loaded {len(self.segments)} segments")

        # Lazy-load CLIP encoder (loaded on first search to avoid startup crashes)
        self._embedding_generator = None
        self._device = device

        # Fusion weights (tunable)
        self.weights = weights or {
            'visual': 0.4,
            'trajectory': 0.3,
            'events': 0.2,
            'metadata': 0.1
        }

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"✓ Score fusion weights: {self.weights}")

        # Lazy-load reranker
        self._reranker = None
        self._enable_reranking = enable_reranking
        self._reranker_type = reranker_type
        self._reranker_api_key = reranker_api_key
        self._gemini_api_key = gemini_api_key
        self._clip_dir = clip_dir

        if enable_reranking:
            if reranker_type == 'gemini' and not (reranker_api_key or gemini_api_key):
                raise ValueError("gemini_api_key or reranker_api_key required for Gemini reranking")
            if reranker_type == 'gpt4v' and not reranker_api_key:
                raise ValueError("reranker_api_key (OpenAI) required for GPT-4V reranking")
            if not clip_dir or not clip_dir.exists():
                raise ValueError("clip_dir must exist when enable_reranking=True")

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Lazy-load embedding generator on first access."""
        if self._embedding_generator is None:
            logger.info("Initializing CLIP model for query encoding...")
            try:
                self._embedding_generator = EmbeddingGenerator(device=self._device)
                logger.info("✓ CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise RuntimeError(
                    f"Failed to load CLIP model. This may be due to NumPy/torch "
                    f"compatibility issues. Error: {e}"
                )
        return self._embedding_generator

    @property
    def reranker(self):
        """Lazy-load reranker on first access based on configured type."""
        if self._reranker is None and self._enable_reranking:
            if self._reranker_type == 'gemini':
                logger.info("Initializing Gemini 2.0 Flash reranker (native video)...")
                from src.gemini_flash_reranker import GeminiFlashReranker
                api_key = self._gemini_api_key or self._reranker_api_key
                self._reranker = GeminiFlashReranker(
                    api_key=api_key,
                    model='gemini-2.0-flash-exp'
                )
                logger.info("✓ Gemini reranker loaded")

            elif self._reranker_type == 'gpt4v':
                logger.info("Initializing GPT-4V reranker (frame-based)...")
                from src.gpt4v_reranker import GPT4VReranker
                self._reranker = GPT4VReranker(
                    api_key=self._reranker_api_key,
                    model='gpt-4o'
                )
                logger.info("✓ GPT-4V reranker loaded")

            else:
                raise ValueError(f"Unknown reranker type: {self._reranker_type}")

        return self._reranker

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
        relevance_threshold: float = 7.0
    ) -> List[SearchResult]:
        """
        Multi-modal search with score fusion.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional metadata filters
            weights: Optional custom weights for this query
            relevance_threshold: Minimum AI relevance score for reranked results (0-10)

        Returns:
            List of SearchResult objects ranked by combined score
        """
        # Use custom weights if provided
        search_weights = weights if weights else self.weights

        # Parse query into components
        query_components = self._parse_query(query)

        # Generate query embeddings
        visual_emb = self.embedding_generator.encode_text(query_components['visual_query'])
        trajectory_emb = self.embedding_generator.encode_text(query_components['motion_query'])

        # Ensure embeddings are normalized
        visual_emb = visual_emb / np.linalg.norm(visual_emb)
        trajectory_emb = trajectory_emb / np.linalg.norm(trajectory_emb)

        # Search each index (get more candidates than top_k for fusion)
        search_k = min(top_k * 5, self.visual_index.ntotal)

        visual_scores, visual_indices = self.visual_index.search(
            visual_emb.reshape(1, -1).astype('float32'), search_k
        )

        trajectory_scores, trajectory_indices = self.trajectory_index.search(
            trajectory_emb.reshape(1, -1).astype('float32'), search_k
        )

        # Collect all candidates
        candidates = {}
        for idx, score in zip(visual_indices[0], visual_scores[0]):
            if idx == -1:  # Faiss returns -1 for padding
                continue
            seg_id = self.index_to_segment_id.get(int(idx))
            if seg_id and seg_id in self.segments:
                if seg_id not in candidates:
                    candidates[seg_id] = {'visual': 0, 'trajectory': 0, 'events': 0, 'metadata': 0}
                candidates[seg_id]['visual'] = float(score)

        for idx, score in zip(trajectory_indices[0], trajectory_scores[0]):
            if idx == -1:
                continue
            seg_id = self.index_to_segment_id.get(int(idx))
            if seg_id and seg_id in self.segments:
                if seg_id not in candidates:
                    candidates[seg_id] = {'visual': 0, 'trajectory': 0, 'events': 0, 'metadata': 0}
                candidates[seg_id]['trajectory'] = float(score)

        # Score metadata/events for each candidate
        for seg_id, scores in candidates.items():
            segment = self.segments[seg_id]

            # Event-based scoring
            event_score = self._score_events(segment, query_components['event_hints'])
            scores['events'] = event_score

            # Metadata-based scoring
            metadata_score = self._score_metadata(segment, query_components['metadata_filters'])
            scores['metadata'] = metadata_score

        # Fuse scores
        results = []
        for seg_id, scores in candidates.items():
            combined_score = (
                search_weights['visual'] * scores['visual'] +
                search_weights['trajectory'] * scores['trajectory'] +
                search_weights['events'] * scores['events'] +
                search_weights['metadata'] * scores['metadata']
            )

            results.append(SearchResult(
                segment_id=seg_id,
                segment=self.segments[seg_id],
                visual_score=scores['visual'],
                trajectory_score=scores['trajectory'],
                events_score=scores['events'],
                metadata_score=scores['metadata'],
                combined_score=combined_score,
                rank=0  # Will update after sorting
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        results = results[:top_k]

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        # Apply reranking if enabled (Stage 2)
        if self._enable_reranking and self.reranker:
            logger.info(f"Reranking top {len(results)} candidates with {self._reranker_type}...")

            # Get segments for reranking
            segments_to_rerank = [r.segment for r in results]

            # Rerank
            rerank_results = self.reranker.rerank(
                query=query,
                segments=segments_to_rerank,
                clip_dir=self._clip_dir,
                top_k=top_k
            )

            # Create mapping from segment_id to rerank result
            rerank_map = {rr.segment_id: rr for rr in rerank_results}

            # Update SearchResult objects
            reranked_results = []
            for result in results:
                rerank_result = rerank_map.get(result.segment_id)
                if rerank_result:
                    # Update fields
                    result.reranked = True
                    result.rerank_score = rerank_result.relevance_score

                    # Store explanation in segment metadata
                    if result.segment.metadata is None:
                        result.segment.metadata = {}
                    result.segment.metadata['rerank_explanation'] = rerank_result.explanation

                    # Only include if score meets threshold
                    if rerank_result.relevance_score >= relevance_threshold:
                        reranked_results.append(result)

            # Re-sort by rerank_score
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1

            logger.info(f"Reranking complete: {len(reranked_results)}/{len(results)} results passed threshold (>= {relevance_threshold:.1f})")

            results = reranked_results

        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)

        return results

    def _parse_query(self, query: str) -> Dict[str, any]:
        """
        Parse query into components using simple keyword matching.

        Args:
            query: Natural language query

        Returns:
            Dictionary with query components
        """
        query_lower = query.lower()

        # Visual query (use full query for visual similarity)
        visual_query = query

        # Motion query (extract motion-related terms)
        motion_query = self._extract_motion_terms(query_lower)

        # Event hints (detect action keywords)
        event_hints = self._extract_event_terms(query_lower)

        # Metadata filters (detect zone/spatial keywords)
        metadata_filters = self._extract_metadata(query_lower)

        return {
            'visual_query': visual_query,
            'motion_query': motion_query,
            'event_hints': event_hints,
            'metadata_filters': metadata_filters
        }

    def _extract_motion_terms(self, query: str) -> str:
        """Extract motion-related terms from query."""
        # Map query terms to motion descriptions
        motion_mappings = {
            'screen': 'screen set, roll to basket',
            'pick': 'screen set, roll to basket',
            'roll': 'roll to basket after screen',
            'drive': 'player drives to basket',
            'cut': 'player cuts to basket',
            'fast break': 'fast movement, transition',
            'transition': 'fast movement, transition',
            'iso': 'isolation, one on one',
            'post up': 'post position, back to basket',
        }

        motion_terms = []
        for keyword, description in motion_mappings.items():
            if keyword in query:
                motion_terms.append(description)

        return ', '.join(motion_terms) if motion_terms else query

    def _extract_event_terms(self, query: str) -> List[str]:
        """Extract event-related terms from query."""
        event_keywords = {
            # Basic events
            'screen': ['screen', 'pick'],
            'drive': ['drive', 'penetrate', 'attack'],
            'shot': ['shot', 'shoot', 'three', 'dunk', 'layup'],
            'pass': ['pass'],
            'rebound': ['rebound', 'board'],
            # Advanced events
            'steal': ['steal', 'turnover', 'interception'],
            'block': ['block', 'rejection', 'denied'],
            'cut': ['cut', 'backdoor', 'baseline cut'],
            'post_up': ['post', 'post-up', 'back down'],
            'pick_and_roll': ['pick and roll', 'pick-and-roll', 'pnr'],
            'fast_break': ['fast break', 'transition', 'fastbreak'],
            'defensive_rotation': ['rotation', 'help defense', 'switch'],
            'assist': ['assist'],
            'dribble_move': ['crossover', 'dribble move', 'behind the back', 'through the legs']
        }

        hints = []
        for event, keywords in event_keywords.items():
            if any(kw in query for kw in keywords):
                hints.append(event)

        return hints

    def _extract_metadata(self, query: str) -> Dict[str, any]:
        """Extract metadata filters from query."""
        filters = {}

        # Zone detection
        zone_keywords = {
            'corner': 'corner',
            'wing': 'wing',
            'key': 'top_of_key',
            'paint': 'paint',
            'perimeter': 'mid_range',
            'top of the key': 'top_of_key',
            'elbow': 'mid_range'
        }

        for keyword, zone in zone_keywords.items():
            if keyword in query:
                filters['zone'] = zone
                break

        # Speed detection
        if 'fast' in query or 'quick' in query:
            filters['speed'] = 'fast'
        elif 'slow' in query:
            filters['speed'] = 'slow'

        return filters

    def _score_events(self, segment: Segment, event_hints: List[str]) -> float:
        """Score segment based on weak event detection."""
        if not event_hints or not segment.metadata:
            return 0.0

        events = segment.metadata.get('weak_events', {})
        if not events:
            return 0.0

        # Match event hints to weak event probabilities
        scores = []
        for hint in event_hints:
            # Basic events
            if hint == 'screen':
                scores.append(events.get('possible_screen', 0))
            elif hint == 'drive':
                scores.append(events.get('possible_drive', 0))
            elif hint == 'shot':
                scores.append(events.get('possible_shot', 0))
            elif hint == 'pass':
                scores.append(events.get('possible_pass', 0))
            elif hint == 'rebound':
                scores.append(events.get('possible_rebound', 0))
            # Advanced events
            elif hint == 'steal':
                scores.append(events.get('possible_steal', 0))
            elif hint == 'block':
                scores.append(events.get('possible_block', 0))
            elif hint == 'cut':
                scores.append(events.get('possible_cut', 0))
            elif hint == 'post_up':
                scores.append(events.get('possible_post_up', 0))
            elif hint == 'pick_and_roll':
                scores.append(events.get('possible_pick_and_roll', 0))
                # Also boost screen score for pick-and-roll
                scores.append(events.get('possible_screen', 0))
            elif hint == 'fast_break':
                scores.append(events.get('possible_fast_break', 0))
            elif hint == 'defensive_rotation':
                scores.append(events.get('possible_defensive_rotation', 0))
            elif hint == 'assist':
                scores.append(events.get('possible_assist', 0))
                # Assist implies pass
                scores.append(events.get('possible_pass', 0))
            elif hint == 'dribble_move':
                scores.append(events.get('possible_dribble_move', 0))

        # Return max probability (most confident event match)
        return max(scores) if scores else 0.0

    def _score_metadata(self, segment: Segment, metadata_filters: Dict[str, any]) -> float:
        """Score segment based on metadata match."""
        if not metadata_filters or not segment.metadata:
            return 0.0

        semantics = segment.metadata.get('court_semantics', {})
        derived = segment.metadata.get('derived', {})

        score = 0.0

        # Zone matching (binary: match or no match)
        if 'zone' in metadata_filters:
            target_zone = metadata_filters['zone']
            ball_zone = semantics.get('ball_zone', '')
            primary_zone = semantics.get('primary_zone', '')

            # Exact match on ball zone
            if ball_zone == target_zone:
                score += 1.0
            # Partial match on primary zone
            elif primary_zone == target_zone:
                score += 0.7
            # Partial match on zone name substring
            elif target_zone in ball_zone or target_zone in primary_zone:
                score += 0.5

        # Speed matching
        if 'speed' in metadata_filters:
            if derived.get('ball_speed_estimate') == metadata_filters['speed']:
                score += 0.5

        return min(score, 1.0)

    def _apply_filters(self, results: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Apply hard filters to results."""
        filtered = []

        for result in results:
            segment = result.segment
            if not segment.metadata:
                continue

            # Apply duration filter
            if 'min_duration' in filters:
                if segment.duration < filters['min_duration']:
                    continue

            if 'max_duration' in filters:
                if segment.duration > filters['max_duration']:
                    continue

            # Apply event threshold filter
            if 'min_event_score' in filters:
                if result.events_score < filters['min_event_score']:
                    continue

            filtered.append(result)

        return filtered

    def get_segment_by_id(self, segment_id: str) -> Optional[Segment]:
        """Get segment by ID."""
        return self.segments.get(segment_id)

    def get_stats(self) -> Dict[str, any]:
        """Get retrieval engine statistics."""
        return {
            'num_segments': len(self.segments),
            'num_indexed': self.visual_index.ntotal,
            'embedding_dim': self.visual_index.d,
            'weights': self.weights
        }
