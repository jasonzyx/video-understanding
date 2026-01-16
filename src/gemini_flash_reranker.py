"""Gemini 2.0 Flash based reranking for search results (native video)."""
import logging
from pathlib import Path
from typing import List
import google.generativeai as genai

from src.data_models import Segment
from src.base_reranker import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class GeminiFlashReranker(BaseReranker):
    """Rerank search results using Gemini 2.0 Flash with native video understanding."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """
        Args:
            api_key: Google AI API key
            model: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

        # Track API usage
        self.total_calls = 0
        self.total_tokens = 0

        logger.info(f"Initialized GeminiFlashReranker with model: {model}")

    def rerank(
        self,
        query: str,
        segments: List[Segment],
        clip_dir: Path,
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank segments using Gemini video understanding.

        Args:
            query: Original search query
            segments: Candidate segments from Stage 1
            clip_dir: Directory containing extracted clips
            top_k: Number of results to return after reranking

        Returns:
            List of RerankResult, sorted by relevance_score descending
        """
        logger.info(f"Reranking {len(segments)} segments for query: '{query}' using {self.model_name}")
        results = []

        for segment in segments:
            clip_path = clip_dir / f"{segment.segment_id}.mp4"

            if not clip_path.exists():
                logger.warning(f"Clip not found: {clip_path}")
                results.append(RerankResult(
                    segment_id=segment.segment_id,
                    relevance_score=0.0,
                    explanation="Clip file not found",
                    is_relevant=False
                ))
                continue

            try:
                rerank_result = self._score_relevance(query, clip_path, segment)
                results.append(rerank_result)
            except Exception as e:
                logger.error(f"Error reranking {segment.segment_id}: {e}")
                results.append(RerankResult(
                    segment_id=segment.segment_id,
                    relevance_score=5.0,
                    explanation=f"Error during reranking: {str(e)}",
                    is_relevant=False
                ))

        # Sort by relevance score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Log results summary
        relevant_count = sum(1 for r in results if r.is_relevant)
        logger.info(f"Reranking complete: {relevant_count}/{len(results)} segments are relevant (score >= 7)")
        for i, r in enumerate(results[:3], 1):
            logger.info(f"  Top {i}: {r.segment_id} - Score: {r.relevance_score:.1f}/10 - {r.explanation[:60]}...")

        return results[:top_k]

    def _score_relevance(
        self,
        query: str,
        clip_path: Path,
        segment: Segment
    ) -> RerankResult:
        """
        Score relevance of a single segment to the query.

        Args:
            query: Search query
            clip_path: Path to video clip
            segment: Segment metadata

        Returns:
            RerankResult with relevance score
        """
        import time

        # Build prompt
        prompt = self._build_rerank_prompt(query, segment)

        # Upload video file
        video_file = genai.upload_file(path=str(clip_path))

        # Wait for file to be processed (Gemini requirement)
        max_wait_time = 30  # seconds
        wait_start = time.time()
        while video_file.state.name == "PROCESSING":
            if time.time() - wait_start > max_wait_time:
                raise TimeoutError(f"Video processing timeout after {max_wait_time}s")
            time.sleep(0.5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            raise ValueError(f"Video file failed to process: {video_file.state.name}")

        # Query Gemini with native video
        response = self.model.generate_content(
            [prompt, video_file],
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=200
            )
        )

        self.total_calls += 1
        # Note: Gemini usage tracking may differ, estimate tokens
        self.total_tokens += len(prompt) + 200

        # Parse response
        relevance_score, explanation = self._parse_response(response.text)

        return RerankResult(
            segment_id=segment.segment_id,
            relevance_score=relevance_score,
            explanation=explanation,
            is_relevant=relevance_score >= 7.0
        )

    def _build_rerank_prompt(self, query: str, segment: Segment) -> str:
        """Build prompt for Gemini relevance scoring."""
        # Include segment metadata if available
        metadata_str = ""
        if segment.metadata:
            events = segment.metadata.get('weak_events', {})
            semantics = segment.metadata.get('court_semantics', {})

            if events:
                high_conf = [(k.replace('possible_', ''), v)
                            for k, v in events.items()
                            if isinstance(v, (int, float)) and v > 0.5]
                if high_conf:
                    metadata_str = f"\n\n**Detected events:** {', '.join([f'{e}({s:.1f})' for e, s in high_conf])}"

            if semantics:
                zone = semantics.get('ball_zone', 'unknown')
                if zone != 'unknown':
                    metadata_str += f"\n**Ball zone:** {zone}"

        prompt = f"""You are evaluating if a basketball video segment matches a search query.

**Query:** "{query}"

**Segment duration:** {segment.duration:.1f}s{metadata_str}

I will show you a video clip. Your task:
1. Watch the ENTIRE video carefully, paying attention to the motion and actions
2. Determine if the segment is RELEVANT to the query
3. Rate relevance on a scale of 0-10:
   - 10: Perfect match (exactly what the query describes)
   - 7-9: Good match (clearly relevant)
   - 4-6: Partial match (somewhat related)
   - 1-3: Poor match (barely relevant)
   - 0: Not relevant at all

**Important guidelines:**
- For queries like "corner three pointer", the video MUST show a shot attempt from the corner
- For queries like "drive to basket", the video MUST show a player actively driving
- Use the FULL VIDEO CONTEXT - actions happen over time, not just in single frames
- Don't give high scores just because the court location matches - the ACTION must match
- Be strict: only score 7+ if you clearly see the described action

Respond in this JSON format:
{{
  "relevance_score": <0-10>,
  "explanation": "<1-2 sentences explaining why this score, mentioning specific actions you saw>"
}}

Only output JSON."""

        return prompt

    def _parse_response(self, response_text: str) -> tuple[float, str]:
        """Parse Gemini response to extract score and explanation."""
        import json

        try:
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            data = json.loads(response_text)

            relevance_score = float(data.get('relevance_score', 5.0))
            explanation = data.get('explanation', 'No explanation provided')

            # Clamp score to 0-10
            relevance_score = max(0.0, min(10.0, relevance_score))

            return relevance_score, explanation

        except Exception as e:
            logger.error(f"Failed to parse rerank response: {e}")
            logger.error(f"Response: {response_text}")
            return 5.0, f"Parse error: {str(e)}"

    def get_api_usage(self) -> dict:
        """Get API usage statistics."""
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.total_calls * 0.01  # Rough estimate for video
        }
