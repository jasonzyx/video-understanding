"""GPT-4V based reranking for search results."""
import logging
from pathlib import Path
from typing import List
import base64
from openai import OpenAI

from src.data_models import Segment
from src.base_reranker import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class GPT4VReranker(BaseReranker):
    """Rerank search results using GPT-4V to verify relevance."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o recommended)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Track API usage
        self.total_calls = 0
        self.total_tokens = 0

    def rerank(
        self,
        query: str,
        segments: List[Segment],
        clip_dir: Path,
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank segments using GPT-4V verification.

        Args:
            query: Original search query
            segments: Candidate segments from Stage 1
            clip_dir: Directory containing extracted clips
            top_k: Number of results to return after reranking

        Returns:
            List of RerankResult, sorted by relevance_score descending
        """
        results = []

        for segment in segments:
            clip_path = clip_dir / f"{segment.segment_id}.mp4"

            if not clip_path.exists():
                logger.warning(f"Clip not found: {clip_path}")
                # Give low score if clip missing
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
                # Give neutral score on error
                results.append(RerankResult(
                    segment_id=segment.segment_id,
                    relevance_score=5.0,
                    explanation=f"Error during reranking: {str(e)}",
                    is_relevant=False
                ))

        # Sort by relevance score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top_k
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
        # Extract 3 frames from clip (beginning, middle, end)
        frames = self._extract_frames(clip_path, num_frames=3)

        if not frames:
            return RerankResult(
                segment_id=segment.segment_id,
                relevance_score=0.0,
                explanation="Could not extract frames",
                is_relevant=False
            )

        # Build prompt
        prompt = self._build_rerank_prompt(query, segment)

        # Query GPT-4V
        response = self._query_gpt4v(prompt, frames)

        # Parse response
        relevance_score, explanation = self._parse_response(response)

        return RerankResult(
            segment_id=segment.segment_id,
            relevance_score=relevance_score,
            explanation=explanation,
            is_relevant=relevance_score >= 7.0
        )

    def _extract_frames(self, clip_path: Path, num_frames: int = 3) -> List:
        """Extract evenly-spaced frames from clip."""
        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices (evenly spaced)
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]

        frames = []
        current_idx = 0

        while True:
            ret, frame_img = cap.read()
            if not ret:
                break

            if current_idx in frame_indices:
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame_img)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)

            current_idx += 1

        cap.release()
        return frames

    def _build_rerank_prompt(self, query: str, segment: Segment) -> str:
        """Build prompt for GPT-4V relevance scoring."""
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

I will show you 3 frames from this segment. Your task:
1. Watch the frames carefully
2. Determine if the segment is RELEVANT to the query
3. Rate relevance on a scale of 0-10:
   - 10: Perfect match (exactly what the query describes)
   - 7-9: Good match (clearly relevant)
   - 4-6: Partial match (somewhat related)
   - 1-3: Poor match (barely relevant)
   - 0: Not relevant at all

**Important guidelines:**
- For queries like "corner three pointer", the segment MUST show a shot attempt from the corner
- For queries like "drive to basket", the segment MUST show a player actively driving
- Don't give high scores just because the court location matches - the ACTION must match
- Be strict: only score 7+ if you clearly see the described action

Respond in this JSON format:
{{
  "relevance_score": <0-10>,
  "explanation": "<1-2 sentences explaining why this score>"
}}

Only output JSON."""

        return prompt

    def _query_gpt4v(self, prompt: str, frames: List[str]) -> str:
        """Query GPT-4V with prompt and frames."""
        # Build message content
        content = [{"type": "text", "text": prompt}]

        # Add frames
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low"  # Cost optimization
                }
            })

        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
            temperature=0.0
        )

        self.total_calls += 1
        self.total_tokens += response.usage.total_tokens

        return response.choices[0].message.content

    def _parse_response(self, response_text: str) -> tuple[float, str]:
        """Parse GPT-4V response to extract score and explanation."""
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
            'estimated_cost': self.total_tokens * 0.001 / 1000  # Rough estimate
        }
