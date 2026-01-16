"""Streamlit viewer for exploring temporal segments."""
import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import jsonlines
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Video Moment Viewer", layout="wide")


def load_index_data(game_id: str, output_dir: Path):
    """Load possessions, segments, and metadata for a game."""
    possessions_file = output_dir / f"{game_id}_possessions.jsonl"
    segments_file = output_dir / f"{game_id}_segments.jsonl"
    metadata_file = output_dir / f"{game_id}_index_metadata.json"

    if not all([f.exists() for f in [possessions_file, segments_file, metadata_file]]):
        return None, None, None

    with jsonlines.open(possessions_file) as reader:
        possessions = list(reader)

    with jsonlines.open(segments_file) as reader:
        segments = list(reader)

    with open(metadata_file) as f:
        metadata = json.load(f)

    return possessions, segments, metadata


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ms"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_clip_path(segment_id: str, clips_dir: Path) -> Path:
    """Get path to extracted clip for a segment."""
    return clips_dir / f"{segment_id}.mp4"


def show_video_player(video_path: str, start: float, end: float, player_id: str, segment_id: str = None, clips_dir: Path = None):
    """Show video player - either extracted clip or full video with time controls.

    Args:
        video_path: Path to original video file
        start: Start time in seconds
        end: End time in seconds
        player_id: Unique ID for this player instance
        segment_id: Segment ID (for finding extracted clip)
        clips_dir: Directory containing extracted clips
    """
    # Check if extracted clip exists
    if segment_id and clips_dir:
        clip_path = get_clip_path(segment_id, clips_dir)
        if clip_path.exists():
            # Use extracted clip - much simpler and more reliable!
            st.video(str(clip_path))
            duration = end - start
            st.caption(f"**Clip:** {format_time(start)} → {format_time(end)} ({duration:.1f}s)")
            return

    # Fallback: show full video with start time
    # Note: This doesn't provide precise end control, but clips should be extracted
    if Path(video_path).exists():
        st.video(video_path, start_time=int(start))
        st.warning(f"⚠️ Showing full video from {format_time(start)}. Run build_index.py with --extract-clips for better experience.")
    else:
        st.error(f"Video not found: {video_path}")


def main():
    st.title("🏀 Sports Video Moment Viewer (M1)")

    st.sidebar.header("Configuration")

    # Select game
    output_dir = Path("outputs")
    if not output_dir.exists():
        st.error("No outputs directory found. Run build_index.py first.")
        return

    # Find available games
    metadata_files = list(output_dir.glob("*_index_metadata.json"))
    if not metadata_files:
        st.error("No indexed games found. Run build_index.py first.")
        return

    game_ids = [f.stem.replace("_index_metadata", "") for f in metadata_files]
    selected_game = st.sidebar.selectbox("Select Game", game_ids)

    # Load data
    possessions, segments, metadata = load_index_data(selected_game, output_dir)

    if possessions is None:
        st.error(f"Could not load data for game: {selected_game}")
        return

    # Display metadata
    st.sidebar.subheader("Game Info")
    st.sidebar.text(f"Duration: {metadata['video_metadata']['duration']:.1f}s")
    st.sidebar.text(f"Possessions: {metadata['num_possessions']}")
    st.sidebar.text(f"Segments: {metadata['num_segments']}")
    st.sidebar.text(f"Window sizes: {metadata['window_sizes']}")

    if metadata.get('auto_detected', False):
        st.sidebar.success("✓ Auto-detected possessions")
    else:
        st.sidebar.info("Manual possession config")

    # Check if clips are available
    clips_extracted = metadata.get('clips_extracted', False)
    clips_dir = None
    if clips_extracted:
        clips_dir = Path(metadata['clips_dir'])
        st.sidebar.success(f"✓ Extracted clips available")
    else:
        st.sidebar.info("💡 Run with --extract-clips for better playback")

    # Main view
    tab1, tab2, tab3 = st.tabs(["Possessions", "Segments Browser", "Search Segments"])

    with tab1:
        st.header("Possessions")
        st.write(f"Total: {len(possessions)}")

        poss_df = pd.DataFrame(possessions)
        st.dataframe(poss_df, use_container_width=True)

        # Possession selector
        selected_poss_id = st.selectbox(
            "Select Possession to View",
            range(len(possessions)),
            format_func=lambda i: f"Possession {i} ({format_time(possessions[i]['start'])} - {format_time(possessions[i]['end'])})"
        )

        poss = possessions[selected_poss_id]
        st.subheader(f"Possession {selected_poss_id}")
        st.write(f"**Time:** {format_time(poss['start'])} - {format_time(poss['end'])}")
        st.write(f"**Duration:** {poss['end'] - poss['start']:.1f}s")

        # Show video for possession
        video_path = metadata['video_path']
        # For possessions, we don't have extracted clips (only segments have them)
        # So just show with start time
        if Path(video_path).exists():
            st.video(video_path, start_time=int(poss['start']))
        else:
            st.warning(f"Video not found: {video_path}")

    with tab2:
        st.header("Segments Browser")

        # Filter by possession
        filter_poss = st.selectbox(
            "Filter by Possession",
            ["All"] + [f"Possession {i}" for i in range(len(possessions))],
            key="seg_filter"
        )

        # Filter segments
        if filter_poss == "All":
            filtered_segments = segments
        else:
            poss_id = int(filter_poss.split()[1])
            filtered_segments = [s for s in segments if s['possession_id'] == poss_id]

        st.write(f"Showing {len(filtered_segments)} segments")

        # Pagination
        page_size = 20
        page = st.number_input("Page", min_value=1, max_value=(len(filtered_segments) // page_size) + 1, value=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Display segments
        for seg in filtered_segments[start_idx:end_idx]:
            with st.expander(
                f"🎬 {seg['segment_id']} | {format_time(seg['start'])} - {format_time(seg['end'])} ({seg['duration']:.1f}s)"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write(f"**Possession:** {seg['possession_id']}")
                    st.write(f"**Window size:** {seg['metadata']['window_size']}s")
                    st.write(f"**Start:** {format_time(seg['start'])}")
                    st.write(f"**End:** {format_time(seg['end'])}")

                with col2:
                    show_video_player(
                        video_path=video_path,
                        start=seg['start'],
                        end=seg['end'],
                        player_id=f"seg_player_{seg['segment_id']}",
                        segment_id=seg['segment_id'],
                        clips_dir=clips_dir
                    )

    with tab3:
        st.header("🔍 Multi-Modal Search")

        # Check if indices exist
        visual_index_path = output_dir / f"{selected_game}_visual_index.bin"
        trajectory_index_path = output_dir / f"{selected_game}_trajectory_index.bin"
        mapping_path = output_dir / f"{selected_game}_index_mapping.json"

        indices_exist = all([
            visual_index_path.exists(),
            trajectory_index_path.exists(),
            mapping_path.exists()
        ])

        if not indices_exist:
            st.warning("⚠️ Search indices not found. Run enrichment first:")
            st.code("""python build_index.py \\
  --video data/nba_final.mp4 \\
  --use-mllm \\
  --extract-clips \\
  --enrich-segments \\
  --generate-embeddings \\
  --stride 2.0""")
        else:
            # Reranking options (show before loading engine to get API keys)
            st.subheader("Search Options")
            enable_reranking = st.checkbox(
                "Enable AI Reranking",
                value=True,
                help="Use AI to verify search results. Slower but more accurate."
            )

            # Reranker type selection (only shown if reranking enabled)
            reranker_type = 'gemini'  # default
            if enable_reranking:
                reranker_type = st.selectbox(
                    "Reranker Model",
                    options=['gemini', 'gpt4v'],
                    format_func=lambda x: {
                        'gemini': 'Gemini 2.0 Flash (video-native) ⭐ Recommended',
                        'gpt4v': 'GPT-4V (frame-based)'
                    }[x],
                    help="Gemini uses native video understanding. GPT-4V analyzes individual frames."
                )

                # Show info based on selected reranker
                if reranker_type == 'gemini':
                    st.info("🎥 Gemini: Sees full video motion. ~$0.01/query, ~3s latency")
                else:
                    st.info("🖼️ GPT-4V: Analyzes 3 frames. ~$0.02/query, ~2-4s latency")

                # Relevance threshold slider
                st.write("**Relevance Threshold**")
                relevance_threshold = st.slider(
                    "Minimum AI Score to show results",
                    min_value=0.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.5,
                    help="Only show segments with AI relevance score >= this value. Higher = stricter filtering."
                )
                st.caption(f"Currently filtering to show only segments scoring ≥ {relevance_threshold:.1f}/10")
            else:
                relevance_threshold = 7.0  # Default when reranking disabled

            st.markdown("---")

            # Create cache key based on reranking settings
            cache_key = f"engine_{enable_reranking}_{reranker_type if enable_reranking else 'none'}"

            # Load retrieval engine (recreate if settings changed)
            if 'retrieval_engine_key' not in st.session_state or st.session_state.retrieval_engine_key != cache_key:
                with st.spinner("Loading retrieval engine..."):
                    from src.multimodal_retrieval_engine import MultiModalRetrievalEngine
                    import os

                    # Check for enriched segments first, then demo, then original
                    enriched_segments_path = output_dir / f"{selected_game}_segments_enriched.jsonl"
                    demo_segments_path = output_dir / f"{selected_game}_segments_demo.jsonl"
                    original_segments_path = output_dir / f"{selected_game}_segments.jsonl"

                    if enriched_segments_path.exists():
                        segments_path = enriched_segments_path
                    elif demo_segments_path.exists():
                        segments_path = demo_segments_path
                    else:
                        segments_path = original_segments_path

                    # Get API key if reranking enabled
                    api_key = None
                    if enable_reranking:
                        if reranker_type == 'gemini':
                            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
                            if not api_key:
                                st.error("GOOGLE_API_KEY or GEMINI_API_KEY not set. Please set it in .env file.")
                                st.stop()
                        else:  # gpt4v
                            api_key = os.getenv('OPENAI_API_KEY')
                            if not api_key:
                                st.error("OPENAI_API_KEY not set. Please set it in .env file.")
                                st.stop()

                    st.session_state.retrieval_engine = MultiModalRetrievalEngine(
                        segments_path=segments_path,
                        visual_index_path=visual_index_path,
                        trajectory_index_path=trajectory_index_path,
                        mapping_path=mapping_path,
                        device='cpu',
                        enable_reranking=enable_reranking,
                        reranker_type=reranker_type,
                        reranker_api_key=api_key,
                        clip_dir=clips_dir if enable_reranking else None
                    )
                    st.session_state.retrieval_engine_key = cache_key

                    if enable_reranking:
                        st.success(f"✓ Reranking enabled with {reranker_type.upper()}")

            engine = st.session_state.retrieval_engine

            # Search interface
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "Search query",
                    placeholder="e.g., 'pick and roll', 'drive to basket', 'corner three pointer'",
                    key="search_query"
                )
            with col2:
                top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)

            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                st.write("**Fusion Weights** (must sum to 1.0)")
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                with col_w1:
                    visual_weight = st.slider("Visual", 0.0, 1.0, 0.4, 0.05)
                with col_w2:
                    trajectory_weight = st.slider("Motion", 0.0, 1.0, 0.3, 0.05)
                with col_w3:
                    events_weight = st.slider("Events", 0.0, 1.0, 0.2, 0.05)
                with col_w4:
                    metadata_weight = st.slider("Metadata", 0.0, 1.0, 0.1, 0.05)

                # Filters
                st.write("**Filters**")
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    min_duration = st.number_input("Min duration (s)", min_value=0.0, value=0.0, step=0.5)
                with col_f2:
                    max_duration = st.number_input("Max duration (s)", min_value=0.0, value=10.0, step=0.5)
                with col_f3:
                    min_event_score = st.number_input("Min event score", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

            # Execute search
            if query:
                try:
                    # Build custom weights
                    weights = {
                        'visual': visual_weight,
                        'trajectory': trajectory_weight,
                        'events': events_weight,
                        'metadata': metadata_weight
                    }

                    # Build filters
                    filters = {}
                    if min_duration > 0:
                        filters['min_duration'] = min_duration
                    if max_duration > 0:
                        filters['max_duration'] = max_duration
                    if min_event_score > 0:
                        filters['min_event_score'] = min_event_score

                    # Search
                    with st.spinner("Searching..."):
                        results = engine.search(
                            query=query,
                            top_k=top_k,
                            filters=filters if filters else None,
                            weights=weights,
                            relevance_threshold=relevance_threshold if enable_reranking else 7.0
                        )

                    if not results:
                        if enable_reranking:
                            st.warning(f"No results found with AI score ≥ {relevance_threshold:.1f}/10. Try lowering the threshold slider.")
                        else:
                            st.warning("No results found")
                    else:
                        if enable_reranking:
                            st.success(f"Found {len(results)} results with AI score ≥ {relevance_threshold:.1f}/10")
                        else:
                            st.success(f"Found {len(results)} results")

                        # Display results
                        for result in results:
                            # Create header with prominent rerank info
                            if result.reranked:
                                header = f"#{result.rank} ⭐ RERANKED | {result.segment_id} | **AI Score: {result.rerank_score:.1f}/10** | Time: {format_time(result.segment.start)}"
                            else:
                                header = f"#{result.rank} {result.segment_id} | Score: {result.combined_score:.3f} | Time: {format_time(result.segment.start)}"

                            with st.expander(header):
                                col1, col2 = st.columns([1, 2])

                                with col1:
                                    # Show rerank info prominently if available
                                    if result.reranked:
                                        st.markdown("### 🎯 AI Reranking")
                                        st.metric("Relevance Score", f"{result.rerank_score:.1f}/10",
                                                 delta="Verified by Gemini" if result.rerank_score >= 7 else "Low confidence")
                                        if result.segment.metadata and result.segment.metadata.get('rerank_explanation'):
                                            st.success(f"💡 **AI Explanation:**\n\n{result.segment.metadata['rerank_explanation']}")
                                        st.markdown("---")

                                    # Basic info
                                    st.write(f"**Duration:** {result.segment.duration:.1f}s")
                                    st.write(f"**Start:** {format_time(result.segment.start)}")
                                    st.write(f"**End:** {format_time(result.segment.end)}")

                                    # Score breakdown
                                    st.write("**Stage 1 Scores:**" if result.reranked else "**Score Breakdown:**")
                                    score_data = {
                                        "Component": ["Combined", "Visual", "Motion", "Events", "Metadata"],
                                        "Score": [
                                            f"{result.combined_score:.3f}",
                                            f"{result.visual_score:.3f}",
                                            f"{result.trajectory_score:.3f}",
                                            f"{result.events_score:.3f}",
                                            f"{result.metadata_score:.3f}"
                                        ]
                                    }
                                    st.dataframe(pd.DataFrame(score_data), hide_index=True, use_container_width=True)

                                    # Enriched metadata
                                    if result.segment.metadata:
                                        with st.expander("📊 Metadata"):
                                            metadata = result.segment.metadata

                                            # Court semantics
                                            semantics = metadata.get('court_semantics', {})
                                            if semantics:
                                                st.write("**Court Semantics:**")
                                                st.write(f"- Ball: {semantics.get('ball_zone', 'N/A')}")
                                                st.write(f"- Action: {semantics.get('primary_zone', 'N/A')}")
                                                st.write(f"- Paint: {'Yes' if semantics.get('paint_occupied') else 'No'}")

                                            # Derived metrics
                                            derived = metadata.get('derived', {})
                                            if derived:
                                                st.write("**Metrics:**")
                                                st.write(f"- Ball speed: {derived.get('ball_speed_estimate', 'N/A')}")
                                                st.write(f"- Spacing: {derived.get('offensive_spacing', 'N/A')}")
                                                st.write(f"- Motion: {derived.get('motion_intensity', 'N/A'):.2f}")
                                                if derived.get('screen_angle_est', 'none') != 'none':
                                                    st.write(f"- Screen: {derived.get('screen_angle_est')}")

                                            # Weak events - organized by category
                                            events = metadata.get('weak_events', {})
                                            if events:
                                                # Group events by category
                                                basic_events = {
                                                    'Screen': events.get('possible_screen', 0),
                                                    'Drive': events.get('possible_drive', 0),
                                                    'Shot': events.get('possible_shot', 0),
                                                    'Pass': events.get('possible_pass', 0),
                                                    'Rebound': events.get('possible_rebound', 0)
                                                }

                                                advanced_events = {
                                                    'Pick & Roll': events.get('possible_pick_and_roll', 0),
                                                    'Fast Break': events.get('possible_fast_break', 0),
                                                    'Steal': events.get('possible_steal', 0),
                                                    'Block': events.get('possible_block', 0),
                                                    'Cut': events.get('possible_cut', 0),
                                                    'Post Up': events.get('possible_post_up', 0),
                                                    'Assist': events.get('possible_assist', 0),
                                                    'Turnover': events.get('possible_turnover', 0),
                                                    'Dribble Move': events.get('possible_dribble_move', 0),
                                                    'Def. Rotation': events.get('possible_defensive_rotation', 0)
                                                }

                                                # Show high confidence events (>0.5)
                                                high_conf_basic = [(k, v) for k, v in basic_events.items() if v > 0.5]
                                                high_conf_advanced = [(k, v) for k, v in advanced_events.items() if v > 0.5]

                                                if high_conf_basic or high_conf_advanced:
                                                    st.write("**Detected Events:**")

                                                    if high_conf_basic:
                                                        st.write("*Basic:*")
                                                        for event, prob in sorted(high_conf_basic, key=lambda x: -x[1]):
                                                            st.write(f"- {event}: {prob:.2f}")

                                                    if high_conf_advanced:
                                                        st.write("*Advanced:*")
                                                        for event, prob in sorted(high_conf_advanced, key=lambda x: -x[1]):
                                                            st.write(f"- {event}: {prob:.2f}")

                                                    # Show rebound type if applicable
                                                    rebound_type = events.get('rebound_type', 'none')
                                                    if rebound_type and rebound_type != 'none':
                                                        st.write(f"*Rebound:* {rebound_type.title()}")

                                with col2:
                                    # Video player
                                    show_video_player(
                                        video_path=video_path,
                                        start=result.segment.start,
                                        end=result.segment.end,
                                        player_id=f"search_player_{result.segment_id}",
                                        segment_id=result.segment_id,
                                        clips_dir=clips_dir
                                    )

                except RuntimeError as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.info("💡 **Troubleshooting:**\n"
                           "- This error may be due to NumPy/torch compatibility\n"
                           "- Try: `pip install 'numpy<2' torch torchvision`\n"
                           "- Or restart the Streamlit server")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    import logging
                    logging.exception("Search failed")


if __name__ == "__main__":
    main()
