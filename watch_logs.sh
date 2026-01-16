#!/bin/bash
# Watch Streamlit logs in real-time
echo "Watching Streamlit logs for reranking activity..."
echo "Search for something with reranking enabled and you'll see the logs here"
echo "============================================================"
tail -f /tmp/claude/-Users-jasonxu-workspace-video-understanding/tasks/b8dc97d.output | grep --line-buffered -i "rerank\|gemini\|Loading\|error"
