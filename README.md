# Sports Video Moment Retrieval

A system for temporal retrieval of sports moments with natural language queries (e.g., "find all pick-and-rolls").

## Features

- **Temporal Segmentation**: Automatic possession detection and multi-scale windowing
- **Multi-Modal Search**: Visual + trajectory + metadata fusion with AI reranking
- **Interactive Viewer**: Browse possessions, segments, and search results with video playback

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your video file in the `data/` directory:
```bash
data/your_game.mp4
```

### 3. Build Index

**Full pipeline with search enabled:**

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="your-gemini-key"  # For AI reranking

python build_index.py \
  --video data/your_game.mp4 \
  --use-mllm \
  --extract-clips \
  --enrich-segments \
  --generate-embeddings \
  --stride 2.0
```

**Quick start without search:**

```bash
export OPENAI_API_KEY="sk-..."

python build_index.py \
  --video data/your_game.mp4 \
  --use-mllm
```

**Options:**
- `--use-mllm`: Use GPT-4 Vision for possession detection (recommended)
- `--auto-detect-possessions`: Use OpenCV motion detection (free, less accurate)
- `--extract-clips`: Extract video clips for each segment
- `--enrich-segments`: Add trajectory and event metadata
- `--generate-embeddings`: Build FAISS indices for search
- `--stride`: Frame sampling rate for embeddings (default: 2.0)

### 4. Launch Viewer

**On macOS (recommended):**

```bash
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
streamlit run viewer/app.py --server.headless=true
```






## Troubleshooting

**Viewer crashes on macOS:**
Use the environment variables when starting:
```bash
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
streamlit run viewer/app.py --server.headless=true
```
