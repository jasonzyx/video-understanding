# Sports Video Moment Retrieval

A system for temporal retrieval of sports moments (e.g., "find all pick-and-rolls").

## Project Status: M1 - Temporal Substrate

Currently implementing the foundational temporal indexing layer.

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

**Option A: MLLM-Based Detection (Recommended) 🚀**

Uses GPT-4 Vision for intelligent possession detection:

```bash
export OPENAI_API_KEY="sk-..."

python build_index.py \
  --video data/your_game.mp4 \
  --use-mllm
```

Tune for cost/accuracy:
```bash
python build_index.py \
  --video data/your_game.mp4 \
  --use-mllm \
  --mllm-model gpt-4o-mini \      # Cheaper option
  --mllm-sample-rate 3.0 \        # Sample less frequently (lower cost)
  --mllm-batch-size 8             # More frames per request
```

**Available models:**
- `gpt-4o` - Best quality (recommended)
- `gpt-4-turbo` - Good balance
- `gpt-4o-mini` - Most cost-effective

**Estimated costs:** $0.10-0.50 per 10min video

**Option B: Basic Auto-Detection**

Uses OpenCV motion detection (free, local, less accurate):

```bash
python build_index.py \
  --video data/your_game.mp4 \
  --auto-detect-possessions
```

Tune detection parameters if needed:
```bash
python build_index.py \
  --video data/your_game.mp4 \
  --auto-detect-possessions \
  --min-possession 5.0 \
  --max-possession 30.0 \
  --activity-threshold 2.0
```

**Option C: Manual Possession Config**

Create a YAML config (see `data/example_possessions.yaml`):
```yaml
possessions:
  - start: 0.0
    end: 12.0
```

Then run:
```bash
python build_index.py \
  --video data/your_game.mp4 \
  --config data/your_possessions.yaml
```

This creates:
- `outputs/{game_id}_possessions.jsonl` - possession boundaries
- `outputs/{game_id}_segments.jsonl` - candidate segments
- `outputs/{game_id}_index_metadata.json` - index metadata

### 4. Launch Viewer

Explore segments interactively:
```bash
streamlit run viewer/app.py
```

Open browser to `http://localhost:8501`

## Project Structure

```
video-understanding/
├── data/                      # Video files & configs
│   └── example_possessions.yaml
├── outputs/                   # Generated indexes (gitignored)
├── src/
│   ├── data_models.py                # Possession & Segment classes
│   ├── segment_generator.py          # Multi-scale window generation
│   ├── frame_extractor.py            # Video frame sampling
│   ├── mllm_possession_detector.py   # GPT-4V-based detection
│   └── possession_detector.py        # Basic OpenCV detection
├── viewer/
│   └── app.py                # Streamlit viewer
├── build_index.py            # Main indexing CLI
└── requirements.txt
```

## Roadmap

- [x] **M1** - Temporal substrate (possessions → segments + viewer)
- [ ] **M2** - High-recall retrieval (find candidates for queries)
- [ ] **M3** - Temporal grounding + verification (precise boundaries)
- [ ] **M4** - Multi-concept expansion + captioning

## Testing Possession Detection

Before building the full index, test the possession detector:

**Test MLLM detector (recommended):**
```bash
export OPENAI_API_KEY="sk-..."

python test_detector.py \
  --video data/your_game.mp4 \
  --use-mllm
```

**Test basic detector:**
```bash
python test_detector.py --video data/your_game.mp4
```

This will show detected possession boundaries without generating segments.

**Tuning tips:**
- MLLM: Adjust `--mllm-sample-rate` (higher = fewer samples, lower cost)
- Basic: Lower `--activity-threshold` for more possessions
- Both: Increase `--min-possession` to filter short periods

## M1 Definition of Done

Can you answer: "Show me 50 random segments from this game and scrub through them"?

If yes, M1 is complete.
