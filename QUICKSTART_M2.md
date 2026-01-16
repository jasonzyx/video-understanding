# M2 Quick Start Guide

## 🚀 Run the Complete M2 System

### Step 1: Enrich All Segments (one-time, ~15 mins, ~$1.50)

```bash
# Activate environment
source venv/bin/activate

# Run full enrichment pipeline
python build_index.py \
  --video data/nba_final.mp4 \
  --use-mllm \
  --extract-clips \
  --enrich-segments \
  --generate-embeddings \
  --stride 2.0 \
  --device cpu \
  --output-dir outputs

# This generates:
# ✓ 364 enriched segments
# ✓ Visual Faiss index (512-dim)
# ✓ Trajectory Faiss index (512-dim)
# ✓ Index mappings
```

### Step 2: Test Search (CLI)

```bash
# Simple search
python search_segments.py --query "pick and roll"

# With detailed scores
python search_segments.py \
  --query "drive to basket" \
  --show-scores \
  --show-metadata \
  --top-k 5

# Examples to try:
python search_segments.py --query "screen action" --show-scores
python search_segments.py --query "fast break" --show-scores
python search_segments.py --query "corner three" --show-scores
python search_segments.py --query "player cuts to basket" --show-scores
```

### Step 3: Launch Viewer

```bash
streamlit run viewer/app.py

# In the browser:
# 1. Click "Search Segments" tab
# 2. Enter query: "pick and roll"
# 3. Adjust fusion weights (optional)
# 4. View results with score breakdown
# 5. Watch matched video clips
```

---

## 🎬 Demo with Test Segments (Already Works!)

If you want to test immediately without full enrichment:

```bash
# Run the complete demo (uses 3 already-enriched segments)
python demo_m2_complete.py

# Output shows:
# ✓ Enrichment working (GPT-4V + CLIP)
# ✓ Multi-modal search working
# ✓ Score fusion working
# ✓ Example queries with results
```

---

## 🔍 Example Queries to Try

### Action-Based
- "pick and roll"
- "screen action"
- "drive to basket"
- "player cuts to basket"
- "fast break"
- "transition play"

### Location-Based
- "action at top of key"
- "corner three pointer"
- "paint attack"
- "wing play"

### Combination
- "screen near the key"
- "drive from wing"
- "fast break dunk"

---

## 📊 Understanding Results

### Score Breakdown
- **Combined**: Final fused score (0.4*vis + 0.3*traj + 0.2*evt + 0.1*meta)
- **Visual**: CLIP similarity to query text
- **Trajectory**: Motion description similarity
- **Events**: Weak event detection match
- **Metadata**: Zone/spatial filter match

### Enriched Metadata Display
- **Motion**: GPT-4V description of player movements
- **Court Semantics**: ball_zone, primary_zone, paint_occupied
- **Metrics**: ball_speed, spacing, motion_intensity
- **Events**: High confidence actions (screen, drive, shot, etc.)

---

## ⚙️ Customization

### Tune Fusion Weights

For queries emphasizing different aspects:

```bash
# Visual-heavy (formations, positions)
python search_segments.py \
  --query "defensive setup" \
  --visual-weight 0.6 \
  --trajectory-weight 0.2 \
  --events-weight 0.1 \
  --metadata-weight 0.1

# Motion-heavy (actions, movement)
python search_segments.py \
  --query "cutting action" \
  --visual-weight 0.2 \
  --trajectory-weight 0.5 \
  --events-weight 0.2 \
  --metadata-weight 0.1

# Event-heavy (specific actions)
python search_segments.py \
  --query "screen" \
  --visual-weight 0.3 \
  --trajectory-weight 0.2 \
  --events-weight 0.4 \
  --metadata-weight 0.1
```

### Apply Filters

```bash
# Only 4s+ segments
python search_segments.py \
  --query "pick and roll" \
  --min-duration 4.0

# High confidence events only
python search_segments.py \
  --query "screen" \
  --min-event-score 0.7

# Specific duration range
python search_segments.py \
  --query "fast action" \
  --min-duration 2.0 \
  --max-duration 6.0
```

---

## 🐛 Troubleshooting

### Indices not found
```
ERROR: Indices not found. Run build_index.py with --enrich-segments --generate-embeddings first.
```
→ Run Step 1 above to generate indices

### NumPy compatibility error
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
→ Fix: `pip install 'numpy<2' opencv-python==4.9.0.80`

### GPU/CUDA errors
→ Add `--device cpu` to build_index.py command

### Slow enrichment
→ Normal - takes ~15-20 mins for 364 segments
→ Each segment: GPT-4V call (~2s) + CLIP encoding (~0.1s)

---

## 📈 Performance

- **Enrichment**: ~2.5s per segment × 364 = ~15 minutes
- **Index building**: ~5 seconds
- **Search**: <20ms per query (Stage 1 only)
- **Re-ranking**: ~2s per query (Stage 2, optional)

---

## 💡 Tips

1. **Start with test demo** to understand the system
2. **Run full enrichment** once, reuse indices
3. **Experiment with weights** for different query types
4. **Use viewer** for visual exploration
5. **Check metadata** to understand why segments matched

---

**Ready to go!** Run `python demo_m2_complete.py` to see it working immediately.
