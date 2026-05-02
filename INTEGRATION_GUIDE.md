# Integration Guide

## Overview

This guide explains how to use the `concepts/` module and its CLI tools, with practical examples for different scenarios.

---

## Quick Start

### Installation

```bash
# Clone the repo (already done)
cd sam3

# Install SAM3 with dependencies
pip install -e .

# Verify installation
python -c "from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy; print('✓ Installation successful')"
```

### Minimal Example: Traffic Sign Segmentation

```bash
# Segment speed limit signs in a video
python run_concept_segmentation.py \
  --video traffic_video.mp4 \
  --concepts "speed limit sign" \
  --output output.mp4
```

---

## Usage Scenarios

### Scenario 1: Quick Visual Inspection (Overlay Mode)

**Goal**: Quickly see what the model detects in your video

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign. speed limit. stop sign" \
  --output visualized.mp4 \
  --output-mode overlay \
  --overlay-alpha 0.5
```

**Output**: Video with semi-transparent colored overlays of detected objects

**When to use**: Quick validation, presentations, team review

---

### Scenario 2: Isolated Region Extraction (Segmented-Only Mode)

**Goal**: Extract only the detected regions (mask out everything else)

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign" \
  --output masked_only.mp4 \
  --output-mode segmented_only
```

**Output**: Video showing only masked regions on black background

**When to use**: Detailed mask analysis, mask quality inspection

**Use case example**: Verify if the model captures the full extent of traffic signs without false positives

---

### Scenario 3: Binary Mask Extraction (For Analysis/Post-Processing)

**Goal**: Get raw binary masks for further processing or analysis

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit" \
  --output ./masks_output/ \
  --output-mode binary_masks_only \
  --threshold 0.6
```

**Output**: Directory with PNG images for each frame's binary mask

**Alternative: Save binary masks as a video file**

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit" \
  --output ./masks_output.mp4 \
  --output-mode binary_masks_only \
  --threshold 0.6
```

When `--output` has a video extension (`.mp4`, `.avi`, etc.), binary masks are written as a video instead of a PNG directory.

**Follow-up: Process masks with converter**

```bash
# Invert masks (background becomes foreground)
python run_binary_mask_converter.py \
  --batch-dir ./masks_output \
  --output-dir ./masks_inverted \
  --invert

# Apply morphological cleaning
python run_binary_mask_converter.py \
  --batch-dir ./masks_output \
  --output-dir ./masks_cleaned \
  --dilate-kernel 3 \
  --erode-kernel 2

# Apply standalone morphological op
python run_binary_mask_converter.py \
  --batch-dir ./masks_output \
  --output-dir ./masks_closed \
  --morph-op close \
  --morph-kernel 5
```

**Follow-up: Process a mask video directly**

```bash
# Convert an entire mask video to binary + invert
python run_binary_mask_converter.py \
  --input-video ./masks_output.mp4 \
  --output-video ./masks_binary_inverted.mp4 \
  --threshold 0.08 \
  --invert \
  --verbose
```

**When to use**: Post-processing pipelines, downstream analysis, model evaluation

---

### Scenario 3b: Video Compositing with Binary Selector

**Goal**: Composite a segmented mask video onto a different base video (e.g., generated or edited) using a binary selector to control per-pixel source selection.

```bash
python scripts/overlay_with_binary_selector.py \
  --base-video cosmos_output.mp4 \
  --mask-video input_traffic_signs_masks.mp4 \
  --binary-video input_traffic_signs_binary.mp4 \
  --output-video composited_output.mp4
```

**How it works**:
- White pixels in the binary selector → pixels from the mask video (foreground).
- Black pixels in the binary selector → pixels from the base video (background).
- Threshold (default 127) controls binarization of the selector.

**Optional flags**:
- `--binary-inverted-video path.mp4` — Provide inverted selector for consistency checking.
- `--threshold 200` — Custom binarization threshold [0-255].
- `--max-frames 500` — Cap the number of frames processed.
- `--strict-size-check` — Fail on size mismatch instead of resizing.

**When to use**: Combining segmentation masks with world-model outputs (e.g., Cosmos), re-injecting detected regions into generated videos, or replacing backgrounds while preserving foreground.

---

### Scenario 4: Long Video Processing (Memory-Efficient Chunking)

**Goal**: Process a very long video without running out of GPU memory

```bash
python run_concept_segmentation.py \
  --video 30min_video.mp4 \
  --concepts "traffic sign" \
  --output output.mp4 \
  --chunk-size 150 \
  --overlap 30 \
  --memory-strategy reset_per_chunk
```

**Parameters explained**:
- `--chunk-size 150`: Process 150 frames at a time
- `--overlap 30`: 30-frame overlap between chunks (20%)
- `--memory-strategy reset_per_chunk`: Fresh start at each chunk (lowest memory)

**Tuning for your GPU**:
- 8GB VRAM: `--chunk-size 80 --overlap 15`
- 16GB VRAM: `--chunk-size 150 --overlap 30`
- 24GB VRAM: `--chunk-size 200 --overlap 40`

---

### Scenario 5: Maximum Accuracy (Bidirectional Propagation)

**Goal**: Achieve maximum segmentation accuracy at the cost of speed

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit sign" \
  --output output.mp4 \
  --propagation-direction both \
  --memory-strategy continuous \
  --chunk-size 500
```

**What this does**:
- `--propagation-direction both`: Process forward AND backward (slower but more accurate)
- `--memory-strategy continuous`: Keep full video in memory (maximum context)
- `--chunk-size 500`: Large chunks to reduce discontinuities

**When to use**: Critical applications, publication results, final validation

**Trade-off**: Much slower, requires more GPU memory

---

### Scenario 6: External Exemplar Injection (Text + Visual Fusion)

**Goal**: Condition each chunk with one external exemplar image and bbox while still using text concepts.

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit 30 sign" \
  --exemplar-image ../SL30_image_examplar_00.webp \
  --exemplar-bbox "145,98,62,64" \
  --exemplar-placement letterbox \
  --exemplar-box-label 1 \
  --debug-exemplar-preview ./debug/sl30_remap_preview.jpg \
  --output output.mp4 \
  --output-mode overlay
```

**What this does**:
- Uses your external exemplar image as a pseudo-frame at the start of every chunk session.
- Automatically remaps bbox coordinates after image fitting to the video resolution.
- Fuses text prompt and visual bbox cue in the same add-prompt request.
- Drops pseudo-frame outputs and maps predictions back to original video frame indices.

**Placement modes**:
- `letterbox` (recommended): Preserve aspect ratio, pad to video size, remap bbox automatically.
- `canvas`: No resizing, centered placement on padded canvas, remap bbox by offset only.

**Repo-provided sample exemplars**:
- From the `sam3/` directory, use `../SL30_image_examplar_00.webp` through `../SL30_image_examplar_05.webp`.
- Use `--exemplar-box-label 1` for a positive exemplar and `--exemplar-box-label 0` for a negative/exclusion exemplar.

**When to use**: Fine-grained disambiguation such as speed-limit-30 vs speed-limit-50/70.

---

## Programmatic Usage (Python API)

### Basic End-to-End Example

```python
from concepts.config import ConceptSegmentationConfig
from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy
from concepts.mask_processor import MaskProcessor, MaskProcessorConfig, OutputMode
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# 1. Configure segmentation
seg_config = ConceptSegmentationConfig(
    concepts=["speed limit sign"],
    chunk_size=130,
    overlap=26,
    exemplar_image_path="../SL30_image_examplar_00.webp",
    exemplar_image_bbox=(145, 98, 62, 64),
)

# 2. Load video and plan chunks
strategy = ConceptSegmentationStrategy(seg_config)
strategy.load_video("input.mp4")
strategy.plan_chunks()

# 3. Run segmentation
predictor = Sam3VideoPredictor()
strategy.process_chunks(predictor)

# 4. Get outputs
all_outputs = strategy.get_all_outputs()
print(f"Processed {len(all_outputs)} frames")

# 5. Render to output
proc_config = MaskProcessorConfig(output_mode=OutputMode.OVERLAY)
processor = MaskProcessor(proc_config)
processor.set_palette(num_colors=1)

output_frames = []
for frame_idx, frame in enumerate(strategy.video_frames):
    outputs = all_outputs.get(frame_idx, {})
    processed_frame, labels = processor.process_frame(frame, outputs)
    output_frames.append(processed_frame)

# 6. Save video
fps = strategy.video_metadata["fps"]
processor.save_video(output_frames, "output.mp4", fps)
```

---

### Advanced: Custom Memory Strategy

```python
from concepts.config import (
  ConceptSegmentationConfig,
  MemoryStrategy,
  PropagationDirection,
)

# Temporal decay for very long videos
config = ConceptSegmentationConfig(
    concepts=["traffic sign", "speed limit", "stop sign"],
    chunk_size=100,
    overlap=20,
    memory_strategy=MemoryStrategy.TEMPORAL_DECAY,
  propagation_direction=PropagationDirection.FORWARD,
)

strategy = ConceptSegmentationStrategy(config)
strategy.load_video("very_long_video.mp4")
strategy.plan_chunks()

# Monitor memory bank updates
def progress_callback(chunk_idx, total, msg):
    print(f"[{chunk_idx+1}/{total}] {msg}")
    memory_summary = strategy.memory_bank.get_summary()
    print(f"  Memory: {memory_summary}")

strategy.process_chunks(predictor, progress_callback=progress_callback)
```

---

### Advanced: Custom Rendering

```python
from concepts.mask_processor import MaskProcessor
from concepts.config import MaskProcessorConfig, OutputMode

# Create processor with custom settings
config = MaskProcessorConfig(
    output_mode=OutputMode.OVERLAY,
    overlay_alpha=0.3,  # More transparent
    max_labels_to_draw=5,
)
processor = MaskProcessor(config)

# Custom color palette
processor.palette = [
    (0, 255, 0),    # Green for traffic signs
    (255, 0, 0),    # Red for speed limits
    (0, 0, 255),    # Blue for stop signs
]

# Process each frame with custom logic
for frame_idx in range(len(strategy.video_frames)):
    frame = strategy.video_frames[frame_idx]
    outputs = strategy.get_outputs_for_frame(frame_idx)
    
    # Custom rendering
    processed_frame, labels = processor.process_frame(
        frame, outputs,
        concept_names=["traffic sign", "speed limit", "stop sign"]
    )
    
    # Custom post-processing
    # e.g., add additional annotations
```

---

### Advanced: Binary Mask Operations

```python
from concepts.config import BinaryMaskConfig
from concepts.binary_mask_converter import BinaryMaskConverter
import cv2

# Create converter
config = BinaryMaskConfig(
    threshold=0.5,
    invert=False,
    dilate_kernel_size=3,
    erode_kernel_size=None,
)
converter = BinaryMaskConverter(config)

# Process masks from segmentation
all_outputs = strategy.get_all_outputs()

for frame_idx, outputs in all_outputs.items():
    if "out_binary_masks" in outputs:
        # Get masks (might be numpy or torch)
        masks = outputs["out_binary_masks"]
        
        # Convert to binary with threshold
        binary = converter.convert_to_binary(masks, threshold=0.6)
        
        # Get statistics
        stats = converter.get_mask_statistics(binary)
        print(f"Frame {frame_idx}: {stats['coverage_percent']:.1f}% coverage")
        
        # Save
        converter.save_binary_mask(binary, f"mask_{frame_idx:06d}.png")

# Batch operations
batch_masks = {i: outputs["out_binary_masks"] 
               for i, outputs in all_outputs.items()}

binary_masks = converter.batch_convert_masks(batch_masks, threshold=0.6)
```

---

## Configuration Examples

### Config 1: Real-Time Processing (Speed Optimized)

```python
from concepts.config import ConceptSegmentationConfig, MemoryStrategy, PropagationDirection

config = ConceptSegmentationConfig(
    concepts=["traffic sign"],
    chunk_size=250,          # Large chunks = fewer boundaries
    overlap=20,              # Minimal overlap
    propagation_direction=PropagationDirection.FORWARD,
    memory_strategy=MemoryStrategy.RESET_PER_CHUNK,
    apply_temporal_disambiguation=False,  # Disable extra processing
    compile=True,           # Enable torch compile (opt-in)
)
```

**Best for**: Live processing, real-time applications, speed demos

Note: in CLI usage, compile is disabled by default for stability. Use `--compile`
explicitly only when your environment is stable with Torch Inductor.

---

### Config 2: Accuracy Optimized (Quality First)

```python
from concepts.config import ConceptSegmentationConfig, MemoryStrategy, PropagationDirection

config = ConceptSegmentationConfig(
    concepts=["traffic sign", "speed limit", "stop sign"],
    chunk_size=150,
    overlap=50,              # High overlap for smoothness
    propagation_direction=PropagationDirection.BOTH,
    memory_strategy=MemoryStrategy.CONTINUOUS,
    apply_temporal_disambiguation=True,  # Extra refinement
    compile=False,          # Keep compile off for stability/reproducibility
)
```

**Best for**: Final validation, publication results, critical applications

---

### Config 3: Memory Constrained (Limited VRAM)

```python
from concepts.config import ConceptSegmentationConfig, MemoryStrategy, PropagationDirection

config = ConceptSegmentationConfig(
    concepts=["speed limit"],
    chunk_size=80,           # Small chunks
    overlap=10,              # Minimal overlap
  propagation_direction=PropagationDirection.FORWARD,
  memory_strategy=MemoryStrategy.RESET_PER_CHUNK,
    async_loading_frames=True,  # Load frames progressively
)
```

**Best for**: GPU with < 8GB VRAM, embedded systems

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1: Reduce chunk size**
```bash
--chunk-size 80 --overlap 15
```

**Solution 2: Use reset strategy**
```bash
--memory-strategy reset_per_chunk
```

**Solution 3: Keep torch compile disabled**
```bash
# compile is already OFF by default; do not pass --compile
```

**Solution 4: Offload video frames to CPU**
```bash
--offload-video-to-cpu
```

**Solution 5: Disable TF32 (slower but more stable on MIG)**
```bash
--disable-tf32
```

**Solution 6: Limit tracked objects**
```bash
--max-num-objects 5000
```

If you previously enabled compile, remove `--compile` from your command.

### Issue: CUDA Allocator Asserts (MIG/A100)

The strategy layer automatically retries with progressive fallback strategies.
Manual mitigation:

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign" \
  --output output.mp4 \
  --offload-video-to-cpu \
  --disable-tf32 \
  --inter-chunk-sleep-sec 1.0
```

### Issue: PyTorch Inductor crash on MIG/A100

Symptoms may include assertions such as:

```text
Expected curr_block->next == nullptr to be true, but got false
```

Mitigation:
```bash
# stable path
# do not pass --compile
```

Only enable compile with `--compile` after confirming stable behavior in your runtime.

---

### Issue: Poor Mask Quality

**Hypothesis 1: Threshold too high**
```bash
--threshold 0.3  # Lower threshold
```

**Hypothesis 2: Single direction missing context**
```bash
--propagation-direction both
```

**Hypothesis 3: Chunk boundary artifacts**
```bash
--overlap 50  # Increase overlap
```

---

### Issue: Slow Processing

**Solution 1: Use forward propagation only**
```bash
--propagation-direction forward
```

**Solution 2: Increase chunk size**
```bash
--chunk-size 200
```

**Solution 3: Reduce overlap**
```bash
--overlap 15
```

## Best Practices

### 1. Always Validate with Dry-Run First

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign" \
  --output output.mp4 \
  --dry-run  # Only validates, no inference
```

### 2. Use Verbose Mode for Debugging

```bash
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign" \
  --output output.mp4 \
  --verbose
```

### 3. Start Conservative, Tune Later

1. Try default settings first
2. If memory issues → reduce chunk_size
3. If quality issues → increase overlap, try "both" propagation
4. If speed issues → increase chunk_size, use "forward" only

### 4. Keep Videos Reasonable Size

- < 1000 frames: Default settings work well
- 1k-10k: Tune chunk_size to 130-200
- 10k-100k: Use reset_per_chunk strategy
- > 100k: Consider splitting into multiple files

### 5. Document Your Configuration

```bash
# Save working config to file for reproducibility
cat > config_traffic_signs.sh << 'EOF'
python run_concept_segmentation.py \
  --video "$1" \
  --concepts "traffic sign. speed limit sign" \
  --output "${1%.mp4}_segmented.mp4" \
  --chunk-size 150 \
  --overlap 30 \
  --memory-strategy reset_per_chunk \
  --propagation-direction forward
EOF

chmod +x config_traffic_signs.sh
./config_traffic_signs.sh input_video.mp4
```

---

## Extending the Code

### Adding Custom Rendering Mode

```python
# In concepts/mask_processor.py
class MaskProcessor:
    def _process_custom_mode(self, frame, obj_ids, masks):
        """Custom rendering logic."""
        # Your implementation here
        return vis, labels
    
    def process_frame(self, frame, outputs, concept_names=None):
        # ... existing code ...
        elif self.config.output_mode == OutputMode.CUSTOM:
            return self._process_custom_mode(...)
```

### Adding Custom Memory Strategy

```python
# In concepts/sam3_concepts_segmenter.py
class MemoryBank:
    def apply_custom_strategy(self):
        """Your custom memory management logic."""
        # Your implementation here
```

---

## References

- [Original SAM3 Repository](https://github.com/facebookresearch/sam3)
- [Concepts Module README](./sam3/concepts/README.md)

