# SAM3 Concept-Based Video Segmentation Module

## Overview

This module adds concept-driven video segmentation on top of SAM3. It provides:

- **Memory bank management** for tracking concept embeddings across video chunks
- **Chunk processing** with configurable overlap
- **Multiple output formats** (overlay, segmented regions, binary masks)
- **Standalone binary mask tools** for post-processing

## Architecture

### Core Components

```
concepts/
├── __init__.py                      # Package exports
├── config.py                        # Configuration dataclasses
├── utils.py                         # Image/video utility functions
├── sam3_concepts_segmenter.py       # Core segmentation strategy
├── mask_processor.py                # Flexible mask rendering
└── binary_mask_converter.py         # Binary mask operations
```

### CLI Entry Points

```
run_concept_segmentation.py           # Main orchestrator (end-to-end workflow)
run_binary_mask_converter.py          # Standalone binary mask tool
```

## Installation

The module requires SAM3 to be installed:

```bash
cd sam3
pip install -e .
```

Optional: Install development dependencies for linting and testing:

```bash
pip install ruff black mypy pytest
```

## Usage

### 1. Basic Concept Segmentation (End-to-End)

```bash
# Segment traffic signs in a video with overlay output
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign. speed limit sign" \
  --output output_overlay.mp4 \
  --output-mode overlay \
  --overlay-alpha 0.5
```

### 2. Segmented-Only Output (Masked Regions)

```bash
# Save only the segmented regions (black background elsewhere)
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit" \
  --output output_masked.mp4 \
  --output-mode segmented_only
```

### 3. Binary Mask Output

```bash
# Extract binary masks for further processing
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "traffic sign" \
  --output masks_dir/ \
  --output-mode binary_masks_only \
  --threshold 0.6
```

### 4. Advanced Chunk Strategy

```bash
# Custom chunk size and memory strategy with temporal decay
python run_concept_segmentation.py \
  --video long_video.mp4 \
  --concepts "traffic sign" \
  --output output.mp4 \
  --chunk-size 200 \
  --overlap 50 \
  --memory-strategy temporal_decay \
  --propagation-direction both
```

### 5. External Exemplar Injection (Text + Visual Fusion)

```bash
# Use one of the repo's sample exemplar images to disambiguate similar concepts
python run_concept_segmentation.py \
  --video input.mp4 \
  --concepts "speed limit 30 sign" \
  --exemplar-image ../SL30_image_examplar_00.webp \
  --exemplar-bbox "145,98,62,64" \
  --exemplar-placement letterbox \
  --exemplar-box-label 1 \
  --debug-exemplar-preview ./debug/preview.jpg \
  --output output.mp4 \
  --output-mode overlay
```

The exemplar image is injected as a pseudo-frame at the start of every chunk session.
The bbox is automatically remapped after fitting to the video resolution.
Pseudo-frame outputs are dropped and predictions are mapped back to original indices.
The repo ships six sample assets for this workflow: `../SL30_image_examplar_00.webp` through `../SL30_image_examplar_05.webp`.

### 6. Binary Mask Conversion (Standalone)

```bash
# Convert probability mask to binary with inversion (default threshold: 0.08)
python run_binary_mask_converter.py \
  --input mask_prob.png \
  --output mask_binary_inverted.png \
  --threshold 0.08 \
  --invert

# Apply morphological operations via config
python run_binary_mask_converter.py \
  --input mask.png \
  --output mask_closed.png \
  --dilate-kernel 3 \
  --erode-kernel 2

# Apply standalone morphological operation (open, close, dilate, erode)
python run_binary_mask_converter.py \
  --input mask.png \
  --output mask_opened.png \
  --morph-op open \
  --morph-kernel 5

# Get mask statistics
python run_binary_mask_converter.py \
  --input mask.png \
  --stats

# Batch process masks
python run_binary_mask_converter.py \
  --batch-dir ./raw_masks \
  --output-dir ./binary_masks \
  --threshold 0.6 \
  --invert \
  --verbose
```

### 7. Video Mask Processing (Standalone)

Process an entire mask video frame-by-frame (binarize, invert, morph ops):

```bash
# Convert a mask video to binary and invert
python run_binary_mask_converter.py \
  --input-video ./masks.mp4 \
  --output-video ./masks_binary.mp4 \
  --threshold 0.08 \
  --invert

# With custom codec and FPS
python run_binary_mask_converter.py \
  --input-video ./masks.mp4 \
  --output-video ./masks_binary.mp4 \
  --video-codec mp4v \
  --video-fps 30.0 \
  --morph-op close \
  --morph-kernel 5 \
  --verbose
```

Reports mean coverage percentage across all processed frames.
When processing compressed mask videos, thresholding operates on scaled grayscale intensities rather than treating every non-zero pixel as foreground.

### Runtime Stability Note

- Torch compile is **disabled by default** in `run_concept_segmentation.py`.
- Use `--compile` only if your runtime is stable with Torch Inductor.
- On some MIG/A100 environments, keeping compile disabled is more reliable.
- Use `--offload-video-to-cpu` on constrained/MIG GPUs for safer inference.
- Use `--disable-tf32` for stability at the cost of speed.
- Use `--max-num-objects N` to cap tracked objects and reduce memory pressure.
- Use `--no-inter-chunk-cuda-drain` to disable explicit GPU cleanup between chunks.
- Use `--inter-chunk-sleep-sec S` for optional delay between chunk processing.

## Configuration

### Memory Bank Strategies

The module supports three memory bank management strategies:

**1. RESET_PER_CHUNK** (default)
- Reset embeddings at chunk boundaries
- Best for independent chunk processing
- Lower memory usage

**2. CONTINUOUS**
- Keep embeddings across entire video
- Better temporal consistency
- Higher memory usage

**3. TEMPORAL_DECAY**
- Apply exponential decay to older embeddings
- Balance between consistency and independence
- Useful for very long videos

### Output Modes

**1. OVERLAY** (default)
- Alpha-blended masks on original frames
- Best for visual inspection
- Supports custom transparency

**2. SEGMENTED_ONLY**
- Isolated masked regions on black background
- Useful for detailed mask inspection
- Composite multiple masks automatically

**3. BINARY_MASKS_ONLY**
- Raw binary thresholded masks
- Saved as image sequence (PNG directory) or as a video file (when `--output` has a video extension)
- Suitable for downstream processing

### Propagation Directions

- **forward**: Process frames left-to-right (fastest)
- **backward**: Process frames right-to-left
- **both**: Bidirectional propagation (most accurate but slower)

## Configuration Classes

### ConceptSegmentationConfig

Controls core segmentation behavior:

```python
from concepts.config import (
    ConceptSegmentationConfig,
    PropagationDirection,
    MemoryStrategy,
    ExemplarPlacement,
)

config = ConceptSegmentationConfig(
    concepts=["traffic sign", "speed limit"],
    chunk_size=130,
    overlap=26,
    propagation_direction=PropagationDirection.FORWARD,
    memory_strategy=MemoryStrategy.RESET_PER_CHUNK,
    output_prob_thresh=0.5,
    apply_temporal_disambiguation=True,
    # Exemplar injection (optional):
  exemplar_image_path="../SL30_image_examplar_00.webp",
    exemplar_image_bbox=(145, 98, 62, 64),  # x, y, w, h in source image pixels
    exemplar_placement=ExemplarPlacement.LETTERBOX,
    exemplar_box_label=1,  # 1=positive, 0=negative
    debug_exemplar_preview_path="./debug/preview.jpg",
    # CUDA stability (optional):
    offload_video_to_cpu=False,
    disable_tf32=False,
    max_num_objects=None,
    inter_chunk_cuda_drain=True,
    inter_chunk_sleep_sec=0.0,
    # Model options:
    compile=False,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    async_loading_frames=False,
    video_loader_type="cv2",
)
```

### MaskProcessorConfig

Controls mask rendering:

```python
from concepts.config import MaskProcessorConfig, OutputMode

config = MaskProcessorConfig(
    output_mode=OutputMode.OVERLAY,
    overlay_alpha=0.45,
    max_labels_to_draw=8,
)
```

### BinaryMaskConfig

Controls binary mask conversion:

```python
from concepts.config import BinaryMaskConfig

config = BinaryMaskConfig(
    threshold=0.08,  # Default: 0.08 (low to capture soft masks)
    invert=False,
    dilate_kernel_size=None,
    erode_kernel_size=None,
)
```

### VideoIOConfig

Controls video codec and encoding:

```python
from concepts.config import VideoIOConfig

config = VideoIOConfig(
    video_codec="mp4v",
    quality=23,        # CRF; lower = better quality
    target_fps=None,   # None = same as input
)
```

### ExemplarPlacement Enum

How an external exemplar image is fitted to the target video resolution:

- **LETTERBOX**: Preserve aspect ratio, pad to video size, remap bbox with scale+offset.
- **CANVAS**: No resizing, center on padded canvas, remap bbox by offset only (source must fit target).

## Programmatic Usage

### End-to-End Segmentation

```python
from concepts.config import ConceptSegmentationConfig, ExemplarPlacement
from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy
from concepts.mask_processor import MaskProcessor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# Configure (exemplar fields optional)
config = ConceptSegmentationConfig(
    concepts=["speed limit sign"],
    chunk_size=130,
    overlap=26,
    # Optional exemplar injection:
    # exemplar_image_path="../SL30_image_examplar_00.webp",
    # exemplar_image_bbox=(145, 98, 62, 64),
    # exemplar_placement=ExemplarPlacement.LETTERBOX,
)

# Load video and plan chunks
strategy = ConceptSegmentationStrategy(config)
strategy.load_video("input.mp4")
strategy.plan_chunks()

# Run segmentation
predictor = Sam3VideoPredictor()
strategy.process_chunks(predictor)

# Get outputs
outputs = strategy.get_all_outputs()
print(strategy.get_summary())
```

### Custom Rendering

```python
from concepts.mask_processor import MaskProcessor
from concepts.config import MaskProcessorConfig, OutputMode

config = MaskProcessorConfig(output_mode=OutputMode.OVERLAY)
processor = MaskProcessor(config)
processor.set_palette(num_colors=2)

for frame_idx, frame in enumerate(frames):
    outputs = strategy.get_outputs_for_frame(frame_idx)
    processed_frame, labels = processor.process_frame(frame, outputs)
```

### Binary Mask Operations

```python
from concepts.binary_mask_converter import BinaryMaskConverter
from concepts.config import BinaryMaskConfig

config = BinaryMaskConfig(threshold=0.5, invert=True)
converter = BinaryMaskConverter(config)

# Convert and invert
binary = converter.convert_to_binary(probability_mask)

# Get statistics
stats = converter.get_mask_statistics(binary)
print(f"Coverage: {stats['coverage_percent']:.1f}%")

# Apply morphological cleaning
cleaned = converter.apply_morphological_ops(binary, operation="close", kernel_size=5)
```

## Advanced Topics

### Memory Bank Management

The `MemoryBank` class tracks concept embeddings across video chunks:

```python
from concepts.sam3_concepts_segmenter import MemoryBank, MemoryStrategy

bank = MemoryBank(
    concepts=["speed limit", "stop sign"],
    strategy=MemoryStrategy.TEMPORAL_DECAY
)

# Embeddings are accumulated during processing
# Reset per chunk or apply decay based on strategy
bank.reset()  # or bank.apply_temporal_decay(current_frame)

# Get statistics
print(bank.get_summary())
```

### Chunk Planning

The `ChunkPlanner` organizes frames into overlapping chunks:

```python
from concepts.sam3_concepts_segmenter import ChunkPlanner

planner = ChunkPlanner(
    num_frames=1000,
    chunk_size=130,
    overlap=26,
)

for chunk_start, chunk_end in planner.get_chunks():
    # Process frames [chunk_start, chunk_end)
    pass

print(planner.get_summary())
```

## Logging and Debugging

Enable verbose logging for troubleshooting:

```bash
# Main tool
python run_concept_segmentation.py --video input.mp4 --concepts "traffic sign" \
  --output output.mp4 --verbose

# Binary mask tool
python run_binary_mask_converter.py --input mask.png --output out.png --verbose
```

Check GPU memory usage during processing:

```bash
# Monitor in separate terminal
watch nvidia-smi
```

## Performance Tips

1. **Reduce chunk size** for lower memory usage (trade-off: more chunk boundaries)
2. **Use RESET_PER_CHUNK** strategy for memory-constrained environments
3. **Increase overlap** for smoother transitions between chunks (costs more compute)
4. **Use forward propagation** only for faster processing (backward is slower)
5. **Disable temporal disambiguation** if model errors occur

## Common Issues

### CUDA Out of Memory

- Reduce `--chunk-size`
- Switch to `--memory-strategy reset_per_chunk`
- Use `--offload-video-to-cpu` to keep frames on CPU
- Use `--disable-tf32` for more stable (but slower) computation on MIG GPUs
- Use `--max-num-objects 5000` to limit tracked objects
- Clear GPU cache manually: `torch.cuda.empty_cache()`

### CUDA Allocator Asserts (MIG/A100)

On some MIG setups, propagation can trigger PyTorch allocator asserts (`NVML_SUCCESS`/`CUDACachingAllocator`). The strategy layer automatically retries with progressive fallbacks:
1. Original config
2. + `offload_video_to_cpu=True`
3. + `disable_tf32=True`
4. + `max_num_objects=5000`

Manual mitigation: use `--offload-video-to-cpu --disable-tf32` from the start.

### Poor Mask Quality

- Increase `--threshold` (fewer false positives)
- Decrease `--threshold` (fewer false negatives)
- Use `--memory-strategy temporal_decay` for consistency
- Try `--propagation-direction both` for accuracy

### Slow Processing

- Use `--propagation-direction forward` only
- Increase `--chunk-size` (more frames per session)
- Reduce `--overlap`
- Disable `--no-temporal-disambiguation`

## References

- SAM3 Paper: [Segment Anything 3](https://arxiv.org/pdf/2511.16719)
- Facebook Research: [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)

## License

This module follows the same license as the SAM3 project (see LICENSE file in sam3 repo).
