# SAM3 Concept-Based Video Segmentation Module

## Overview

This module provides a professional, modular architecture for concept-based semantic video segmentation using SAM3. It enables sophisticated video analysis with:

- **Memory bank management** for tracking concept embeddings across video chunks
- **Flexible chunk processing** with configurable overlap and temporal strategies
- **Multiple output formats** (overlay, segmented regions, binary masks)
- **Standalone binary mask tools** for post-processing and mask inversion
- **Comprehensive documentation** and type hints throughout

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

### 5. Binary Mask Conversion (Standalone)

```bash
# Convert probability mask to binary with inversion
python run_binary_mask_converter.py \
  --input mask_prob.png \
  --output mask_binary_inverted.png \
  --threshold 0.5 \
  --invert

# Apply morphological operations
python run_binary_mask_converter.py \
  --input mask.png \
  --output mask_closed.png \
  --dilate-kernel 3 \
  --erode-kernel 2

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
- Saved as image sequence or PNG stack
- Suitable for downstream processing

### Propagation Directions

- **forward**: Process frames left-to-right (fastest)
- **backward**: Process frames right-to-left
- **both**: Bidirectional propagation (most accurate but slower)

## Configuration Classes

### ConceptSegmentationConfig

Controls core segmentation behavior:

```python
from concepts.config import ConceptSegmentationConfig, PropagationDirection, MemoryStrategy

config = ConceptSegmentationConfig(
    concepts=["traffic sign", "speed limit"],
    chunk_size=130,
    overlap=26,
    propagation_direction=PropagationDirection.FORWARD,
    memory_strategy=MemoryStrategy.RESET_PER_CHUNK,
    output_prob_thresh=0.5,
    apply_temporal_disambiguation=True,
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
    threshold=0.5,
    invert=False,
    dilate_kernel_size=None,
    erode_kernel_size=None,
)
```

## Programmatic Usage

### End-to-End Segmentation

```python
from concepts.config import ConceptSegmentationConfig
from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy
from concepts.mask_processor import MaskProcessor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# Configure
config = ConceptSegmentationConfig(
    concepts=["speed limit sign"],
    chunk_size=130,
    overlap=26,
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
- Clear GPU cache manually: `torch.cuda.empty_cache()`

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
