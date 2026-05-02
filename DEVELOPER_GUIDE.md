# Developer Guide: Code Structure & Maintenance

## Overview

This guide explains the internal structure of the `concepts/` module for developers who want to understand, maintain, or extend the codebase.

---

## Directory Structure

```
segmentation_with_concepts/
├── README.md                            # Workspace overview
├── INTEGRATION_GUIDE.md                 # Practical usage examples
├── DEVELOPER_GUIDE.md                   # This file
├── SL30_image_examplar_*.webp           # Sample external exemplar assets (repo root)
└── sam3/
│   ├── concepts/                        # Concept segmentation module
│   │   ├── __init__.py                  # Package exports
│   │   ├── config.py                    # Configuration dataclasses
│   │   ├── utils.py                     # Image/video utilities
│   │   ├── sam3_concepts_segmenter.py   # Core strategy classes
│   │   ├── mask_processor.py            # Rendering module
│   │   ├── binary_mask_converter.py     # Binary mask tools
│   │   └── README.md                    # Module documentation
│   ├── run_concept_segmentation.py      # Main CLI tool
│   ├── run_binary_mask_converter.py     # Binary mask CLI tool
│   ├── scripts/
│   │   └── overlay_with_binary_selector.py  # Binary-guided compositing tool
│   ├── model/                           # SAM3 model files
│   ├── examples/                        # SAM3 examples
│   └── ... (other SAM3 files)
```

---

## Module Responsibilities

### `config.py` - Configuration Layer

**Purpose**: Define all user-configurable options with validation

**Key Classes**:
- `PropagationDirection` - Enum for forward/backward/both
- `OutputMode` - Enum for overlay/segmented/binary
- `MemoryStrategy` - Enum for reset/continuous/decay
- `ExemplarPlacement` - Enum for letterbox/canvas (exemplar image fitting mode)
- `ConceptSegmentationConfig` - Main segmentation parameters (includes exemplar injection, CUDA stability, and compile settings)
- `MaskProcessorConfig` - Rendering parameters
- `BinaryMaskConfig` - Binary conversion parameters (default threshold: 0.08)
- `VideoIOConfig` - Video codec, quality, and FPS settings

**Design Principles**:
- ✅ Use enums for safety (prevent invalid values)
- ✅ All parameters have sensible defaults
- ✅ Explicit `validate()` method for consistency checks
- ✅ Type hints for IDE autocomplete

**Adding New Configuration**:

```python
@dataclass
class NewFeatureConfig:
    """Your new feature configuration."""
    param1: str = "default"
    param2: int = 10
    
    def validate(self):
        """Validate consistency."""
        if self.param2 <= 0:
            raise ValueError("param2 must be > 0")
        return True
```

---

### `utils.py` - Utility Functions

**Purpose**: Reusable image/video operations

**Function Categories**:

1. **Color Utilities**
   - `make_color_palette()` - Generate distinct colors

2. **Mask Operations**
   - `normalize_mask()` - Convert to bool numpy array
   - `resize_mask_to_frame()` - Resize mask to frame dimensions
   - `overlay_mask_on_frame()` - Alpha-blend mask on frame
   - `extract_segmented_region()` - Isolate masked area
   - `compose_multiple_masks()` - Combine multiple masks

3. **Video Operations**
   - `get_video_metadata()` - Extract fps, resolution
   - `load_video_frames()` - Load all frames to memory

4. **Drawing**
   - `draw_label_on_frame()` - Render text with outline

**Design Principles**:
- ✅ Pure functions where possible (no side effects)
- ✅ Handle edge cases (missing files, invalid shapes)
- ✅ Use type hints and docstrings
- ✅ Avoid dependencies on SAM3 (reusable elsewhere)

**Adding New Utility**:

```python
def new_utility_function(param1: np.ndarray, param2: int) -> np.ndarray:
    """
    Brief description.
    
    Args:
        param1: Description.
        param2: Description.
    
    Returns:
        Processed result.
    
    Raises:
        ValueError: If validation fails.
    """
    # Validate inputs
    if param2 <= 0:
        raise ValueError("param2 must be > 0")
    
    # Implementation
    result = param1 * param2
    
    return result
```

---

### `sam3_concepts_segmenter.py` - Core Strategy

**Purpose**: Orchestrate SAM3 inference with smart chunking and memory management

**Key Classes**:

1. **MemoryBank**
   - Tracks concept embeddings metadata
   - Supports 3 strategies: reset/continuous/decay
   - **Not** fed into SAM3, user-level bookkeeping only

2. **ChunkPlanner**
   - Splits video into overlapping chunks
   - Validates overlap < chunk_size
   - Computes step size automatically

3. **ConceptSegmentationStrategy**
   - Main orchestration class
   - Coordinates video loading, chunking, prediction, aggregation
   - Exposes clear API: load → plan → process → get_outputs
   - **Exemplar injection**: Supports external image as pseudo-frame per chunk (letterbox/canvas fitting, bbox remapping)
   - **CUDA stability**: Progressive fallback retry on allocator asserts (offload → disable TF32 → reduce objects)
   - **Inter-chunk drain**: Explicit GPU cleanup and optional sleep between chunks

**Architecture Notes**:
- ✅ User-level orchestration: controls when SAM3 sessions are opened and closed
- ✅ SAM3 propagation algorithm is not modified
- ✅ Clear separation of concerns
- ✅ Extensible for future enhancements

**Key Methods**:

```python
class ConceptSegmentationStrategy:
    def load_video(video_path)           # Stage 1: Load + exemplar prep
    def plan_chunks()                    # Stage 2: Plan
    def process_chunks(predictor, progress_callback=None)  # Stage 3: Process (with CUDA retry)
    def get_outputs_for_frame(idx)       # Stage 4: Query results
    def get_all_outputs()                # Stage 4: Get all results
    def get_summary()                    # Reporting
    def dry_run()                        # Validation

    # Internal (exemplar injection):
    def _prepare_exemplar_prompt_if_needed()   # Fit exemplar image + remap bbox
    def _build_pseudo_chunk_video(start, end)  # Temp video with exemplar as frame 0
    def _add_initial_prompt(predictor, ...)     # Text and/or visual prompt seeding

    # Internal (CUDA stability):
    def _process_chunk(predictor, start, end)  # Progressive fallback retry
    def _drain_between_chunks(has_next)        # GPU drain + optional sleep
    def _safe_cuda_cleanup()                   # gc + empty_cache + ipc_collect
    def _configure_session_stability(...)      # TF32 / max_num_objects
```

**Extending ConceptSegmentationStrategy**:

```python
class ConceptSegmentationStrategy:
    # Add new method for custom logic
    def process_chunks_with_callback(self, predictor, callback):
        """Process chunks with custom callback."""
        for chunk_idx, (start, end) in enumerate(self.chunk_planner.get_chunks()):
            # Your custom logic
            callback(chunk_idx, start, end)
            self._process_chunk(predictor, start, end)
```

**Exemplar Injection Architecture**:

When `exemplar_image_path` and `exemplar_image_bbox` are configured:
1. `load_video()` calls `_prepare_exemplar_prompt_if_needed()` which fits the exemplar to video resolution (letterbox or canvas) and remaps the bbox.
2. Each chunk creates a temporary video with the fitted exemplar as frame 0 via `_build_pseudo_chunk_video()`.
3. `_add_initial_prompt()` sends both text and visual bbox cues in a single `add_prompt` request.
4. Propagation results for frame 0 (pseudo-frame) are dropped; remaining indices are offset by -1 to map back to the original video.

For this repo specifically, the root-level files `SL30_image_examplar_00.webp` through `SL30_image_examplar_05.webp` are sample assets for this workflow.

**CUDA Fallback Strategy**:

`_process_chunk()` implements progressive retry for allocator asserts (common on MIG/A100):
1. Original config settings
2. + `offload_video_to_cpu=True`
3. + `disable_tf32=True`
4. + `max_num_objects=5000`

Between chunks, `_drain_between_chunks()` runs `gc.collect()`, `torch.cuda.empty_cache()`, `ipc_collect()`, and optional sleep.

---

### `mask_processor.py` - Rendering Layer

**Purpose**: Flexible mask rendering with multiple output formats

**Key Class**: `MaskProcessor`

**Design Pattern**: Strategy pattern for output modes

```python
class MaskProcessor:
    def process_frame(self, frame, outputs):
        if self.config.output_mode == OutputMode.OVERLAY:
            return self._process_overlay(...)
        elif self.config.output_mode == OutputMode.SEGMENTED_ONLY:
            return self._process_segmented_only(...)
        elif self.config.output_mode == OutputMode.BINARY_MASKS_ONLY:
            return self._process_binary_masks(...)
```

**Key Methods**:
- `set_palette()` - Configure colors
- `process_frame()` - Process single frame
- `save_video()` - Write output video with codec fallback (`mp4v`, `XVID`, `MJPG`, `DIVX`, `WMV1`)
- `save_mask_frames()` - Write mask sequence after collapsing multi-object stacks to one per-frame binary mask

**Adding New Output Mode**:

1. Add to `OutputMode` enum in `config.py`
2. Add `_process_new_mode()` method in `MaskProcessor`
3. Update `process_frame()` switch statement
4. Document in README

---

### `binary_mask_converter.py` - Post-Processing

**Purpose**: Manipulate masks (threshold, invert, morphology)

**Key Class**: `BinaryMaskConverter`

**Key Methods**:
- `convert_to_binary()` - Threshold conversion
- `invert_mask()` - Foreground/background swap
- `apply_morphological_ops()` - Dilation, erosion, open, close
- `merge_masks()` - Combine multiple masks
- `get_mask_statistics()` - Coverage, contours, etc.
- `extract_connected_components()` - Find separate objects

**Design Principles**:
- ✅ Decoupled from segmentation (can work on any masks)
- ✅ Batch operations support
- ✅ Comprehensive statistics
- ✅ `convert_to_binary()` preserves numeric mask intensities before thresholding, which avoids treating codec artifacts in mask videos as foreground

**Adding New Operation**:

```python
def new_operation(self, mask, param):
    """
    New mask operation.
    
    Args:
        mask: Input mask.
        param: Operation parameter.
    
    Returns:
        Processed mask.
    """
    mask_uint8 = normalize_mask(mask).astype(np.uint8) * 255
    # Your implementation
    return result
```

---

## CLI Tools Structure

### `run_concept_segmentation.py` - Main Orchestrator

**Flow**:
1. Parse arguments → `build_argument_parser()`
2. Validate configuration
3. Load video → `ConceptSegmentationStrategy.load_video()`
4. Plan chunks → `ConceptSegmentationStrategy.plan_chunks()`
5. Process → `ConceptSegmentationStrategy.process_chunks()`
6. Render → `MaskProcessor.process_frame()`
7. Save → `MaskProcessor.save_video()` or `MaskProcessor.save_mask_frames()`

**Notable current behavior**:
- External exemplar prompting is enabled through `--exemplar-image`, `--exemplar-bbox`, `--exemplar-placement`, `--exemplar-box-label`, and `--debug-exemplar-preview`.
- Binary-mask mode can save either a PNG directory or a mask video depending on whether `--output` has a video extension.
- Stability flags include `--offload-video-to-cpu`, `--disable-tf32`, `--max-num-objects`, `--no-inter-chunk-cuda-drain`, and `--inter-chunk-sleep-sec`.

**Adding New CLI Flag**:

```python
parser.add_argument(
    "--new-flag",
    type=str,
    default="default_value",
    help="Description of what this does",
)

# Then use in code:
args.new_flag
```

### `run_binary_mask_converter.py` - Mask Tool

**Three input modes** (mutually exclusive):
1. **Single image** (`--input`): Process one mask PNG. Supports `--stats` for statistics-only.
2. **Batch directory** (`--batch-dir`): Process all masks matching `--file-pattern` (default `*.png`).
3. **Video** (`--input-video`): Process a mask video frame-by-frame.

**Flow (single/batch)**:
1. Parse arguments
2. Validate input paths
3. Load mask(s)
4. Apply conversions → `BinaryMaskConverter` (threshold, invert, dilate, erode)
5. Optionally apply `--morph-op` (open/close/dilate/erode) with `--morph-kernel`
6. Save result(s)

**Flow (video mode)**:
1. Parse arguments (`--input-video`, `--output-video`)
2. Open input video, read frames sequentially
3. Convert each frame to grayscale → apply morph ops → binarize
4. Write processed frames to output video (`--video-codec`, `--video-fps`)
5. Report mean coverage percentage across all frames

**Key defaults**: threshold=0.08, codec=mp4v

**Design**: Supports single-file, batch, and video processing modes

### `scripts/overlay_with_binary_selector.py` - Video Compositing Tool

**Purpose**: Composite a segmented mask video onto a base video using a binary per-pixel selector.

**Flow**:
1. Parse arguments (base video, mask video, binary selector, output)
2. Open all input videos and validate sizes
3. For each frame: threshold binary selector → use white pixels to pick from mask video, black from base
4. Write composited frames to output video

**Key Features**:
- Automatic resizing of selector/mask if sizes differ (unless `--strict-size-check`)
- Optional inverted binary selector for sanity-checking complement consistency
- Configurable threshold [0-255] for binarization
- Frame cap via `--max-frames`

**Design**: Standalone script, no dependency on `concepts/` module. Works with any three input videos of matching frame counts.

---

## Testing Strategy

### Current State
No formal tests yet (future enhancement)

### Recommended Additions

```python
# tests/test_config.py
def test_concept_segmentation_config_validation():
    config = ConceptSegmentationConfig(chunk_size=-1)
    with pytest.raises(ValueError):
        config.validate()

# tests/test_utils.py
def test_normalize_mask():
    torch_mask = torch.rand(10, 10)
    numpy_mask = normalize_mask(torch_mask)
    assert isinstance(numpy_mask, np.ndarray)
    assert numpy_mask.dtype == bool

# tests/test_integration.py
def test_end_to_end_workflow(sample_video):
    config = ConceptSegmentationConfig(concepts=["test"])
    strategy = ConceptSegmentationStrategy(config)
    strategy.load_video(sample_video)
    strategy.plan_chunks()
    # ... test processing
```

---

## Code Style Guidelines

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief one-line description.
    
    Longer description if needed (optional).
    Can span multiple lines.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When input is invalid.
        RuntimeError: When operation fails.
    """
```

### Type Hints

Always include type hints:

```python
# Good
def process_frame(self, frame: np.ndarray, idx: int) -> np.ndarray:
    pass

# Avoid
def process_frame(self, frame, idx):
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ConceptSegmentationStrategy`)
- **Functions**: `snake_case` (e.g., `normalize_mask`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CHUNK_SIZE`)
- **Private**: Leading underscore (e.g., `_process_chunk`)

### Comments

Use comments sparingly, code should be self-documenting:

```python
# Good: explain WHY
# We use overlap to smooth transitions at chunk boundaries
overlap_frames = overlap

# Avoid: state the obvious
# Convert chunk_size to integer
chunk_size = int(chunk_size)

# Good: document non-obvious design decisions
# ARCHITECTURAL NOTE: MemoryBank is user-level orchestration,
# not fed into SAM3's model. SAM3 handles temporal memory internally.
```

---

## Performance Considerations

### Memory Usage

**Track sources**:
- Video frames in memory: O(num_frames × frame_size)
- SAM3 model: ~7GB
- Masks: O(num_frames × H × W × data_type)

**Optimization opportunities**:
- Stream frames instead of loading all to memory
- Use lower precision (float16) where possible
- Release intermediate results aggressively

### Computation

**Bottlenecks**:
- SAM3 forward passes (model inference) - 90% of time
- Mask post-processing - 5% of time
- Video I/O - 5% of time

**Safe optimizations**:
- Increase chunk_size (fewer sessions)
- Use forward-only propagation
- Enable model compilation

---

## Debugging Techniques

### Enable Verbose Logging

```python
# In run_concept_segmentation.py
if args.verbose:
    print(f"[DEBUG] Config: {segmentation_config.__dict__}", file=sys.stderr)
    print(f"[DEBUG] Video metadata: {strategy.video_metadata}", file=sys.stderr)
```

### Checkpoint Processing

```python
# Save intermediate results
if (chunk_idx % 5) == 0:
    with open(f"checkpoint_chunk_{chunk_idx}.pkl", "wb") as f:
        pickle.dump(per_frame_outputs, f)
```

### Profile Execution

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
strategy.process_chunks(predictor)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

---

## Common Maintenance Tasks

### Adding a New Configuration Option

1. Add to dataclass in `config.py`
2. Add to argparse in CLI tool(s)
3. Use in relevant class (e.g., `ConceptSegmentationStrategy`)
4. Update docstrings
5. Update README

### Adding a New Output Format

1. Add to `OutputMode` enum in `config.py`
2. Implement `_process_new_format()` in `MaskProcessor`
3. Update `process_frame()` method
4. Test with sample data

### Updating SAM3 Compatibility

If SAM3 API changes:
1. Update `sam3_concepts_segmenter._process_chunk()`
2. Check `ConceptSegmentationConfig` parameters
3. Test backward compatibility if possible
4. Document breaking changes

---

## Version Management

### Semantic Versioning

Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Example progression:
- `0.1.0` - Initial release (preview)
- `0.1.1` - Bug fixes
- `0.2.0` - New output format
- `1.0.0` - Stable release

---

## Documentation Updates

When making changes, update:
1. **Code docstrings** - Always
2. **concepts/README.md** - For feature additions
3. **INTEGRATION_GUIDE.md** - For API changes
4. **This file** - For structural changes

---

## Future Enhancement Ideas

1. **Checkpoint/Resume**
   - Save per-chunk results
   - Resume interrupted jobs

2. **Adaptive Chunk Sizing**
   - Monitor GPU memory
   - Adjust chunk_size dynamically

3. **Multi-GPU Support**
   - Distribute chunks across GPUs

4. **Distributed Processing**
   - Remote workers process chunks

5. **Interactive Mode**
   - Web UI for real-time configuration

6. **Formal Unit Tests**
   - pytest-based test suite

7. **Configuration Serialization**
   - YAML config file support

8. **Result Caching**
   - Cache predictions to avoid recomputation

---

## References

- Python style guide: [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Type hints: [PEP 484](https://www.python.org/dev/peps/pep-0484/)
- Dataclasses: [PEP 557](https://www.python.org/dev/peps/pep-0557/)
- SAM3: [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)

