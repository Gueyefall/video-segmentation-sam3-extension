# Developer Guide: Code Structure & Maintenance

## Overview

This guide explains the internal structure of the `concepts/` module for developers who want to understand, maintain, or extend the codebase.

---

## Directory Structure

```
segmentation_with_concepts/
├── INTEGRATION_GUIDE.md                 # Practical usage examples
├── sam3/
│   ├── concepts/                        # Our new module
│   │   ├── __init__.py                  # Package exports
│   │   ├── config.py                    # Configuration dataclasses
│   │   ├── utils.py                     # Image/video utilities
│   │   ├── sam3_concepts_segmenter.py   # Core strategy classes
│   │   ├── mask_processor.py            # Rendering module
│   │   ├── binary_mask_converter.py     # Binary mask tools
│   │   └── README.md                    # Module documentation
│   ├── run_concept_segmentation.py      # Main CLI tool
│   ├── run_binary_mask_converter.py     # Binary mask CLI tool
│   ├── model/                           # Original SAM3 models (unchanged)
│   ├── examples/                        # Official examples
│   └── ... (other original SAM3 files)
└── Developer_Guide.md                   # This file
```

---

## Module Responsibilities

### `config.py` - Configuration Layer

**Purpose**: Define all user-configurable options with validation

**Key Classes**:
- `PropagationDirection` - Enum for forward/backward/both
- `OutputMode` - Enum for overlay/segmented/binary
- `MemoryStrategy` - Enum for reset/continuous/decay
- `ConceptSegmentationConfig` - Main segmentation parameters
- `MaskProcessorConfig` - Rendering parameters
- `BinaryMaskConfig` - Binary conversion parameters

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

**Architecture Notes**:
- ✅ User-level orchestration (we control WHEN to call SAM3)
- ✅ SAM3 unchanged (we don't modify propagation algorithm)
- ✅ Clear separation of concerns
- ✅ Extensible for future enhancements

**Key Methods**:

```python
class ConceptSegmentationStrategy:
    def load_video(video_path)           # Stage 1: Load
    def plan_chunks()                    # Stage 2: Plan
    def process_chunks(predictor)        # Stage 3: Process (main inference)
    def get_outputs_for_frame(idx)       # Stage 4: Query results
    def get_all_outputs()                # Stage 4: Get all results
    def get_summary()                    # Reporting
    def dry_run()                        # Validation
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
- `save_video()` - Write output video
- `save_mask_frames()` - Write mask sequence

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
7. Save → `MaskProcessor.save_video()`

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

**Flow**:
1. Parse arguments
2. Validate input paths
3. Load mask(s)
4. Apply conversions → `BinaryMaskConverter`
5. Save result(s)

**Design**: Supports both single-file and batch processing

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

