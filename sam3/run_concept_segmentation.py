#!/usr/bin/env python
"""
Main orchestrator for SAM3 concept-based video segmentation.

Entry point for end-to-end workflow:
1. Load video and parse concepts
2. Plan chunk-based processing
3. Run segmentation with memory bank management
4. Render and save outputs in desired format

Supports:
- Multiple output formats (overlay, segmented-only, binary masks)
- Flexible chunk and memory strategies
- Progress tracking and detailed logging
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from concepts.config import (
        ConceptSegmentationConfig,
        MaskProcessorConfig,
        OutputMode,
        PropagationDirection,
        MemoryStrategy,
        ExemplarPlacement,
    )
    from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy
    from concepts.mask_processor import MaskProcessor
    from concepts.utils import make_color_palette
except ImportError:
    # Support direct execution when the script directory is not on PYTHONPATH.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from concepts.config import (
        ConceptSegmentationConfig,
        MaskProcessorConfig,
        OutputMode,
        PropagationDirection,
        MemoryStrategy,
        ExemplarPlacement,
    )
    from concepts.sam3_concepts_segmenter import ConceptSegmentationStrategy
    from concepts.mask_processor import MaskProcessor
    from concepts.utils import make_color_palette


def fail(msg: str, code: int = 1):
    """Print error message and exit."""
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_concepts(concept_text: str) -> List[str]:
    """
    Parse dot-separated concept list.
    
    Args:
        concept_text: Text like "traffic sign. speed limit. stop sign."
    
    Returns:
        List of individual concept strings.
    
    Raises:
        SystemExit: If no valid concepts found.
    """
    parts = [c.strip() for c in concept_text.split(".")]
    concepts = [c for c in parts if c]
    if not concepts:
        fail("No valid concepts found in --concepts.")
    return concepts


def parse_exemplar_box(box_text: str) -> Tuple[float, float, float, float]:
    """Parse an absolute-pixel exemplar box in x,y,w,h format."""
    parts = [part.strip() for part in box_text.split(",")]
    if len(parts) != 4:
        fail("--exemplar-box must contain exactly four comma-separated values: x,y,w,h")

    try:
        x, y, width, height = (float(part) for part in parts)
    except ValueError as exc:
        fail(f"Invalid --exemplar-box value '{box_text}': {exc}")

    return x, y, width, height


def build_argument_parser():
    """Build and return CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_concept_segmentation",
        description=(
            "Run SAM3 video segmentation with text prompts and optional external image exemplar injection.\n"
            "Supports multiple output modes and flexible memory/chunk strategies."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic overlay mode with default chunk settings:
  python run_concept_segmentation.py \\
    --video input.mp4 \\
    --concepts "traffic sign. speed limit" \\
    --output output.mp4

  # Binary masks only with custom threshold:
  python run_concept_segmentation.py \\
    --video input.mp4 \\
    --concepts "speed limit" \\
    --output masks/ \\
    --output-mode binary_masks_only \\
    --threshold 0.6

  # Segmented-only with custom chunk strategy:
  python run_concept_segmentation.py \\
    --video input.mp4 \\
    --concepts "traffic sign" \\
    --output output.mp4 \\
    --output-mode segmented_only \\
    --chunk-size 200 \\
    --overlap 40

    # External exemplar pseudo-frame injection with text+visual fusion:
    python run_concept_segmentation.py \
        --video input.mp4 \
        --concepts "speed limit 30 sign" \
        --exemplar-image exemplar_30.jpg \
        --exemplar-bbox "145,98,62,64" \
        --exemplar-placement letterbox \
        --output output.mp4
        """,
    )
    
    # Input/output
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video (mp4, avi, etc.)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output video (mp4) or directory (for binary masks)",
    )
    parser.add_argument(
        "--concepts",
        required=True,
        help="Dot-separated semantic concepts (e.g., 'traffic sign. speed limit sign.')",
    )

    # Optional external exemplar pseudo-frame injection
    parser.add_argument(
        "--exemplar-image",
        help="Path to external exemplar image used as pseudo-frame at each chunk start",
    )
    parser.add_argument(
        "--exemplar-bbox",
        help="Exemplar bbox in source image absolute pixel x,y,w,h format",
    )
    parser.add_argument(
        "--exemplar-placement",
        type=str,
        default="letterbox",
        choices=["letterbox", "canvas"],
        help="How exemplar image is fitted to video size before remapping bbox (default: letterbox)",
    )
    parser.add_argument(
        "--exemplar-box-label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Exemplar box label: 1 for positive (include region), 0 for negative (exclude region) (default: 1)",
    )
    parser.add_argument(
        "--debug-exemplar-preview",
        help="Optional output image path for fitted exemplar preview with remapped bbox",
    )
    
    # Output rendering
    parser.add_argument(
        "--output-mode",
        type=str,
        default="overlay",
        choices=["overlay", "segmented_only", "binary_masks_only"],
        help="Output rendering mode (default: overlay)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Mask overlay transparency in overlay mode (0-1, default: 0.45)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for mask outputs (0-1, default: 0.5)",
    )
    
    # Chunk strategy
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=130,
        help="Frames per processing chunk (default: 130)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=26,
        help="Overlap between chunks in frames (default: 26)",
    )
    
    # Memory strategy
    parser.add_argument(
        "--memory-strategy",
        type=str,
        default="reset_per_chunk",
        choices=["reset_per_chunk", "continuous", "temporal_decay"],
        help="Memory bank management strategy (default: reset_per_chunk)",
    )
    
    # Propagation
    parser.add_argument(
        "--propagation-direction",
        type=str,
        default="forward",
        choices=["forward", "backward", "both"],
        help="Temporal propagation direction (default: forward)",
    )
    
    # Model options
    parser.add_argument(
        "--no-temporal-disambiguation",
        action="store_true",
        help="Disable temporal disambiguation in SAM3",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep loaded video frames on CPU in the SAM3 inference state (safer on constrained/MIG GPUs).",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable Tensor Float 32 (slower but more stable on some MIG GPUs).",
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=None,
        help="Limit maximum number of tracked objects (default: model default ~10000). Lower values reduce memory pressure.",
    )
    parser.add_argument(
        "--no-inter-chunk-cuda-drain",
        action="store_true",
        help="Disable explicit CUDA drain between chunks (enabled by default).",
    )
    parser.add_argument(
        "--inter-chunk-sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep after chunk cleanup before starting next chunk (default: 0.0).",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch compile optimization (can be unstable on some MIG setups)",
    )
    compile_group.add_argument(
        "--no-compile",
        action="store_true",
        help="Deprecated alias kept for compatibility; compile is already disabled by default.",
    )
    
    # Utility
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running inference",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


def main():
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # Validate PyTorch
    if torch is None:
        fail("PyTorch is not installed. Install with: pip install torch")
    
    # Parse and validate paths
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    binary_video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    binary_output_is_video = (
        args.output_mode == "binary_masks_only"
        and output_path.suffix.lower() in binary_video_exts
    )
    binary_output_dir = output_path
    
    if not video_path.exists():
        fail(f"Input video does not exist: {video_path}")
    
    # Create output directory if needed
    if args.output_mode == "binary_masks_only":
        # Binary mask mode supports either video output or PNG frame directory output.
        if binary_output_is_video:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        elif output_path.suffix:
            binary_output_dir = output_path.with_suffix("")
            print(
                f"[WARN] --output '{output_path}' looks like a file path; "
                f"using directory '{binary_output_dir}' for binary mask frames.",
                file=sys.stderr,
            )
        binary_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse concepts
    concepts = parse_concepts(args.concepts)
    exemplar_bbox = parse_exemplar_box(args.exemplar_bbox) if args.exemplar_bbox else None
    prompt_labels = concepts

    if bool(args.exemplar_image) != bool(args.exemplar_bbox):
        fail("--exemplar-image and --exemplar-bbox must be provided together")

    if args.verbose:
        print(f"[INFO] Concepts: {concepts}", file=sys.stderr)
        if args.exemplar_image:
            print(
                f"[INFO] Exemplar image: {args.exemplar_image}, bbox={exemplar_bbox}, placement={args.exemplar_placement}",
                file=sys.stderr,
            )
            if args.debug_exemplar_preview:
                print(
                    f"[INFO] Exemplar preview path: {args.debug_exemplar_preview}",
                    file=sys.stderr,
                )
    
    # Build configurations
    try:
        compile_enabled = bool(args.compile)
        segmentation_config = ConceptSegmentationConfig(
            concepts=concepts,
            exemplar_image_path=args.exemplar_image,
            exemplar_image_bbox=exemplar_bbox,
            exemplar_placement=ExemplarPlacement(args.exemplar_placement),
            exemplar_box_label=args.exemplar_box_label,
            debug_exemplar_preview_path=args.debug_exemplar_preview,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            propagation_direction=PropagationDirection(args.propagation_direction),
            memory_strategy=MemoryStrategy(args.memory_strategy),
            apply_temporal_disambiguation=not args.no_temporal_disambiguation,
            offload_video_to_cpu=args.offload_video_to_cpu,
            disable_tf32=args.disable_tf32,
            max_num_objects=args.max_num_objects,
            inter_chunk_cuda_drain=not args.no_inter_chunk_cuda_drain,
            inter_chunk_sleep_sec=args.inter_chunk_sleep_sec,
            output_prob_thresh=args.threshold,
            compile=compile_enabled,
        )
        
        mask_processor_config = MaskProcessorConfig(
            output_mode=OutputMode(args.output_mode),
            overlay_alpha=args.overlay_alpha,
        )
    except ValueError as e:
        fail(f"Configuration error: {e}")
    
    # Validate configs
    try:
        segmentation_config.validate()
        mask_processor_config.validate()
    except ValueError as e:
        fail(f"Validation error: {e}")
    
    # Load strategy and video
    strategy = ConceptSegmentationStrategy(segmentation_config)
    
    try:
        if args.verbose:
            print(f"[INFO] Loading video: {video_path}", file=sys.stderr)
        strategy.load_video(str(video_path))
        strategy.plan_chunks()
    except Exception as e:
        fail(f"Failed to load video: {e}")
    
    if args.verbose:
        summary = strategy.get_summary()
        print(f"[INFO] Video metadata: {summary['video']}", file=sys.stderr)
        print(f"[INFO] Chunk plan: {summary['chunks']}", file=sys.stderr)
    
    # Dry-run mode
    if args.dry_run:
        if args.verbose:
            print("[INFO] Running dry-run validation...", file=sys.stderr)
        if strategy.dry_run():
            print("[OK] Dry-run passed.", file=sys.stderr)
            return
        else:
            fail("Dry-run validation failed")
    
    # Import predictor (GPU required for actual inference)
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name()}", file=sys.stderr)
    else:
        print("[WARN] CUDA not available; using CPU (slow)", file=sys.stderr)
    
    try:
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    except ImportError as e:
        fail(
            f"Could not import Sam3VideoPredictor. Ensure sam3 is installed "
            f"('pip install -e .') and on PYTHONPATH. Original error: {e}"
        )
    
    # Initialize predictor
    if args.verbose:
        print("[INFO] Initializing Sam3VideoPredictor...", file=sys.stderr)
    
    try:
        predictor = Sam3VideoPredictor(
            checkpoint_path=None,
            bpe_path=None,
            has_presence_token=segmentation_config.has_presence_token,
            geo_encoder_use_img_cross_attn=segmentation_config.geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=segmentation_config.strict_state_dict_loading,
            async_loading_frames=segmentation_config.async_loading_frames,
            video_loader_type=segmentation_config.video_loader_type,
            apply_temporal_disambiguation=segmentation_config.apply_temporal_disambiguation,
            compile=segmentation_config.compile,
        )
    except Exception as e:
        fail(f"Failed to initialize predictor: {e}")
    
    # Process video
    print(f"[INFO] Starting segmentation of {strategy.video_metadata['num_frames']} frames...", file=sys.stderr)
    
    def progress_callback(chunk_idx, total_chunks, message):
        print(f"[INFO] {message}", file=sys.stderr)
    
    try:
        strategy.process_chunks(predictor, progress_callback=progress_callback)
    except Exception as e:
        fail(f"Segmentation failed: {e}")
    
    # Render output
    print("[INFO] Rendering output...", file=sys.stderr)
    
    processor = MaskProcessor(mask_processor_config)
    processor.set_palette(len(prompt_labels))
    
    output_frames = []
    all_outputs = strategy.get_all_outputs()
    
    for frame_idx, frame in enumerate(strategy.video_frames):
        outputs = all_outputs.get(frame_idx, {})
        processed_frame, labels = processor.process_frame(
            frame,
            outputs,
            concept_names=prompt_labels,
        )
        output_frames.append(processed_frame)
    
    # Save output
    if args.output_mode == "binary_masks_only":
        if binary_output_is_video:
            print(f"[INFO] Saving binary mask video to: {output_path}", file=sys.stderr)
            fps = strategy.video_metadata["fps"]
            codec = "mp4v"
            try:
                processor.save_video(
                    output_frames,
                    str(output_path),
                    fps,
                    codec=codec,
                )
            except Exception as e:
                fail(f"Failed to save binary mask video: {e}")
        else:
            print(f"[INFO] Saving binary masks to: {binary_output_dir}", file=sys.stderr)
            processor.save_mask_frames(
                {i: outputs.get("out_binary_masks", np.array([])) for i, outputs in all_outputs.items()},
                str(binary_output_dir),
                prefix="concept_mask"
            )
    else:
        print(f"[INFO] Saving output video to: {output_path}", file=sys.stderr)
        fps = strategy.video_metadata["fps"]
        codec = "mp4v"  # Standard MP4 codec
        
        try:
            processor.save_video(
                output_frames,
                str(output_path),
                fps,
                codec=codec,
            )
        except Exception as e:
            fail(f"Failed to save output video: {e}")
    
    # Summary
    final_summary = strategy.get_summary()
    print(f"[INFO] Segmentation complete. Processed {final_summary['frames_with_output']} frames.", file=sys.stderr)
    if args.output_mode == "binary_masks_only":
        if binary_output_is_video:
            print(f"[SUCCESS] Output saved to: {output_path}", file=sys.stderr)
        else:
            print(f"[SUCCESS] Output saved to: {binary_output_dir}", file=sys.stderr)
    else:
        print(f"[SUCCESS] Output saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
