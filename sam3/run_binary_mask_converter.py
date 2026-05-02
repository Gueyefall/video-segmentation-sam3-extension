#!/usr/bin/env python
"""
Standalone CLI tool for binary mask conversion and manipulation.

Provides operations on mask files:
- Convert probability masks to binary masks
- Invert binary masks
- Apply morphological operations
- Compute mask statistics
- Batch convert multiple masks
- Process mask videos frame-by-frame
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from concepts.config import BinaryMaskConfig
    from concepts.binary_mask_converter import BinaryMaskConverter
except ImportError:
    # Support direct execution when the script directory is not on PYTHONPATH.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from concepts.config import BinaryMaskConfig
    from concepts.binary_mask_converter import BinaryMaskConverter


def fail(msg: str, code: int = 1):
    """Print error message and exit."""
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def build_argument_parser():
    """Build and return CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_binary_mask_converter",
        description=(
            "Convert and manipulate binary masks.\n"
            "Supports thresholding, inversion, morphological operations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert probability mask to binary with threshold:
  python run_binary_mask_converter.py \\
    --input mask_prob.png \\
    --output mask_binary.png \\
    --threshold 0.08

  # Invert a binary mask:
  python run_binary_mask_converter.py \\
    --input mask.png \\
    --output mask_inverted.png \\
    --invert

  # Apply dilation to close small holes:
  python run_binary_mask_converter.py \\
    --input mask.png \\
    --output mask_dilated.png \\
    --dilate-kernel 5

  # Get mask statistics:
  python run_binary_mask_converter.py \\
    --input mask.png \\
    --stats

  # Batch convert with inversion:
  python run_binary_mask_converter.py \\
    --batch-dir ./masks \\
    --output-dir ./masks_binary \\
    --threshold 0.6 \\
    --invert

  # Convert a mask video to binary output video:
  python run_binary_mask_converter.py \\
    --input-video ./masks.mp4 \\
    --output-video ./masks_binary.mp4 \\
    --invert
        """,
    )

    # Input/output (single file)
    parser.add_argument(
        "--input",
        help="Input mask file (PNG)",
    )
    parser.add_argument(
        "--output",
        help="Output mask file (PNG)",
    )

    # Batch processing
    parser.add_argument(
        "--batch-dir",
        help="Input directory for batch processing",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch results",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.png",
        help="File pattern for batch processing (default: *.png)",
    )

    # Video processing
    parser.add_argument(
        "--input-video",
        help="Input mask video file",
    )
    parser.add_argument(
        "--output-video",
        help="Output processed mask video file",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="FourCC codec for video output (default: mp4v)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output FPS (default: input FPS)",
    )

    # Conversion options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="Confidence threshold for binarization (0-1, default: 0.08)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the binary mask (background becomes foreground)",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=None,
        help="Dilation kernel size (optional, default: None)",
    )
    parser.add_argument(
        "--erode-kernel",
        type=int,
        default=None,
        help="Erosion kernel size (optional, default: None)",
    )

    # Operations
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute and print mask statistics only (single image mode)",
    )
    parser.add_argument(
        "--morph-op",
        choices=["open", "close", "dilate", "erode"],
        help="Apply morphological operation",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Morphological operation kernel size (default: 5)",
    )

    # Utility
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def process_single_mask(
    input_path: str,
    output_path: Optional[str],
    converter: BinaryMaskConverter,
    morph_op: Optional[str] = None,
    morph_kernel: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Process a single mask file.

    Args:
        input_path: Path to input mask.
        output_path: Path to output mask (None for no save).
        converter: BinaryMaskConverter instance.
        morph_op: Optional morphological operation.
        morph_kernel: Kernel size for morphological ops.
        verbose: Enable logging.

    Returns:
        Dict with processing info.
    """
    try:
        mask = converter.load_binary_mask(input_path)
        if verbose:
            print(f"[INFO] Loaded mask from: {input_path}", file=sys.stderr)

        # Apply morphological operation if requested
        if morph_op:
            mask = converter.apply_morphological_ops(
                mask, operation=morph_op, kernel_size=morph_kernel
            )
            if verbose:
                print(f"[INFO] Applied '{morph_op}' morphology", file=sys.stderr)

        # Convert to binary
        binary_mask = converter.convert_to_binary(mask)
        if verbose:
            print(
                "[INFO] Converted to binary "
                f"(threshold={converter.config.threshold}, invert={converter.config.invert})",
                file=sys.stderr,
            )

        # Save if output path provided
        if output_path:
            output_p = Path(output_path)
            output_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_p), binary_mask)
            if verbose:
                print(f"[INFO] Saved to: {output_path}", file=sys.stderr)

        # Get statistics
        stats = converter.get_mask_statistics(binary_mask)

        return {
            "status": "success",
            "input": input_path,
            "output": output_path,
            "statistics": stats,
        }

    except Exception as e:
        return {
            "status": "failed",
            "input": input_path,
            "error": str(e),
        }


def process_video_mask(
    input_video_path: str,
    output_video_path: str,
    converter: BinaryMaskConverter,
    morph_op: Optional[str] = None,
    morph_kernel: int = 5,
    verbose: bool = False,
    output_fps: Optional[float] = None,
    codec: str = "mp4v",
) -> dict:
    """Process an input mask video and write a processed binary mask video."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {
            "status": "failed",
            "input": input_video_path,
            "error": f"Cannot open input video: {input_video_path}",
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_to_use = output_fps if output_fps is not None else fps
    if not fps_to_use or fps_to_use <= 0:
        fps_to_use = 30.0

    output_p = Path(output_video_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if len(codec) != 4:
        cap.release()
        return {
            "status": "failed",
            "input": input_video_path,
            "error": f"Codec must be 4 characters (got '{codec}')",
        }

    writer = cv2.VideoWriter(
        str(output_p),
        cv2.VideoWriter_fourcc(*codec),
        float(fps_to_use),
        (width, height),
        True,
    )

    if not writer.isOpened():
        cap.release()
        return {
            "status": "failed",
            "input": input_video_path,
            "error": f"Cannot open output video writer: {output_video_path}",
        }

    processed_frames = 0
    coverage_values = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Convert to single-channel mask if video frame is color.
            gray_mask = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )

            if morph_op:
                gray_mask = converter.apply_morphological_ops(
                    gray_mask,
                    operation=morph_op,
                    kernel_size=morph_kernel,
                )

            binary_mask = converter.convert_to_binary(gray_mask)
            mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            writer.write(mask_bgr)

            frame_stats = converter.get_mask_statistics(binary_mask)
            coverage_values.append(frame_stats["coverage_percent"])
            processed_frames += 1

            if verbose and (processed_frames % 50 == 0):
                print(
                    f"[INFO] Processed {processed_frames} frames",
                    file=sys.stderr,
                )

        mean_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0

        return {
            "status": "success",
            "input": input_video_path,
            "output": output_video_path,
            "frames": processed_frames,
            "input_frames": frame_count,
            "input_fps": fps,
            "output_fps": fps_to_use,
            "resolution": f"{width}x{height}",
            "mean_coverage_percent": mean_coverage,
        }
    except Exception as e:
        return {
            "status": "failed",
            "input": input_video_path,
            "error": str(e),
        }
    finally:
        cap.release()
        writer.release()


def print_statistics(stats: dict) -> None:
    """Pretty-print mask statistics."""
    print("\nMask Statistics:")
    print(f"  Total pixels: {stats['total_pixels']}")
    print(f"  Foreground pixels: {stats['foreground_pixels']}")
    print(f"  Coverage: {stats['coverage_percent']:.2f}%")
    print(f"  Number of contours: {stats['num_contours']}")
    print(f"  Largest contour area: {stats['largest_contour_area']}")


def main():
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    input_modes = [
        bool(args.input),
        bool(args.batch_dir),
        bool(args.input_video),
    ]

    if sum(input_modes) == 0:
        fail("One input mode is required: --input, --batch-dir, or --input-video")
    if sum(input_modes) > 1:
        fail("Specify only one input mode: --input, --batch-dir, or --input-video")

    if args.video_fps is not None and args.video_fps <= 0:
        fail("--video-fps must be > 0")

    # Build converter
    try:
        converter_config = BinaryMaskConfig(
            threshold=args.threshold,
            invert=args.invert,
            dilate_kernel_size=args.dilate_kernel,
            erode_kernel_size=args.erode_kernel,
        )
        converter_config.validate()
    except ValueError as e:
        fail(f"Configuration error: {e}")

    converter = BinaryMaskConverter(converter_config)

    # Single file processing
    if args.input:
        input_path = Path(args.input).expanduser().resolve()

        if not input_path.exists():
            fail(f"Input file does not exist: {input_path}")

        # Statistics only
        if args.stats:
            try:
                mask = converter.load_binary_mask(str(input_path))
                stats = converter.get_mask_statistics(mask)
                print(f"[INFO] Mask file: {input_path}")
                print_statistics(stats)
            except Exception as e:
                fail(f"Failed to compute statistics: {e}")
            return

        # Normal processing
        if not args.output:
            fail("--output is required when processing a single file (unless using --stats)")

        output_path = Path(args.output).expanduser().resolve()

        result = process_single_mask(
            str(input_path),
            str(output_path),
            converter,
            morph_op=args.morph_op,
            morph_kernel=args.morph_kernel,
            verbose=args.verbose,
        )

        if result["status"] == "success":
            if args.verbose:
                print_statistics(result["statistics"])
            print(f"[SUCCESS] Processed: {input_path} -> {output_path}", file=sys.stderr)
        else:
            fail(f"Processing failed: {result.get('error', 'Unknown error')}")

    # Batch processing
    elif args.batch_dir:
        if args.stats:
            fail("--stats is only supported in single image mode (--input)")

        batch_dir = Path(args.batch_dir).expanduser().resolve()

        if not batch_dir.exists():
            fail(f"Batch input directory does not exist: {batch_dir}")

        if not args.output_dir:
            fail("--output-dir is required for batch processing")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find input files
        input_files = sorted(batch_dir.glob(args.file_pattern))

        if not input_files:
            fail(f"No files matching pattern '{args.file_pattern}' found in {batch_dir}")

        print(f"[INFO] Found {len(input_files)} files to process", file=sys.stderr)

        # Process batch
        results = []
        for i, input_file in enumerate(input_files):
            if args.verbose:
                print(
                    f"[INFO] Processing {i + 1}/{len(input_files)}: {input_file.name}",
                    file=sys.stderr,
                )

            output_file = output_dir / input_file.name

            result = process_single_mask(
                str(input_file),
                str(output_file),
                converter,
                morph_op=args.morph_op,
                morph_kernel=args.morph_kernel,
                verbose=args.verbose,
            )
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"\n[SUCCESS] Processed {successful}/{len(results)} files", file=sys.stderr)

        if successful < len(results):
            print(f"[WARN] {len(results) - successful} files failed", file=sys.stderr)

    # Video processing
    elif args.input_video:
        if args.stats:
            fail("--stats is only supported in single image mode (--input)")

        input_video_path = Path(args.input_video).expanduser().resolve()
        if not input_video_path.exists():
            fail(f"Input video does not exist: {input_video_path}")

        if not args.output_video:
            fail("--output-video is required when using --input-video")

        output_video_path = Path(args.output_video).expanduser().resolve()

        result = process_video_mask(
            str(input_video_path),
            str(output_video_path),
            converter,
            morph_op=args.morph_op,
            morph_kernel=args.morph_kernel,
            verbose=args.verbose,
            output_fps=args.video_fps,
            codec=args.video_codec,
        )

        if result["status"] != "success":
            fail(f"Video processing failed: {result.get('error', 'Unknown error')}")

        print(
            f"[SUCCESS] Processed video: {input_video_path} -> {output_video_path}",
            file=sys.stderr,
        )
        print(
            f"[INFO] Frames: {result['frames']} "
            f"(input reported: {result['input_frames']}) | "
            f"Resolution: {result['resolution']} | "
            f"FPS in/out: {result['input_fps']:.3f}/{result['output_fps']:.3f} | "
            f"Mean coverage: {result['mean_coverage_percent']:.2f}%",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
