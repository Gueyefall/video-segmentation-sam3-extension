#!/usr/bin/env python
"""
Standalone CLI tool for binary mask conversion and manipulation.

Provides operations on mask files:
- Convert probability masks to binary masks
- Invert binary masks
- Apply morphological operations
- Compute mask statistics
- Batch convert multiple masks
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
    from concepts.utils import normalize_mask
except ImportError:
    # Support direct execution when the script directory is not on PYTHONPATH.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from concepts.config import BinaryMaskConfig
    from concepts.binary_mask_converter import BinaryMaskConverter
    from concepts.utils import normalize_mask


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
    --threshold 0.5

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

  # Batch convert with prefix:
  python run_binary_mask_converter.py \\
    --batch-dir ./masks \\
    --output-dir ./masks_binary \\
    --threshold 0.6 \\
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
    
    # Conversion options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for binarization (0-1, default: 0.5)",
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
        help="Compute and print mask statistics only",
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
            print(f"[INFO] Converted to binary (threshold={converter.config.threshold}, invert={converter.config.invert})", file=sys.stderr)
        
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
    
    # Validate arguments
    if not args.input and not args.batch_dir:
        fail("Either --input or --batch-dir must be specified")
    
    if args.input and args.batch_dir:
        fail("Cannot specify both --input and --batch-dir")
    
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
                print(f"[INFO] Processing {i+1}/{len(input_files)}: {input_file.name}", file=sys.stderr)
            
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


if __name__ == "__main__":
    main()
