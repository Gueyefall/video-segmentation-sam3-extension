"""
Composite a segmented mask video onto a base video using a binary selector video.

Foreground rule:
- White pixels in the binary selector keep pixels from the mask video.
- Black pixels in the binary selector keep pixels from the base video.

Example:
  python scripts/overlay_with_binary_selector.py \
    --base-video cosmos_output.mp4 \
    --mask-video input_traffic_signs_masks.mp4 \
    --binary-video input_traffic_signs_binary.mp4 \
    --binary-inverted-video input_traffic_signs_binary_inverted.mp4 \
    --output-video composited_output.mp4
"""

import argparse
import os

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay mask video on top of base video using a binary per-pixel selector."
    )
    parser.add_argument(
        "--base-video",
        required=True,
        help="Path to base video (e.g., cosmos_output.mp4).",
    )
    parser.add_argument(
        "--mask-video",
        required=True,
        help="Path to segmented mask video (e.g., input_traffic_signs_masks.mp4).",
    )
    parser.add_argument(
        "--binary-video",
        required=True,
        help="Path to binary selector video where white=foreground, black=background.",
    )
    parser.add_argument(
        "--binary-inverted-video",
        default=None,
        help=(
            "Optional path to inverted binary selector video. "
            "If provided, it is used only for optional consistency checking."
        ),
    )
    parser.add_argument(
        "--output-video",
        required=True,
        help="Path to output composited video.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Threshold in [0, 255] used to interpret binary selector frames (default: 127).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on processed frames. Use -1 for all frames.",
    )
    parser.add_argument(
        "--strict-size-check",
        action="store_true",
        help="Fail if input frame sizes differ instead of resizing selector/mask frames.",
    )
    return parser.parse_args()


def open_video(path: str, label: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {label} video: {path}")
    return cap


def frame_count(cap: cv2.VideoCapture) -> int:
    value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return value if value > 0 else -1


def maybe_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    args = parse_args()

    if not (0 <= args.threshold <= 255):
        raise ValueError("--threshold must be between 0 and 255")

    base_cap = open_video(args.base_video, "base")
    mask_cap = open_video(args.mask_video, "mask")
    binary_cap = open_video(args.binary_video, "binary selector")
    inv_cap = None

    if args.binary_inverted_video is not None:
        inv_cap = open_video(args.binary_inverted_video, "inverted binary selector")

    base_w = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    base_h = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = base_cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    if args.strict_size_check:
        for label, cap in [
            ("mask", mask_cap),
            ("binary", binary_cap),
            ("binary_inverted", inv_cap),
        ]:
            if cap is None:
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (base_w, base_h):
                raise RuntimeError(
                    f"Size mismatch for {label}: got {(w, h)}, expected {(base_w, base_h)}"
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (base_w, base_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output_video}")

    counts = {
        "base": frame_count(base_cap),
        "mask": frame_count(mask_cap),
        "binary": frame_count(binary_cap),
        "binary_inverted": frame_count(inv_cap) if inv_cap is not None else -1,
    }
    known_counts = [v for v in counts.values() if v >= 0]
    estimated_total = min(known_counts) if known_counts else -1

    frame_idx = 0
    while True:
        if args.max_frames >= 0 and frame_idx >= args.max_frames:
            break

        ok_base, base_frame = base_cap.read()
        ok_mask, mask_frame = mask_cap.read()
        ok_binary, binary_frame = binary_cap.read()
        if not (ok_base and ok_mask and ok_binary):
            break

        if inv_cap is not None:
            ok_inv, inv_frame = inv_cap.read()
            if not ok_inv:
                break
        else:
            inv_frame = None

        if not args.strict_size_check:
            mask_frame = maybe_resize(mask_frame, base_w, base_h)
            binary_frame = maybe_resize(binary_frame, base_w, base_h)
            if inv_frame is not None:
                inv_frame = maybe_resize(inv_frame, base_w, base_h)

        binary_gray = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        foreground = binary_gray > args.threshold

        # Optional sanity check: inverted selector should be the complement.
        if inv_frame is not None:
            inv_gray = cv2.cvtColor(inv_frame, cv2.COLOR_BGR2GRAY)
            inv_foreground = inv_gray > args.threshold
            mismatch = np.mean((~foreground) != inv_foreground)
            if mismatch > 0.10 and frame_idx == 0:
                print(
                    "Warning: inverted binary selector does not look like a complement "
                    f"(mismatch ratio={mismatch:.3f})."
                )

        composed = base_frame.copy()
        composed[foreground] = mask_frame[foreground]

        writer.write(composed)
        frame_idx += 1

        if frame_idx % 50 == 0:
            if estimated_total > 0:
                print(f"Processed {frame_idx}/{estimated_total} frames...")
            else:
                print(f"Processed {frame_idx} frames...")

    base_cap.release()
    mask_cap.release()
    binary_cap.release()
    if inv_cap is not None:
        inv_cap.release()
    writer.release()

    print(f"Done. Wrote {frame_idx} frames to: {args.output_video}")


if __name__ == "__main__":
    main()
