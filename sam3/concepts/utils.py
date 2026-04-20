"""
Utility functions for SAM3 concept-based video segmentation.

Provides helpers for:
- Color palette generation
- Mask resizing and format conversion
- Video frame loading and metadata extraction
- Mask visualization and composition
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


def make_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate a color palette with distinct colors in BGR format.
    
    Args:
        num_colors: Number of colors to generate.
    
    Returns:
        List of (B, G, R) tuples.
    """
    base_palette = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Red-Orange
        (128, 255, 0),    # Yellow-Green
    ]
    return [base_palette[i % len(base_palette)] for i in range(num_colors)]


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a mask to boolean format with proper dimensions.
    
    Args:
        mask: Input mask (may be float, torch tensor, or numpy array).
    
    Returns:
        Boolean numpy array with shape (H, W).
    """
    # Convert from torch tensor if needed
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    
    # Ensure numpy array
    mask = np.asarray(mask)
    
    # Squeeze extra dimensions
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # Convert to boolean
    mask = mask.astype(bool)
    return mask


def resize_mask_to_frame(
    mask: np.ndarray,
    target_height: int,
    target_width: int,
    method: str = "nearest"
) -> np.ndarray:
    """
    Resize a mask to match frame dimensions.
    
    Args:
        mask: Boolean mask array.
        target_height: Target frame height.
        target_width: Target frame width.
        method: Interpolation method ("nearest" or "bilinear").
    
    Returns:
        Resized mask with shape (target_height, target_width).
    """
    if mask.shape[:2] == (target_height, target_width):
        return mask
    
    interp = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=interp
    )
    return resized.astype(bool)


def overlay_mask_on_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay a colored mask on a video frame with transparency.
    
    Args:
        frame: Input frame (BGR numpy array, shape: H x W x 3).
        mask: Boolean mask (shape: H x W or H x W x 1).
        color: Color in BGR format (B, G, R).
        alpha: Transparency factor (0-1, where 1 is fully opaque).
    
    Returns:
        Frame with overlaid mask.
    """
    frame = frame.copy()
    mask = normalize_mask(mask)
    
    if mask.shape[:2] != frame.shape[:2]:
        mask = resize_mask_to_frame(mask, frame.shape[0], frame.shape[1])
    
    color_arr = np.array(color, dtype=np.float32)
    frame[mask] = (
        alpha * color_arr +
        (1.0 - alpha) * frame[mask].astype(np.float32)
    ).astype(np.uint8)
    
    return frame


def extract_segmented_region(
    frame: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Extract only the masked regions from a frame (rest is black).
    
    Args:
        frame: Input frame (BGR numpy array).
        mask: Boolean mask.
    
    Returns:
        Frame with only masked regions visible.
    """
    output = np.zeros_like(frame)
    mask = normalize_mask(mask)
    
    if mask.shape[:2] != frame.shape[:2]:
        mask = resize_mask_to_frame(mask, frame.shape[0], frame.shape[1])
    
    output[mask] = frame[mask]
    return output


def compose_multiple_masks(
    frame: np.ndarray,
    masks: np.ndarray
) -> np.ndarray:
    """
    Combine multiple segmented regions into one frame.
    
    Args:
        frame: Input frame (for reference shape).
        masks: Array of extracted mask regions (shape: N x H x W x 3).
    
    Returns:
        Composite frame with max-pooled regions.
    """
    output = np.zeros_like(frame)
    for mask_frame in masks:
        output = np.maximum(output, mask_frame)
    return output


def draw_label_on_frame(
    frame: np.ndarray,
    text: str,
    color: Tuple[int, int, int],
    x: int = 12,
    y: int = 28,
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a text label on a frame with outline for visibility.
    
    Args:
        frame: Input frame (modified in-place).
        text: Text to display.
        color: Color in BGR format.
        x: X position of label.
        y: Y position of label.
        font_scale: Font size scale.
        thickness: Text thickness.
    
    Returns:
        Frame with drawn label.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw black outline
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, (0, 0, 0),
        thickness + 2, cv2.LINE_AA
    )
    # Draw colored text
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, color,
        thickness, cv2.LINE_AA
    )
    return frame


def get_video_metadata(video_path: str) -> Tuple[float, int, int]:
    """
    Extract video metadata without loading all frames.
    
    Args:
        video_path: Path to video file.
    
    Returns:
        Tuple of (fps, width, height).
    
    Raises:
        RuntimeError: If video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return float(fps), width, height, frame_count


def load_video_frames(video_path: str) -> Tuple[List[np.ndarray], float, int, int]:
    """
    Load all frames from a video file.
    
    Args:
        video_path: Path to video file.
    
    Returns:
        Tuple of (frames list, fps, width, height).
    
    Raises:
        RuntimeError: If video cannot be opened or has no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(frame_bgr)
    cap.release()
    
    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    
    return frames, float(fps), width, height
