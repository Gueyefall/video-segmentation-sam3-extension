"""
Configuration dataclasses for SAM3 concept-based video segmentation.

This module defines structured configurations for:
- Concept segmentation strategy
- Mask processing and rendering
- Binary mask conversion
- Video I/O operations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple
from enum import Enum


class PropagationDirection(str, Enum):
    """Supported propagation directions in video."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class OutputMode(str, Enum):
    """Output rendering modes."""
    OVERLAY = "overlay"
    SEGMENTED_ONLY = "segmented_only"
    BINARY_MASKS_ONLY = "binary_masks_only"


class MemoryStrategy(str, Enum):
    """Memory bank management strategies."""
    RESET_PER_CHUNK = "reset_per_chunk"
    CONTINUOUS = "continuous"
    TEMPORAL_DECAY = "temporal_decay"


class ExemplarPlacement(str, Enum):
    """How an external exemplar image is fitted to the video frame size."""
    LETTERBOX = "letterbox"
    CANVAS = "canvas"


@dataclass
class ConceptSegmentationConfig:
    """
    Configuration for concept-based video segmentation strategy.
    
    Attributes:
        concepts: List of semantic concepts to segment (e.g., "traffic sign", "speed limit").
        exemplar_image_path: Optional path to an external exemplar image.
        exemplar_image_bbox: Optional exemplar bbox in source-image absolute pixel xywh format.
        exemplar_placement: How to fit exemplar image to target video resolution.
        exemplar_box_label: Label for exemplar box: 1 for positive (include), 0 for negative (exclude).
        debug_exemplar_preview_path: Optional image path to save fitted exemplar and remapped bbox.
        chunk_size: Number of frames per processing chunk (default: 130).
        overlap: Number of frames overlapping between consecutive chunks (default: 26).
        propagation_direction: Direction to propagate prompts (forward/backward/both).
        memory_strategy: How to manage memory across chunks.
        apply_temporal_disambiguation: Whether to apply temporal disambiguation in SAM3.
        has_presence_token: Whether SAM3 uses presence tokens.
        geo_encoder_use_img_cross_attn: Whether geometry encoder uses image cross-attention.
        strict_state_dict_loading: Whether to enforce strict state dict matching.
        async_loading_frames: Whether to load frames asynchronously.
        offload_video_to_cpu: Whether to keep loaded video frames on CPU in inference state.
        disable_tf32: Disable Tensor Float 32 for stability on constrained GPUs (slower).
        max_num_objects: Limit maximum number of tracked objects to reduce memory pressure.
        inter_chunk_cuda_drain: Run explicit CUDA drain between chunks.
        inter_chunk_sleep_sec: Optional wait time after drain before next chunk.
        video_loader_type: Type of video loader ("cv2" or other).
        output_prob_thresh: Confidence threshold for mask outputs (0-1).
        compile: Whether to compile the model for performance.
    """
    concepts: List[str] = field(default_factory=lambda: ["traffic sign"])
    exemplar_image_path: Optional[str] = None
    exemplar_image_bbox: Optional[Tuple[float, float, float, float]] = None
    exemplar_placement: ExemplarPlacement = ExemplarPlacement.LETTERBOX
    exemplar_box_label: int = 1  # 1 for positive (foreground), 0 for negative (background)
    debug_exemplar_preview_path: Optional[str] = None
    chunk_size: int = 130
    overlap: int = 26
    propagation_direction: PropagationDirection = PropagationDirection.FORWARD
    memory_strategy: MemoryStrategy = MemoryStrategy.RESET_PER_CHUNK
    apply_temporal_disambiguation: bool = True
    has_presence_token: bool = True
    geo_encoder_use_img_cross_attn: bool = True
    strict_state_dict_loading: bool = True
    async_loading_frames: bool = False
    offload_video_to_cpu: bool = False
    disable_tf32: bool = False  # Force TF32 off for more stable but slower computation on some MIG setups.
    max_num_objects: Optional[int] = None  # Limit max tracked objects; None uses model default (~10k).
    inter_chunk_cuda_drain: bool = True
    inter_chunk_sleep_sec: float = 0.0
    video_loader_type: str = "cv2"
    output_prob_thresh: float = 0.5
    compile: bool = False

    def validate(self):
        """Validate configuration consistency."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if not 0.0 <= self.output_prob_thresh <= 1.0:
            raise ValueError(
                f"output_prob_thresh must be in [0, 1], got {self.output_prob_thresh}"
            )
        if self.exemplar_box_label not in (0, 1):
            raise ValueError(
                f"exemplar_box_label must be 0 (negative) or 1 (positive), got {self.exemplar_box_label}"
            )
        if self.max_num_objects is not None and self.max_num_objects <= 0:
            raise ValueError(
                f"max_num_objects must be > 0 or None, got {self.max_num_objects}"
            )
        if self.inter_chunk_sleep_sec < 0:
            raise ValueError(
                f"inter_chunk_sleep_sec must be >= 0, got {self.inter_chunk_sleep_sec}"
            )
        if (not self.concepts or len(self.concepts) == 0) and self.exemplar_image_path is None:
            raise ValueError("At least one concept or an exemplar image must be specified")

        has_exemplar_path = bool(self.exemplar_image_path)
        has_exemplar_bbox = self.exemplar_image_bbox is not None
        if has_exemplar_path != has_exemplar_bbox:
            raise ValueError(
                "exemplar_image_path and exemplar_image_bbox must be provided together"
            )

        if self.exemplar_image_bbox is not None:
            if len(self.exemplar_image_bbox) != 4:
                raise ValueError("exemplar_image_bbox must contain exactly four values: x,y,w,h")
            x, y, width, height = self.exemplar_image_bbox
            if x < 0 or y < 0:
                raise ValueError("exemplar_image_bbox x and y must be >= 0")
            if width <= 0 or height <= 0:
                raise ValueError("exemplar_image_bbox width and height must be > 0")
        return True


@dataclass
class MaskProcessorConfig:
    """
    Configuration for mask processing and rendering.
    
    Attributes:
        output_mode: How to render masks (overlay/segmented_only/binary_masks_only).
        overlay_alpha: Transparency for overlay mode (0-1, default: 0.45).
        color_palette: Custom color palette (BGR tuples) or None for auto-generated.
        save_format: Output format ("mp4" for video, "png" for image sequence).
        max_labels_to_draw: Maximum number of concept labels to draw on frame.
        label_font_scale: Font scale for rendered labels.
    """
    output_mode: OutputMode = OutputMode.OVERLAY
    overlay_alpha: float = 0.45
    color_palette: Optional[List[tuple]] = None
    save_format: str = "mp4"
    max_labels_to_draw: int = 8
    label_font_scale: float = 0.7

    def validate(self):
        """Validate configuration consistency."""
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(
                f"overlay_alpha must be in [0, 1], got {self.overlay_alpha}"
            )
        if self.max_labels_to_draw <= 0:
            raise ValueError(
                f"max_labels_to_draw must be > 0, got {self.max_labels_to_draw}"
            )
        if self.label_font_scale <= 0:
            raise ValueError(
                f"label_font_scale must be > 0, got {self.label_font_scale}"
            )
        return True


@dataclass
class BinaryMaskConfig:
    """
    Configuration for binary mask conversion and extraction.
    
    Attributes:
        threshold: Confidence threshold for binarization (0-1, default: 0.08).
        invert: Whether to invert the binary mask (True = inverted, False = normal).
        dilate_kernel_size: Optional morphological dilation (None or int > 0).
        erode_kernel_size: Optional morphological erosion (None or int > 0).
    """
    threshold: float = 0.08
    invert: bool = False
    dilate_kernel_size: Optional[int] = None
    erode_kernel_size: Optional[int] = None

    def validate(self):
        """Validate configuration consistency."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.dilate_kernel_size is not None and self.dilate_kernel_size <= 0:
            raise ValueError(
                f"dilate_kernel_size must be > 0 or None, got {self.dilate_kernel_size}"
            )
        if self.erode_kernel_size is not None and self.erode_kernel_size <= 0:
            raise ValueError(
                f"erode_kernel_size must be > 0 or None, got {self.erode_kernel_size}"
            )
        return True


@dataclass
class VideoIOConfig:
    """
    Configuration for video input/output operations.
    
    Attributes:
        video_codec: Video codec to use ("mp4v" for MP4).
        quality: Quality/CRF parameter for encoding (lower = better, default: 23).
        target_fps: Target FPS for output (None = same as input).
    """
    video_codec: str = "mp4v"
    quality: int = 23
    target_fps: Optional[float] = None

    def validate(self):
        """Validate configuration consistency."""
        if self.quality < 0 or self.quality > 51:
            raise ValueError(f"quality must be in [0, 51], got {self.quality}")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0 or None, got {self.target_fps}")
        return True
