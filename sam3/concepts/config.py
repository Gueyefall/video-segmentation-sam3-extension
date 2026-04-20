"""
Configuration dataclasses for SAM3 concept-based video segmentation.

This module defines structured configurations for:
- Concept segmentation strategy
- Mask processing and rendering
- Binary mask conversion
- Video I/O operations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
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


@dataclass
class ConceptSegmentationConfig:
    """
    Configuration for concept-based video segmentation strategy.
    
    Attributes:
        concepts: List of semantic concepts to segment (e.g., "traffic sign", "speed limit").
        chunk_size: Number of frames per processing chunk (default: 130).
        overlap: Number of frames overlapping between consecutive chunks (default: 26).
        propagation_direction: Direction to propagate prompts (forward/backward/both).
        memory_strategy: How to manage memory across chunks.
        apply_temporal_disambiguation: Whether to apply temporal disambiguation in SAM3.
        has_presence_token: Whether SAM3 uses presence tokens.
        geo_encoder_use_img_cross_attn: Whether geometry encoder uses image cross-attention.
        strict_state_dict_loading: Whether to enforce strict state dict matching.
        async_loading_frames: Whether to load frames asynchronously.
        video_loader_type: Type of video loader ("cv2" or other).
        output_prob_thresh: Confidence threshold for mask outputs (0-1).
        compile: Whether to compile the model for performance.
    """
    concepts: List[str] = field(default_factory=lambda: ["traffic sign"])
    chunk_size: int = 130
    overlap: int = 26
    propagation_direction: PropagationDirection = PropagationDirection.FORWARD
    memory_strategy: MemoryStrategy = MemoryStrategy.RESET_PER_CHUNK
    apply_temporal_disambiguation: bool = True
    has_presence_token: bool = True
    geo_encoder_use_img_cross_attn: bool = True
    strict_state_dict_loading: bool = True
    async_loading_frames: bool = False
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
        if not self.concepts or len(self.concepts) == 0:
            raise ValueError("At least one concept must be specified")
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
        threshold: Confidence threshold for binarization (0-1, default: 0.5).
        invert: Whether to invert the binary mask (True = inverted, False = normal).
        dilate_kernel_size: Optional morphological dilation (None or int > 0).
        erode_kernel_size: Optional morphological erosion (None or int > 0).
    """
    threshold: float = 0.5
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
