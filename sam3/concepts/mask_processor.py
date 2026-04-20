"""
Mask processing and rendering for concept-based segmentation outputs.

Provides flexible mask rendering in multiple formats:
- Overlay mode: masks alpha-blended onto original video
- Segmented-only mode: isolated masked regions
- Binary masks: thresholded or raw masks for further processing
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from .config import MaskProcessorConfig, OutputMode
from .utils import (
    make_color_palette,
    normalize_mask,
    resize_mask_to_frame,
    overlay_mask_on_frame,
    extract_segmented_region,
    compose_multiple_masks,
)


class MaskProcessor:
    """
    Process and render masks with flexible output modes.
    
    Supports three output modes:
    1. OVERLAY: Alpha-blended masks on original frames
    2. SEGMENTED_ONLY: Isolated masked regions on black background
    3. BINARY_MASKS_ONLY: Binary masks saved as separate frames
    
    Attributes:
        config: MaskProcessorConfig instance
        palette: Color palette for mask visualization
    """
    
    def __init__(self, config: MaskProcessorConfig):
        """
        Initialize mask processor.
        
        Args:
            config: MaskProcessorConfig with rendering options.
        """
        config.validate()
        self.config = config
        self.palette = config.color_palette or []
    
    def set_palette(self, num_colors: int) -> None:
        """
        Generate and set a color palette.
        
        Args:
            num_colors: Number of colors needed for concepts.
        """
        self.palette = make_color_palette(num_colors)
    
    def process_frame(
        self,
        frame: np.ndarray,
        outputs: Dict,
        concept_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process a single frame with segmentation outputs.
        
        Args:
            frame: Input frame (BGR numpy array).
            outputs: Segmentation outputs dict with keys:
                - out_obj_ids: Object identifiers
                - out_binary_masks: Segmentation masks
            concept_names: Names of concepts (for labeling).
        
        Returns:
            Tuple of (processed_frame, labels_drawn).
        """
        if not self.palette:
            self._estimate_palette_from_outputs(outputs)
        
        # Handle empty outputs
        if not outputs or not isinstance(outputs, dict):
            return self._empty_frame(frame), []
        
        if "out_obj_ids" not in outputs or "out_binary_masks" not in outputs:
            return self._empty_frame(frame), []
        
        out_obj_ids = outputs["out_obj_ids"]
        out_binary_masks = outputs["out_binary_masks"]
        
        # Handle empty object list
        if len(out_obj_ids) == 0:
            return self._empty_frame(frame), []
        
        # Process based on output mode
        if self.config.output_mode == OutputMode.OVERLAY:
            return self._process_overlay(
                frame, out_obj_ids, out_binary_masks, concept_names
            )
        elif self.config.output_mode == OutputMode.SEGMENTED_ONLY:
            return self._process_segmented_only(
                frame, out_obj_ids, out_binary_masks, concept_names
            )
        elif self.config.output_mode == OutputMode.BINARY_MASKS_ONLY:
            return self._process_binary_masks(
                frame, out_obj_ids, out_binary_masks, concept_names
            )
        else:
            raise ValueError(f"Unknown output mode: {self.config.output_mode}")
    
    def _process_overlay(
        self,
        frame: np.ndarray,
        obj_ids: np.ndarray,
        masks: np.ndarray,
        concept_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Render masks as overlay on original frame."""
        vis = frame.copy()
        labels = []
        single_concept = concept_names[0] if concept_names and len(concept_names) == 1 else None
        
        for i, obj_id in enumerate(obj_ids):
            mask = normalize_mask(masks[i])
            
            # Select color
            if single_concept is not None:
                color = self.palette[i % len(self.palette)]
            else:
                color = self.palette[(int(obj_id) - 1) % len(self.palette)]
            
            # Overlay mask
            vis = overlay_mask_on_frame(
                vis, mask,
                color=color,
                alpha=self.config.overlay_alpha
            )
            
            # Prepare label
            label = single_concept if single_concept else f"obj_{obj_id}"
            labels.append(label)

        return vis, labels
    
    def _process_segmented_only(
        self,
        frame: np.ndarray,
        obj_ids: np.ndarray,
        masks: np.ndarray,
        concept_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Render only masked regions (black background elsewhere)."""
        vis = np.zeros_like(frame)
        draw_labels = []
        single_concept = concept_names[0] if concept_names and len(concept_names) == 1 else None
        
        extracted_regions = []
        for i, obj_id in enumerate(obj_ids):
            mask = normalize_mask(masks[i])
            isolated = extract_segmented_region(frame, mask)
            extracted_regions.append(isolated)
            
            # Prepare label
            label = single_concept if single_concept else f"obj_{obj_id}"
            draw_labels.append(label)
        
        # Composite all regions with max-pooling
        if extracted_regions:
            vis = compose_multiple_masks(frame, np.array(extracted_regions))
        
        return vis, draw_labels
    
    def _process_binary_masks(
        self,
        frame: np.ndarray,
        obj_ids: np.ndarray,
        masks: np.ndarray,
        concept_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Render binary masks (stacked or composite).
        
        Returns:
            For visualization, returns a grayscale composite.
        """
        # Stack binary masks
        num_masks = len(obj_ids)
        binary_stack = np.zeros(
            (frame.shape[0], frame.shape[1], num_masks),
            dtype=np.uint8
        )
        
        for i, obj_id in enumerate(obj_ids):
            mask = normalize_mask(masks[i])
            if mask.shape[:2] != frame.shape[:2]:
                mask = resize_mask_to_frame(mask, frame.shape[0], frame.shape[1])
            binary_stack[:, :, i] = mask.astype(np.uint8) * 255
        
        # Return max-pooled for visualization
        vis = (np.max(binary_stack, axis=2) > 0).astype(np.uint8) * 255
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        labels = [
            (concept_names[0] if concept_names else f"obj_{obj_id}")
            for obj_id in obj_ids
        ]
        
        return vis_bgr, labels
    
    def _empty_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return empty frame based on output mode."""
        if self.config.output_mode == OutputMode.SEGMENTED_ONLY:
            return np.zeros_like(frame)
        elif self.config.output_mode == OutputMode.BINARY_MASKS_ONLY:
            return np.zeros_like(frame)
        else:
            return frame.copy()
    
    def _estimate_palette_from_outputs(self, outputs: Dict) -> None:
        """Estimate palette size from outputs."""
        if outputs and "out_obj_ids" in outputs:
            num_objects = len(outputs["out_obj_ids"])
            self.set_palette(max(1, num_objects))
        else:
            self.set_palette(1)
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float,
        codec: str = "mp4v"
    ) -> None:
        """
        Save processed frames as video.
        
        Args:
            frames: List of processed frame arrays.
            output_path: Path to output video file.
            fps: Frames per second.
            codec: Video codec (default: "mp4v" for MP4).
        """
        if not frames:
            raise ValueError("Cannot save video: no frames provided")
        
        height, width = frames[0].shape[:2]
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height)
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for: {output_path}")
        
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
    
    def save_mask_frames(
        self,
        masks_per_frame: Dict[int, np.ndarray],
        output_dir: str,
        prefix: str = "mask"
    ) -> None:
        """
        Save binary masks as image sequence.
        
        Args:
            masks_per_frame: Dict mapping frame index to mask array.
            output_dir: Output directory for mask images.
            prefix: Prefix for mask filenames.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for frame_idx, mask in masks_per_frame.items():
            mask_normalized = normalize_mask(mask)
            mask_uint8 = (mask_normalized.astype(np.uint8) * 255)
            
            filename = output_path / f"{prefix}_{frame_idx:06d}.png"
            cv2.imwrite(str(filename), mask_uint8)
