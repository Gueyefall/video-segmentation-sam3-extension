"""
Binary mask conversion and manipulation utilities.

Provides functionality for:
- Converting probability masks to binary masks
- Inverting masks
- Morphological operations (dilation, erosion)
- Mask composition and merging
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2

from .config import BinaryMaskConfig
from .utils import normalize_mask, resize_mask_to_frame


class BinaryMaskConverter:
    """
    Convert and manipulate masks to binary format.
    
    Attributes:
        config: BinaryMaskConfig with conversion options.
    """
    
    def __init__(self, config: BinaryMaskConfig):
        """
        Initialize converter.
        
        Args:
            config: BinaryMaskConfig with threshold and operation options.
        """
        config.validate()
        self.config = config
    
    def convert_to_binary(
        self,
        mask: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert probability mask to binary mask.
        
        Args:
            mask: Input mask (float or bool).
            threshold: Confidence threshold (uses config default if None).
        
        Returns:
            Binary mask (uint8: 0 or 255).
        """
        threshold = threshold or self.config.threshold
        
        mask = normalize_mask(mask).astype(np.float32)
        binary = (mask > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations
        if self.config.dilate_kernel_size is not None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.dilate_kernel_size, self.config.dilate_kernel_size)
            )
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        if self.config.erode_kernel_size is not None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.erode_kernel_size, self.config.erode_kernel_size)
            )
            binary = cv2.erode(binary, kernel, iterations=1)
        
        # Invert if requested
        if self.config.invert:
            binary = 255 - binary
        
        return binary
    
    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Invert a binary mask.
        
        Args:
            mask: Input binary mask.
        
        Returns:
            Inverted mask (same format as input).
        """
        if mask.dtype == np.uint8:
            return 255 - mask
        elif mask.dtype == bool or np.issubdtype(mask.dtype, np.floating):
            return ~(mask.astype(bool))
        else:
            return np.logical_not(mask).astype(mask.dtype)
    
    def apply_morphological_ops(
        self,
        mask: np.ndarray,
        operation: str = "close",
        kernel_size: int = 5,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations to clean up masks.
        
        Args:
            mask: Input binary mask.
            operation: "open", "close", "dilate", or "erode".
            kernel_size: Size of morphological kernel.
            iterations: Number of iterations.
        
        Returns:
            Processed mask.
        """
        mask_uint8 = (normalize_mask(mask).astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        if operation == "open":
            result = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == "dilate":
            result = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        elif operation == "erode":
            result = cv2.erode(mask_uint8, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return result
    
    def merge_masks(
        self,
        masks: list,
        method: str = "union"
    ) -> np.ndarray:
        """
        Merge multiple binary masks.
        
        Args:
            masks: List of mask arrays.
            method: "union" (OR), "intersection" (AND), or "average".
        
        Returns:
            Merged mask (uint8).
        """
        if not masks:
            raise ValueError("No masks to merge")
        
        # Normalize all masks
        normalized = [normalize_mask(m).astype(np.uint8) for m in masks]
        
        if method == "union":
            result = np.zeros_like(normalized[0])
            for mask in normalized:
                result = np.maximum(result, mask)
            return result * 255
        
        elif method == "intersection":
            result = np.ones_like(normalized[0])
            for mask in normalized:
                result = np.minimum(result, mask)
            return result * 255
        
        elif method == "average":
            result = np.mean(normalized, axis=0)
            return (result * 255).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown merge method: {method}")
    
    def get_mask_statistics(self, mask: np.ndarray) -> dict:
        """
        Compute statistics for a binary mask.
        
        Args:
            mask: Input mask.
        
        Returns:
            Dict with area, coverage percentage, and contour info.
        """
        mask_uint8 = normalize_mask(mask).astype(np.uint8) * 255
        
        total_pixels = mask_uint8.size
        foreground_pixels = np.sum(mask_uint8 > 0)
        coverage = (foreground_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Find contours for additional info
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return {
            "total_pixels": total_pixels,
            "foreground_pixels": foreground_pixels,
            "coverage_percent": coverage,
            "num_contours": len(contours),
            "largest_contour_area": max(
                [cv2.contourArea(c) for c in contours],
                default=0
            ),
        }
    
    def extract_connected_components(
        self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Extract connected components from a binary mask.
        
        Args:
            mask: Input binary mask.
        
        Returns:
            Tuple of (labeled_image, num_components).
        """
        mask_uint8 = normalize_mask(mask).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        return labels, num_labels
    
    def save_binary_mask(
        self,
        mask: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save a binary mask as PNG image.
        
        Args:
            mask: Binary mask to save.
            output_path: Path for output PNG file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask_uint8 = normalize_mask(mask).astype(np.uint8) * 255
        cv2.imwrite(str(output_path), mask_uint8)
    
    def load_binary_mask(self, input_path: str) -> np.ndarray:
        """
        Load a binary mask from PNG image.
        
        Args:
            input_path: Path to mask PNG file.
        
        Returns:
            Binary mask (uint8).
        """
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot load mask from: {input_path}")
        return mask
    
    def batch_convert_masks(
        self,
        mask_dict: dict,
        threshold: Optional[float] = None
    ) -> dict:
        """
        Convert multiple masks in batch.
        
        Args:
            mask_dict: Dict mapping frame_id to mask arrays.
            threshold: Conversion threshold (uses config default if None).
        
        Returns:
            Dict of same structure with converted masks.
        """
        return {
            frame_id: self.convert_to_binary(mask, threshold)
            for frame_id, mask in mask_dict.items()
        }
