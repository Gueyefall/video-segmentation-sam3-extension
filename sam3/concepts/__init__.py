"""
SAM3 Concept-Based Video Segmentation Module

This package provides tools for concept-based video segmentation using SAM3,
with support for memory-aware chunk processing, flexible mask rendering, and
binary mask extraction.

Key Components:
- ConceptSegmentationStrategy: Core segmentation with memory bank management
- MaskProcessor: Flexible mask rendering and output formats
- BinaryMaskConverter: Binary mask computation and inversion
"""

from .config import (
    ConceptSegmentationConfig,
    MaskProcessorConfig,
    BinaryMaskConfig,
)
from .sam3_concepts_segmenter import ConceptSegmentationStrategy
from .mask_processor import MaskProcessor
from .binary_mask_converter import BinaryMaskConverter

__all__ = [
    "ConceptSegmentationConfig",
    "MaskProcessorConfig",
    "BinaryMaskConfig",
    "ConceptSegmentationStrategy",
    "MaskProcessor",
    "BinaryMaskConverter",
]
