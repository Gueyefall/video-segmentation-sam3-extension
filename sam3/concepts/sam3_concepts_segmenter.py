"""
Core concept-based video segmentation strategy using SAM3.

Implements:
- Memory bank management for concept embeddings
- Chunk-based processing with overlap handling
- Temporal propagation strategies
- Multi-concept segmentation workflow
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .config import ConceptSegmentationConfig, MemoryStrategy, PropagationDirection
from .utils import load_video_frames, get_video_metadata


class MemoryBank:
    """
    Manages concept embeddings and temporal memory across chunks.
    
    Attributes:
        concepts: List of concept names.
        strategy: Memory management strategy.
    """
    
    def __init__(self, concepts: List[str], strategy: MemoryStrategy = MemoryStrategy.RESET_PER_CHUNK):
        """
        Initialize memory bank.
        
        Args:
            concepts: List of semantic concepts to track.
            strategy: How to manage memory across chunks.
        """
        self.concepts = concepts
        self.strategy = strategy
        self.embeddings: Dict[str, list] = {c: [] for c in concepts}
        self.frame_history: List[int] = []

        # ARCHITECTURAL NOTE: This MemoryBank is a USER-LEVEL ORCHESTRATION layer.
        # It tracks metadata about embeddings but does NOT feed into SAM3's model.
        # SAM3's actual temporal memory is handled by its transformer attention.
        # This class is useful for: statistics, monitoring, future enhancements.
    
    def add_embedding(self, concept: str, embedding: np.ndarray, frame_idx: int) -> None:
        """
        Add concept embedding from a frame.
        
        Args:
            concept: Concept name.
            embedding: Embedding vector or feature.
            frame_idx: Frame index.
        """
        if concept not in self.embeddings:
            self.embeddings[concept] = []
        
        self.embeddings[concept].append({
            "data": embedding,
            "frame_idx": frame_idx,
        })
        self.frame_history.append(frame_idx)
    
    def reset(self) -> None:
        """Reset memory bank (for RESET_PER_CHUNK strategy)."""
        self.embeddings = {c: [] for c in self.concepts}
        self.frame_history = []
    
    def apply_temporal_decay(self, current_frame: int, decay_factor: float = 0.9) -> None:
        """
        Apply temporal decay to older embeddings (for TEMPORAL_DECAY strategy).
        
        Args:
            current_frame: Current frame index.
            decay_factor: Weight factor for temporal distance.
        """
        for concept in self.embeddings:
            for entry in self.embeddings[concept]:
                distance = current_frame - entry["frame_idx"]
                entry["weight"] = decay_factor ** (distance / 10.0)
    
    def get_summary(self) -> Dict:
        """Get memory bank statistics."""
        return {
            "num_concepts": len(self.concepts),
            "embeddings_per_concept": {
                c: len(embs) for c, embs in self.embeddings.items()
            },
            "total_frames_tracked": len(set(self.frame_history)),
        }


class ChunkPlanner:
    """
    Plan and manage chunked processing of video.
    
    Purpose:
        Enables memory-efficient processing of arbitrarily long videos.
        Original SAM3 usually runs a single session over the full video, while
        this planner splits processing into overlapping chunks.

    Trade-off:
        Chunk boundaries may lose some temporal context; overlap mitigates this.

    Attributes:
        num_frames: Total number of frames in video.
        chunk_size: Frames per chunk.
        overlap: Overlap between chunks.
    """
    
    def __init__(self, num_frames: int, chunk_size: int, overlap: int):
        """
        Initialize chunk planner.
        
        Args:
            num_frames: Total video frames.
            chunk_size: Frames per chunk.
            overlap: Overlap between chunks.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        
        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = self._build_chunks()
    
    def _build_chunks(self) -> List[Tuple[int, int]]:
        """Build list of (start_frame, end_frame) tuples."""
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap
        
        while start < self.num_frames:
            end = min(start + self.chunk_size, self.num_frames)
            chunks.append((start, end))
            if end >= self.num_frames:
                break
            start += step
        
        return chunks
    
    def get_chunks(self) -> List[Tuple[int, int]]:
        """Get all chunks."""
        return self.chunks
    
    def get_summary(self) -> Dict:
        """Get chunk planning statistics."""
        return {
            "total_frames": self.num_frames,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "num_chunks": len(self.chunks),
            "step_size": self.chunk_size - self.overlap,
        }


class ConceptSegmentationStrategy:
    """
    High-level strategy for concept-based video segmentation.
    
    Architectural role:
        This class is an orchestration layer. It controls when and how SAM3's
        predictor API is called, while SAM3 itself still performs temporal
        matching, tracking, and mask propagation.

    Example flow:
        1. load_video() parses metadata and frames.
        2. plan_chunks() organizes work into chunk windows.
        3. process_chunks() iterates chunks and runs session lifecycle.
        4. get_all_outputs() returns per-frame aggregated results.

    Manages:
    - Video loading and metadata
    - Memory bank lifecycle
    - Chunk planning and processing
    - Predictor interaction
    - Per-frame output aggregation
    
    Attributes:
        config: ConceptSegmentationConfig
        memory_bank: MemoryBank for tracking embeddings
        chunk_planner: ChunkPlanner for frame organization
    """
    
    def __init__(self, config: ConceptSegmentationConfig):
        """
        Initialize segmentation strategy.
        
        Args:
            config: ConceptSegmentationConfig with all parameters.
        """
        config.validate()
        self.config = config
        self.memory_bank = MemoryBank(config.concepts, config.memory_strategy)
        self.chunk_planner: Optional[ChunkPlanner] = None
        self.video_frames: Optional[List[np.ndarray]] = None
        self.video_metadata: Dict = {}
        self.per_frame_outputs: Dict[int, dict] = {}
    
    def load_video(self, video_path: str) -> None:
        """
        Load video frames and extract metadata.
        
        Args:
            video_path: Path to input video file.
        
        Raises:
            RuntimeError: If video cannot be loaded.
        """
        try:
            self.video_frames, fps, width, height = load_video_frames(video_path)
            self.video_metadata = {
                "fps": fps,
                "width": width,
                "height": height,
                "num_frames": len(self.video_frames),
                "path": video_path,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load video: {e}")
    
    def plan_chunks(self) -> None:
        """
        Plan chunk processing based on loaded video.
        
        Raises:
            RuntimeError: If video not loaded.
        """
        if self.video_frames is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")
        
        self.chunk_planner = ChunkPlanner(
            num_frames=len(self.video_frames),
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )
    
    def process_chunks(
        self,
        predictor,
        progress_callback=None
    ) -> None:
        """
        Process video in chunks using predictor.
        
        Args:
            predictor: Sam3VideoPredictor instance.
            progress_callback: Optional callback(chunk_idx, total_chunks, message).
        
        Raises:
            RuntimeError: If setup incomplete.
        """
        if self.chunk_planner is None:
            raise RuntimeError("Chunks not planned. Call plan_chunks() first.")
        
        chunks = self.chunk_planner.get_chunks()
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            if progress_callback:
                msg = f"Processing chunk {chunk_idx + 1}/{len(chunks)}: frames [{chunk_start}, {chunk_end - 1}]"
                progress_callback(chunk_idx, len(chunks), msg)
            
            self._process_chunk(predictor, chunk_start, chunk_end)
            
            # Memory management between chunks
            if self.config.memory_strategy == MemoryStrategy.RESET_PER_CHUNK:
                self.memory_bank.reset()
            elif self.config.memory_strategy == MemoryStrategy.TEMPORAL_DECAY:
                self.memory_bank.apply_temporal_decay(chunk_end)
            
            # Clear GPU cache if available
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _process_chunk(
        self,
        predictor,
        chunk_start: int,
        chunk_end: int
    ) -> None:
        """
        Process a single chunk with predictor.
        
        Args:
            predictor: Sam3VideoPredictor instance.
            chunk_start: Start frame index.
            chunk_end: End frame index.
        """
        if self.video_frames is None:
            raise RuntimeError("Video not loaded")
        
        max_frames = chunk_end - chunk_start
        video_path = self.video_metadata.get("path", "")
        
        # Start session
        start_resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(video_path),
        })
        session_id = start_resp["session_id"]
        
        try:
            # Add prompts for all concepts
            for concept in self.config.concepts:
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(chunk_start),
                    "text": concept,
                })
            
            # Propagate in video
            propagation_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": self.config.propagation_direction.value,
                "start_frame_index": int(chunk_start),
                "max_frame_num_to_track": int(max_frames),
            }
            
            if self.config.output_prob_thresh != 0.5:
                propagation_request["output_prob_thresh"] = self.config.output_prob_thresh
            
            # Collect outputs
            for resp in predictor.handle_stream_request(propagation_request):
                frame_idx = int(resp["frame_index"])
                outputs = resp.get("outputs", {})
                self.per_frame_outputs[frame_idx] = outputs
        
        finally:
            # Close session
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    
    def get_outputs_for_frame(self, frame_idx: int) -> Dict:
        """
        Get segmentation outputs for a specific frame.
        
        Args:
            frame_idx: Frame index.
        
        Returns:
            Output dict with segmentation masks and IDs.
        """
        return self.per_frame_outputs.get(frame_idx, {})
    
    def get_all_outputs(self) -> Dict[int, dict]:
        """Get all per-frame outputs."""
        return self.per_frame_outputs
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of segmentation run.
        
        Returns:
            Dict with video, chunk, memory, and output statistics.
        """
        return {
            "video": self.video_metadata,
            "chunks": self.chunk_planner.get_summary() if self.chunk_planner else None,
            "memory": self.memory_bank.get_summary(),
            "frames_with_output": len(self.per_frame_outputs),
            "total_frames": len(self.video_frames) if self.video_frames else 0,
        }
    
    def dry_run(self) -> bool:
        """
        Perform a dry-run validation without GPU inference.
        
        Tests:
        - Video loading and format validation
        - Configuration consistency
        - Chunk planning
        
        Returns:
            True if all checks pass.
        """
        try:
            if self.video_frames is None:
                raise RuntimeError("Video not loaded for dry-run")
            
            if self.chunk_planner is None:
                raise RuntimeError("Chunks not planned for dry-run")
            
            # Validate outputs dictionary structure
            test_output = self.get_outputs_for_frame(0)
            if test_output and not all(
                k in test_output for k in ["out_obj_ids", "out_binary_masks"]
            ):
                print("[WARN] Output structure may be incomplete", file=sys.stderr)
            
            return True
        except Exception as e:
            print(f"[ERROR] Dry-run failed: {e}", file=sys.stderr)
            return False
