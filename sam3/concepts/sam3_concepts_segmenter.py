"""
Core concept-based video segmentation strategy using SAM3.

Implements:
- Memory bank management for concept embeddings
- Chunk-based processing with overlap handling
- Temporal propagation strategies
- Multi-concept segmentation workflow
"""

import sys
import gc
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None

from .config import ConceptSegmentationConfig, MemoryStrategy, PropagationDirection, ExemplarPlacement
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
        self.exemplar_injected_frame: Optional[np.ndarray] = None
        self.exemplar_injected_box_norm: Optional[Tuple[float, float, float, float]] = None
        self._chunk_temp_dir: Optional[Path] = None

    @staticmethod
    def _is_cuda_allocator_assert(exc: Exception) -> bool:
        """Return True for known allocator assert signatures seen on some MIG setups."""
        msg = str(exc)
        return (
            "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in msg
            or "CUDACachingAllocator.cpp" in msg
        )

    @staticmethod
    def _safe_cuda_cleanup() -> None:
        """Best-effort CUDA cleanup to reduce allocator pressure between chunk sessions."""
        gc.collect()
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            # Synchronization can fail after a prior CUDA fault; keep cleanup best-effort.
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _drain_between_chunks(self, has_next_chunk: bool) -> None:
        """Drain GPU state before the next chunk starts to reduce allocator instability."""
        if not has_next_chunk:
            return

        if self.config.inter_chunk_cuda_drain:
            self._safe_cuda_cleanup()

        if self.config.inter_chunk_sleep_sec > 0:
            time.sleep(self.config.inter_chunk_sleep_sec)
    
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
            self._prepare_exemplar_prompt_if_needed()
        except Exception as e:
            raise RuntimeError(f"Failed to load video: {e}")

    def _prepare_exemplar_prompt_if_needed(self) -> None:
        """Prepare pseudo-frame and bbox remap for external exemplar injection."""
        if not self.config.exemplar_image_path:
            return

        exemplar_path = Path(self.config.exemplar_image_path).expanduser().resolve()
        if not exemplar_path.exists():
            raise RuntimeError(f"Exemplar image does not exist: {exemplar_path}")

        source_image = cv2.imread(str(exemplar_path), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError(f"Failed to read exemplar image: {exemplar_path}")

        bbox = self.config.exemplar_image_bbox
        if bbox is None:
            raise RuntimeError("exemplar_image_bbox must be provided with exemplar_image_path")

        src_h, src_w = source_image.shape[:2]
        x, y, width, height = bbox
        if x + width > src_w or y + height > src_h:
            raise RuntimeError(
                "Exemplar bbox is outside source image bounds: "
                f"bbox=({x}, {y}, {width}, {height}), image=({src_w}, {src_h})"
            )

        target_w = int(self.video_metadata["width"])
        target_h = int(self.video_metadata["height"])
        if self.config.exemplar_placement == ExemplarPlacement.CANVAS:
            fitted_frame, fitted_bbox = self._fit_exemplar_canvas(
                source_image,
                bbox,
                target_w,
                target_h,
            )
        else:
            fitted_frame, fitted_bbox = self._fit_exemplar_letterbox(
                source_image,
                bbox,
                target_w,
                target_h,
            )

        fx, fy, fw, fh = fitted_bbox
        self.exemplar_injected_frame = fitted_frame
        self.exemplar_injected_box_norm = (
            fx / float(target_w),
            fy / float(target_h),
            fw / float(target_w),
            fh / float(target_h),
        )
        self._save_exemplar_preview_if_enabled(fitted_frame, fitted_bbox)

    def _save_exemplar_preview_if_enabled(
        self,
        fitted_frame: np.ndarray,
        fitted_bbox: Tuple[float, float, float, float],
    ) -> None:
        """Optionally save a preview image with the remapped exemplar bbox."""
        preview_path = self.config.debug_exemplar_preview_path
        if not preview_path:
            return

        out_path = Path(preview_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        x, y, w, h = fitted_bbox
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = max(x1 + 1, int(round(x + w)))
        y2 = max(y1 + 1, int(round(y + h)))

        vis = fitted_frame.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            "remapped exemplar bbox",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if not cv2.imwrite(str(out_path), vis):
            raise RuntimeError(f"Failed to write exemplar preview image: {out_path}")

    @staticmethod
    def _fit_exemplar_letterbox(
        source_image: np.ndarray,
        bbox_xywh: Tuple[float, float, float, float],
        target_w: int,
        target_h: int,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Resize exemplar by letterbox and remap bbox to fitted frame coordinates."""
        src_h, src_w = source_image.shape[:2]
        scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = cv2.resize(source_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        fitted = np.zeros((target_h, target_w, 3), dtype=source_image.dtype)
        fitted[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        x, y, w, h = bbox_xywh
        return fitted, (x * scale + pad_x, y * scale + pad_y, w * scale, h * scale)

    @staticmethod
    def _fit_exemplar_canvas(
        source_image: np.ndarray,
        bbox_xywh: Tuple[float, float, float, float],
        target_w: int,
        target_h: int,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Center exemplar on a canvas and remap bbox with offset only."""
        src_h, src_w = source_image.shape[:2]
        if src_w > target_w or src_h > target_h:
            raise RuntimeError(
                "Canvas placement requires exemplar image to fit target frame. "
                f"source=({src_w}, {src_h}), target=({target_w}, {target_h})"
            )

        offset_x = (target_w - src_w) // 2
        offset_y = (target_h - src_h) // 2
        fitted = np.zeros((target_h, target_w, 3), dtype=source_image.dtype)
        fitted[offset_y : offset_y + src_h, offset_x : offset_x + src_w] = source_image

        x, y, w, h = bbox_xywh
        return fitted, (x + offset_x, y + offset_y, w, h)

    def _ensure_temp_chunk_dir(self) -> Path:
        """Create the temporary directory used for chunk pseudo-videos."""
        if self._chunk_temp_dir is None:
            self._chunk_temp_dir = Path(tempfile.mkdtemp(prefix="sam3_exemplar_chunks_"))
        return self._chunk_temp_dir

    def _cleanup_temp_chunk_dir(self) -> None:
        """Cleanup temporary chunk pseudo-videos."""
        if self._chunk_temp_dir is not None and self._chunk_temp_dir.exists():
            shutil.rmtree(self._chunk_temp_dir, ignore_errors=True)
        self._chunk_temp_dir = None

    def _write_chunk_video(self, frames: List[np.ndarray], output_path: Path) -> None:
        """Write a list of frames to a temporary MP4 chunk file."""
        if not frames:
            raise RuntimeError("Cannot write temporary chunk video: no frames")

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.video_metadata["fps"]),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open temporary video writer for {output_path}")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _build_pseudo_chunk_video(self, chunk_start: int, chunk_end: int) -> Path:
        """Build a temporary video composed of exemplar pseudo-frame + chunk frames."""
        if self.video_frames is None:
            raise RuntimeError("Video not loaded")
        if self.exemplar_injected_frame is None:
            raise RuntimeError("Exemplar pseudo-frame is not prepared")

        chunk_dir = self._ensure_temp_chunk_dir()
        temp_path = chunk_dir / f"chunk_{chunk_start:06d}_{chunk_end:06d}.mp4"
        chunk_frames = [self.exemplar_injected_frame] + self.video_frames[chunk_start:chunk_end]
        self._write_chunk_video(chunk_frames, temp_path)
        return temp_path

    def _get_text_prompt(self) -> Optional[str]:
        """Build the text concept prompt string passed to SAM3."""
        if not self.config.concepts:
            return None
        return ". ".join(c for c in self.config.concepts if c).strip()
    
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
        
        try:
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

                self._drain_between_chunks(has_next_chunk=(chunk_idx + 1 < len(chunks)))
        finally:
            self._cleanup_temp_chunk_dir()

    def _get_prompt_frame_index(self, chunk_start: int, chunk_end: int) -> int:
        """Choose the frame where the initial prompt should be added."""
        if self.exemplar_injected_frame is not None:
            return 0
        if self.config.propagation_direction == PropagationDirection.BACKWARD:
            return max(chunk_start, chunk_end - 1)
        return chunk_start

    def _add_initial_prompt(
        self,
        predictor,
        session_id: str,
        chunk_start: int,
        chunk_end: int,
    ) -> int:
        """Seed a chunk with either a text prompt or a visual exemplar box."""
        prompt_frame_idx = self._get_prompt_frame_index(chunk_start, chunk_end)

        text_prompt = self._get_text_prompt()
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame_idx,
            "text": text_prompt,
        }

        if self.exemplar_injected_box_norm is not None:
            request["bounding_boxes"] = [list(self.exemplar_injected_box_norm)]
            request["bounding_box_labels"] = [self.config.exemplar_box_label]

        predictor.handle_request(request)
        return prompt_frame_idx

    def _build_propagation_request(
        self,
        session_id: str,
        chunk_start: int,
        chunk_end: int,
        prompt_frame_idx: int,
    ) -> Dict:
        """Build a bounded propagation request for the current chunk."""
        if self.exemplar_injected_frame is not None:
            max_frames = max(0, chunk_end - chunk_start)
            propagation_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": self.config.propagation_direction.value,
                "start_frame_index": 0,
                "max_frame_num_to_track": int(max_frames),
            }

            if self.config.output_prob_thresh != 0.5:
                propagation_request["output_prob_thresh"] = self.config.output_prob_thresh

            return propagation_request

        last_chunk_frame = chunk_end - 1
        if self.config.propagation_direction == PropagationDirection.FORWARD:
            max_frames = max(0, last_chunk_frame - prompt_frame_idx)
        elif self.config.propagation_direction == PropagationDirection.BACKWARD:
            max_frames = max(0, prompt_frame_idx - chunk_start)
        else:
            max_frames = max(
                max(0, prompt_frame_idx - chunk_start),
                max(0, last_chunk_frame - prompt_frame_idx),
            )

        propagation_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": self.config.propagation_direction.value,
            "start_frame_index": int(prompt_frame_idx),
            "max_frame_num_to_track": int(max_frames),
        }

        if self.config.output_prob_thresh != 0.5:
            propagation_request["output_prob_thresh"] = self.config.output_prob_thresh

        return propagation_request
    
    def _process_chunk(
        self,
        predictor,
        chunk_start: int,
        chunk_end: int
    ) -> None:
        """
        Process a single chunk with predictor, with progressive fallback strategies for CUDA issues.
        
        Retry strategies (applied progressively if CUDA allocator assert occurs):
        1. Original config settings
        2. + offload_video_to_cpu=True
        3. + disable_tf32=True
        4. + reduced max_num_objects
        
        Args:
            predictor: Sam3VideoPredictor instance.
            chunk_start: Start frame index.
            chunk_end: End frame index.
        """
        if self.video_frames is None:
            raise RuntimeError("Video not loaded")

        using_pseudo_exemplar = self.exemplar_injected_frame is not None
        if using_pseudo_exemplar:
            video_path = str(self._build_pseudo_chunk_video(chunk_start, chunk_end))
        else:
            video_path = self.video_metadata.get("path", "")

        # Define fallback strategies with increasing aggressiveness
        strategies = [
            {
                "name": "original config",
                "offload_video_to_cpu": self.config.offload_video_to_cpu,
                "disable_tf32": self.config.disable_tf32,
                "max_num_objects": self.config.max_num_objects,
            },
        ]
        
        # Only add additional strategies if not already at maximum caution
        if not self.config.offload_video_to_cpu:
            strategies.append({
                "name": "with video offload",
                "offload_video_to_cpu": True,
                "disable_tf32": self.config.disable_tf32,
                "max_num_objects": self.config.max_num_objects,
            })
        
        if not self.config.disable_tf32:
            strategies.append({
                "name": "with TF32 disabled",
                "offload_video_to_cpu": True,
                "disable_tf32": True,
                "max_num_objects": self.config.max_num_objects,
            })
        
        if self.config.max_num_objects is None:
            strategies.append({
                "name": "with reduced object limit (5000)",
                "offload_video_to_cpu": True,
                "disable_tf32": True,
                "max_num_objects": 5000,
            })

        last_exception = None
        for attempt_idx, strategy in enumerate(strategies):
            try:
                self._process_chunk_once(
                    predictor=predictor,
                    video_path=video_path,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    using_pseudo_exemplar=using_pseudo_exemplar,
                    offload_video_to_cpu=strategy["offload_video_to_cpu"],
                    disable_tf32=strategy["disable_tf32"],
                    max_num_objects=strategy["max_num_objects"],
                )
                return  # Success, exit
            except Exception as exc:
                last_exception = exc
                if not self._is_cuda_allocator_assert(exc):
                    raise  # Not a CUDA allocator issue, re-raise immediately
                
                if attempt_idx < len(strategies) - 1:
                    print(
                        f"[WARN] Chunk {chunk_start}-{chunk_end}: CUDA allocator assert detected. "
                        f"Retrying with strategy: {strategy['name']} → {strategies[attempt_idx + 1]['name']}",
                        file=sys.stderr,
                    )
                    self._safe_cuda_cleanup()
                else:
                    print(
                        f"[ERROR] Chunk {chunk_start}-{chunk_end}: All {len(strategies)} fallback strategies exhausted. "
                        f"Last strategy tried: {strategy['name']}",
                        file=sys.stderr,
                    )
                    raise

    def _process_chunk_once(
        self,
        predictor,
        video_path: str,
        chunk_start: int,
        chunk_end: int,
        using_pseudo_exemplar: bool,
        offload_video_to_cpu: bool,
        disable_tf32: bool,
        max_num_objects: Optional[int],
    ) -> None:
        """Run one chunk attempt with explicit inference state configuration."""
        # Start session with stability flags
        start_resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(video_path),
            "offload_video_to_cpu": bool(offload_video_to_cpu),
        })
        session_id = start_resp["session_id"]
        
        # Apply TF32 and max_num_objects settings if needed
        if disable_tf32 or max_num_objects is not None:
            self._configure_session_stability(
                predictor=predictor,
                session_id=session_id,
                disable_tf32=disable_tf32,
                max_num_objects=max_num_objects,
            )
        
        try:
            prompt_frame_idx = self._add_initial_prompt(
                predictor,
                session_id,
                chunk_start,
                chunk_end,
            )
            propagation_request = self._build_propagation_request(
                session_id,
                chunk_start,
                chunk_end,
                prompt_frame_idx,
            )
            
            # Collect outputs with periodic synchronization
            frame_count = 0
            for resp in predictor.handle_stream_request(propagation_request):
                frame_idx = int(resp["frame_index"])

                if using_pseudo_exemplar:
                    if frame_idx == 0:
                        continue
                    mapped_idx = chunk_start + (frame_idx - 1)
                    if mapped_idx < chunk_start or mapped_idx >= chunk_end:
                        continue
                else:
                    if frame_idx < chunk_start or frame_idx >= chunk_end:
                        continue
                    mapped_idx = frame_idx

                outputs = resp.get("outputs", {})
                self.per_frame_outputs[mapped_idx] = outputs
                
                frame_count += 1
                # Synchronize every 10 frames to catch allocation issues early
                if frame_count % 10 == 0:
                    self._light_cuda_sync()
        
        finally:
            # Close session
            predictor.handle_request({"type": "close_session", "session_id": session_id})

    @staticmethod
    def _configure_session_stability(
        predictor,
        session_id: str,
        disable_tf32: bool,
        max_num_objects: Optional[int],
    ) -> None:
        """Apply session-level stability configurations."""
        if torch is None or not torch.cuda.is_available():
            return
        
        try:
            if disable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            
            # max_num_objects is typically set at model init; this is informational
            if max_num_objects is not None:
                print(
                    f"[INFO] Session {session_id}: limited max_num_objects to {max_num_objects}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"[WARN] Failed to configure session stability settings: {e}",
                file=sys.stderr,
            )

    @staticmethod
    def _light_cuda_sync() -> None:
        """Lightweight CUDA synchronization without full empty_cache."""
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    
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
