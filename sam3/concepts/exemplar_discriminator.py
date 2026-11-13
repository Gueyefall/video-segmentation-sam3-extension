"""Opt-in positive/negative exemplar discrimination, separate from mask inference."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


ENCODER_ID = "sam3-last-fpn-spatial-8x8-v1"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("An exemplar descriptor must be a nonempty finite vector.")
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("An exemplar descriptor must have nonzero finite norm.")
    return value / length


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    positive_similarity: Optional[float] = None
    negative_similarity: Optional[float] = None
    margin: Optional[float] = None
    positive_reference: Optional[str] = None
    negative_reference: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Sam3CropEncoder:
    """Reuse a loaded SAM3 detector backbone; do not download another model."""

    encoder_id = ENCODER_ID

    def __init__(self, detector, checkpoint_sha256: str):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.detector = detector
        self.checkpoint_sha256 = checkpoint_sha256
        self.processor = Sam3Processor(detector, device=detector.device)

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        from torchvision.transforms import v2

        if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or not crop.size:
            raise ValueError("The encoder requires a nonempty uint8 BGR crop.")
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.inference_mode(), torch.autocast(
            device_type=self.detector.device.type, dtype=torch.bfloat16,
            enabled=self.detector.device.type == "cuda",
        ):
            image = v2.functional.to_image(image).to(self.detector.device)
            image = self.processor.transform(image).unsqueeze(0)
            features = self.detector.backbone.forward_image(image)["backbone_fpn"][-1]
            descriptor = torch.nn.functional.adaptive_avg_pool2d(
                features.float(), (8, 8)).flatten().cpu().numpy()
        return unit_vector(descriptor)


class ExemplarDiscriminator:
    def __init__(
        self,
        encoder: Callable[[np.ndarray], np.ndarray],
        positive: List[Tuple[str, np.ndarray]],
        negative: List[Tuple[str, np.ndarray]],
        min_similarity: float,
        min_margin: float,
        min_crop_side: int,
    ):
        if not positive or not negative:
            raise ValueError("Both positive exemplars and hard-negative exemplars are required.")
        if not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be finite and in [-1, 1].")
        if not np.isfinite(min_margin) or not 0 < min_margin <= 2:
            raise ValueError("min_margin must be finite and in (0, 2].")
        if isinstance(min_crop_side, bool) or not isinstance(min_crop_side, int) or min_crop_side < 1:
            raise ValueError("min_crop_side must be a positive integer.")
        self.encoder = encoder
        self.positive_ids = [key for key, _ in positive]
        self.negative_ids = [key for key, _ in negative]
        if len(set(self.positive_ids + self.negative_ids)) != len(positive) + len(negative):
            raise ValueError("Exemplar IDs must be unique.")
        vectors = [unit_vector(vector) for _, vector in positive + negative]
        if len({vector.shape for vector in vectors}) != 1:
            raise ValueError("All exemplar descriptors must have the same dimension.")
        self.positive = np.stack(vectors[:len(positive)])
        self.negative = np.stack(vectors[len(positive):])
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.min_crop_side = min_crop_side
        self.provenance: Dict = {}

    def score(self, descriptor: np.ndarray) -> Decision:
        descriptor = unit_vector(descriptor)
        if descriptor.shape != self.positive.shape[1:]:
            raise ValueError("Candidate and exemplar descriptor dimensions differ.")
        positive = np.clip(self.positive @ descriptor, -1.0, 1.0)
        negative = np.clip(self.negative @ descriptor, -1.0, 1.0)
        pos, neg = int(np.argmax(positive)), int(np.argmax(negative))
        best_positive, best_negative = float(positive[pos]), float(negative[neg])
        margin = best_positive - best_negative
        if best_positive < self.min_similarity:
            reason = "review_low_similarity"
        elif margin <= 0:
            reason = "reject_negative_match"
        elif margin < self.min_margin:
            reason = "review_insufficient_margin"
        else:
            reason = "accept"
        return Decision(reason == "accept", reason, best_positive, best_negative, margin,
                        self.positive_ids[pos], self.negative_ids[neg])

    def classify(self, frame: np.ndarray, mask: np.ndarray) -> Decision:
        if mask.ndim != 2 or mask.shape != frame.shape[:2]:
            raise ValueError("The candidate mask must be 2D and match the source frame.")
        if not np.all((mask == 0) | (mask == 1) | (mask == 255)):
            raise ValueError("The candidate mask must be binary, not a probability map.")
        if not mask.any():
            return Decision(False, "review_empty_mask")
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        if min(width, height) < self.min_crop_side:
            return Decision(False, "review_small_crop")
        return self.score(self.encoder(frame[y:y + height, x:x + width]))

    def filter_frame(self, frame: np.ndarray, outputs: dict) -> Tuple[dict, list]:
        ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
        masks = np.asarray(outputs["out_binary_masks"])
        if len(ids) != len(masks):
            raise ValueError("Instance IDs and masks must have equal lengths.")
        decisions = [self.classify(frame, mask) for mask in masks]
        keep = np.array([decision.accepted for decision in decisions], dtype=bool)
        filtered = dict(outputs)
        for key in ("out_obj_ids", "out_binary_masks", "out_boxes_xywh", "out_probs", "out_tracker_probs"):
            if key in outputs:
                values = np.asarray(outputs[key])
                if values.ndim == 0 or len(values) != len(ids):
                    raise ValueError(f"Misaligned per-instance output field: {key}")
                filtered[key] = values[keep]
        return filtered, [
            {"object_id": int(obj_id), **decision.to_dict()}
            for obj_id, decision in zip(ids, decisions)
        ]


def load_discriminator(path: Path, encoder: Sam3CropEncoder) -> ExemplarDiscriminator:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "encoder_id", "checkpoint_sha256", "positive", "negative",
                "min_similarity", "min_margin", "min_crop_side"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError(f"Exemplar configuration requires exactly: {sorted(required)}; schema_version=1.")
    if config["encoder_id"] != encoder.encoder_id:
        raise ValueError("Exemplar configuration uses a different feature encoder.")
    if config["checkpoint_sha256"] != encoder.checkpoint_sha256:
        raise ValueError("Exemplar configuration was calibrated with a different checkpoint.")
    references = {}
    provenance = []
    for label in ("positive", "negative"):
        references[label] = []
        for entry in config[label]:
            if set(entry) != {"id", "image"} or not isinstance(entry["id"], str) or not entry["id"]:
                raise ValueError("Each exemplar requires a nonempty id and an image path.")
            image_path = (path.parent / entry["image"]).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read exemplar image: {image_path}")
            references[label].append((entry["id"], encoder(image)))
            provenance.append({"id": entry["id"], "label": label,
                               "sha256": file_sha256(image_path)})
    result = ExemplarDiscriminator(
        encoder, references["positive"], references["negative"],
        config["min_similarity"], config["min_margin"], config["min_crop_side"])
    result.provenance = {
        "config_sha256": file_sha256(path), "encoder_id": encoder.encoder_id,
        "checkpoint_sha256": encoder.checkpoint_sha256, "references": provenance,
        "min_similarity": result.min_similarity, "min_margin": result.min_margin,
        "min_crop_side": result.min_crop_side,
    }
    return result


def filter_video_outputs(frames: list, outputs: dict, gate: ExemplarDiscriminator, report: Path) -> dict:
    if set(outputs) != set(range(len(frames))):
        raise ValueError("Incomplete inference output; do not infer missing frames as empty masks.")
    filtered = {}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "provenance", **gate.provenance}) + "\n")
        for index, frame in enumerate(frames):
            filtered[index], decisions = gate.filter_frame(frame, outputs[index])
            stream.write(json.dumps({"type": "frame", "frame_index": index,
                                     "decisions": decisions}) + "\n")
    return filtered
