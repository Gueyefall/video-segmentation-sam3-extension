"""Evaluate a frozen exemplar configuration on independently labeled candidate crops."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concepts.exemplar_discriminator import Sam3CropEncoder, file_sha256, load_discriminator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This validation runner requires CUDA; no unvalidated CPU fallback is used.")
    if not args.checkpoint.is_file():
        parser.error("--checkpoint must be a local file.")
    dataset = json.loads(args.examples.read_text(encoding="utf-8"))
    if dataset["split"] not in ("development", "heldout"):
        parser.error("Dataset split must be development or heldout.")
    examples = dataset["examples"]
    if not examples or {entry["label"] for entry in examples} != {"positive", "negative"}:
        parser.error("Validation needs both positive and negative labeled examples.")
    ids = [entry["id"] for entry in examples]
    if len(set(ids)) != len(ids) or any(not entry["scene"] for entry in examples):
        parser.error("Example IDs must be unique and each example must name its source scene.")
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    predictor = Sam3VideoPredictor(checkpoint_path=str(args.checkpoint.resolve()), bpe_path=None)
    gate = load_discriminator(
        args.config, Sam3CropEncoder(predictor.model.detector, file_sha256(args.checkpoint)))
    records = []
    counts = {"true_accept": 0, "false_accept": 0, "positive_nonaccept": 0, "negative_nonaccept": 0,
              "positive_review": 0, "negative_review": 0}
    for entry in examples:
        path = (args.examples.parent / entry["image"]).resolve()
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read labeled candidate crop: {path}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        decision = gate.classify(image, np.ones(image.shape[:2], dtype=bool))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        positive = entry["label"] == "positive"
        outcome = ("true_accept" if positive else "false_accept") if decision.accepted else (
            "positive_nonaccept" if positive else "negative_nonaccept")
        counts[outcome] += 1
        if decision.reason.startswith("review_"):
            counts[f"{entry['label']}_review"] += 1
        records.append({
            "id": entry["id"], "scene": entry["scene"], "label": entry["label"],
            "image_sha256": file_sha256(path), "inference_seconds": seconds, **decision.to_dict(),
        })
    positives = sum(entry["label"] == "positive" for entry in examples)
    negatives = len(examples) - positives
    accepted = counts["true_accept"] + counts["false_accept"]
    result = {
        "split": dataset["split"], "configuration": gate.provenance,
        "examples_sha256": file_sha256(args.examples),
        "annotation_notes": dataset.get("annotation_notes"),
        "counts": counts, "positive_examples": positives, "negative_examples": negatives,
        "unique_scenes": sorted({entry["scene"] for entry in examples}),
        "accepted_precision": counts["true_accept"] / accepted if accepted else None,
        "positive_acceptance_rate": counts["true_accept"] / positives,
        "false_accept_rate": counts["false_accept"] / negatives,
        "examples": records,
        "limitations": "Small or correlated crop sets do not establish population accuracy.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
