import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from concepts.exemplar_discriminator import (
    ENCODER_ID, ExemplarDiscriminator, Sam3CropEncoder, filter_video_outputs, load_discriminator,
)


def make_filter():
    return ExemplarDiscriminator(
        lambda crop: np.array([float(crop[0, 0, 0]), float(crop[0, 0, 1])]),
        [("wanted", np.array([1.0, 0.0]))], [("similar", np.array([0.0, 1.0]))],
        min_similarity=0.8, min_margin=0.2, min_crop_side=2,
    )


class TestExemplarDiscriminator(unittest.TestCase):
    def test_crop_encoder_uses_detector_device_without_requiring_cuda(self):
        detector = Mock(device=torch.device("cpu"))
        detector.backbone.forward_image.side_effect = lambda image: {"backbone_fpn": [image]}
        encoder = Sam3CropEncoder(detector, checkpoint_sha256="test")
        descriptor = encoder(np.full((4, 4, 3), 255, dtype=np.uint8))
        self.assertEqual(encoder.processor.device, torch.device("cpu"))
        self.assertEqual(descriptor.shape, (3 * 8 * 8,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=5)

    def test_requires_positive_similarity_and_margin_over_hard_negative(self):
        gate = make_filter()
        self.assertTrue(gate.score(np.array([1.0, 0.0])).accepted)
        self.assertFalse(gate.score(np.array([0.0, 1.0])).accepted)
        gate.min_margin = 0.3
        decision = gate.score(np.array([0.8, 0.6]))
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "review_insufficient_margin")
        self.assertEqual(decision.positive_reference, "wanted")
        self.assertEqual(decision.negative_reference, "similar")

    def test_small_empty_and_invalid_masks_do_not_trigger_encoding(self):
        gate = make_filter()
        gate.encoder = Mock(side_effect=AssertionError("Must not encode an unobservable candidate"))
        frame = np.ones((5, 5, 3), np.uint8)
        mask = np.zeros((5, 5), bool)
        self.assertEqual(gate.classify(frame, mask).reason, "review_empty_mask")
        mask[0, 0] = True
        self.assertEqual(gate.classify(frame, mask).reason, "review_small_crop")
        with self.assertRaisesRegex(ValueError, "match"):
            gate.classify(frame, np.ones((3, 3), bool))

    def test_filters_all_aligned_instance_fields_without_merging_masks(self):
        frame = np.zeros((8, 8, 3), np.uint8)
        frame[1:4, 1:4, 0] = 255
        frame[4:7, 4:7, 1] = 255
        masks = np.zeros((2, 8, 8), bool)
        masks[0, 1:4, 1:4] = True
        masks[1, 4:7, 4:7] = True
        original = {"out_obj_ids": np.array([5, 9]), "out_binary_masks": masks,
                    "out_boxes_xywh": np.array([[1, 1, 3, 3], [4, 4, 3, 3]]),
                    "out_probs": np.array([0.9, 0.8]), "metadata": "unchanged"}
        filtered, decisions = make_filter().filter_frame(frame, original)
        np.testing.assert_array_equal(filtered["out_obj_ids"], [5])
        np.testing.assert_array_equal(filtered["out_binary_masks"], masks[:1])
        np.testing.assert_array_equal(filtered["out_boxes_xywh"], [[1, 1, 3, 3]])
        self.assertEqual(filtered["metadata"], "unchanged")
        self.assertEqual(len(original["out_obj_ids"]), 2)
        self.assertEqual([entry["accepted"] for entry in decisions], [True, False])

    def test_invalid_descriptors_and_misaligned_outputs_fail_explicitly(self):
        gate = make_filter()
        for vector in (np.array([np.nan, 0]), np.zeros(2), np.ones(3)):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                gate.score(vector)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            gate.filter_frame(np.zeros((3, 3, 3), np.uint8), {
                "out_obj_ids": np.array([1]), "out_binary_masks": np.zeros((0, 3, 3), bool)})

    def test_configuration_checks_checkpoint_and_preserves_asset_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("positive", "negative"):
                self.assertTrue(cv2.imwrite(str(root / f"{name}.png"), np.ones((4, 4, 3), np.uint8)))
            config = {
                "schema_version": 1, "encoder_id": ENCODER_ID, "checkpoint_sha256": "a" * 64,
                "positive": [{"id": "p", "image": "positive.png"}],
                "negative": [{"id": "n", "image": "negative.png"}],
                "min_similarity": 0.8, "min_margin": 0.2, "min_crop_side": 2,
            }
            path = root / "filter.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            encoder = Mock(encoder_id=ENCODER_ID, checkpoint_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                load_discriminator(path, encoder)
            encoder.assert_not_called()
            encoder.checkpoint_sha256 = "a" * 64
            encoder.side_effect = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            gate = load_discriminator(path, encoder)
            self.assertEqual(len(gate.provenance["references"]), 2)
            self.assertEqual(len(gate.provenance["config_sha256"]), 64)

    def test_video_filter_requires_complete_frames_and_persists_decisions(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[:, :, 0] = 255
        output = {"out_obj_ids": np.array([3]), "out_binary_masks": np.ones((1, 4, 4), bool)}
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "decisions.jsonl"
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                filter_video_outputs([frame, frame], {0: output}, make_filter(), report)
            self.assertFalse(report.exists())
            filtered = filter_video_outputs([frame], {0: output}, make_filter(), report)
            np.testing.assert_array_equal(filtered[0]["out_obj_ids"], [3])
            records = [json.loads(line) for line in report.read_text().splitlines()]
            self.assertEqual(records[0]["type"], "provenance")
            self.assertEqual(records[1]["decisions"][0]["reason"], "accept")

    def test_video_filter_never_overwrites_an_existing_report(self):
        frame = np.ones((3, 3, 3), np.uint8)
        outputs = {0: {"out_obj_ids": np.array([3]),
                       "out_binary_masks": np.ones((1, 3, 3), bool)}}
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "existing.jsonl"
            report.write_text("existing data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                filter_video_outputs([frame], outputs, make_filter(), report)
            self.assertEqual(report.read_text(encoding="utf-8"), "existing data")


if __name__ == "__main__":
    unittest.main()
