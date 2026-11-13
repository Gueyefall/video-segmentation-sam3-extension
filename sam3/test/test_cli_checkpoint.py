import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_concept_segmentation as cli


class TestLocalCheckpoint(unittest.TestCase):
    def test_resolves_checkpoint_and_rejects_nonfiles(self):
        base = ["--video", "input.mp4", "--output", "output.mp4", "--concepts", "object"]
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            args = cli.build_argument_parser().parse_args(base + ["--checkpoint", str(checkpoint)])
            self.assertEqual(args.checkpoint, str(checkpoint.resolve()))
            for invalid in (checkpoint.parent, checkpoint.with_name("missing.pt")):
                with self.subTest(path=invalid), self.assertRaises(SystemExit), patch("sys.stderr"):
                    cli.build_argument_parser().parse_args(base + ["--checkpoint", str(invalid)])

    def test_passes_local_checkpoint_to_predictor_and_preserves_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "input.mp4"
            video.touch()
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            base = ["run_concept_segmentation", "--video", str(video), "--output",
                    str(root / "output.mp4"), "--concepts", "object"]
            predictor = Mock(side_effect=RuntimeError("stop after predictor initialization"))
            module = Mock(Sam3VideoPredictor=predictor)
            with patch.object(cli, "ConceptSegmentationStrategy"), \
                    patch.object(cli.torch.cuda, "is_available", return_value=False), \
                    patch.dict("sys.modules", {"sam3.model.sam3_video_predictor": module}), \
                    patch("sys.stderr"):
                for extra, expected in (([], None), (["--checkpoint", str(checkpoint)], str(checkpoint))):
                    with self.subTest(extra=extra), patch("sys.argv", base + extra):
                        with self.assertRaises(SystemExit):
                            cli.main()
                        self.assertEqual(predictor.call_args.kwargs["checkpoint_path"], expected)

    def test_exemplar_filter_requires_checkpoint_before_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video, config = root / "input.mp4", root / "filter.json"
            video.touch()
            config.write_text("{}", encoding="utf-8")
            arguments = ["run_concept_segmentation", "--video", str(video), "--output",
                         str(root / "output.mp4"), "--concepts", "object",
                         "--filter-exemplars", str(config)]
            with patch("sys.argv", arguments), patch("sys.stderr") as error, \
                    patch.object(cli, "ConceptSegmentationStrategy") as strategy:
                with self.assertRaises(SystemExit):
                    cli.main()
                strategy.assert_not_called()
                self.assertIn("--checkpoint", "".join(call.args[0] for call in error.write.call_args_list))

    def test_opt_in_filter_is_applied_before_mask_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video, checkpoint, config = root / "input.mp4", root / "model.pt", root / "filter.json"
            for path in (video, checkpoint, config):
                path.touch()
            frame = np.zeros((3, 3, 3), np.uint8)
            original = {0: {"out_binary_masks": np.ones((2, 3, 3), bool)}}
            selected = {0: {"out_binary_masks": np.ones((1, 3, 3), bool)}}
            arguments = ["run_concept_segmentation", "--video", str(video), "--output",
                         str(root / "masks"), "--concepts", "object", "--checkpoint", str(checkpoint),
                         "--output-mode", "binary_masks_only", "--filter-exemplars", str(config)]
            strategy = Mock(video_frames=[frame], video_metadata={"num_frames": 1})
            strategy.get_all_outputs.return_value = original
            strategy.get_summary.return_value = {"frames_with_output": 1}
            with patch("sys.argv", arguments), patch("sys.stderr"), \
                    patch.object(cli, "ConceptSegmentationStrategy", return_value=strategy), \
                    patch.object(cli, "MaskProcessor") as processor, \
                    patch.object(cli.torch.cuda, "is_available", return_value=False), \
                    patch.dict("sys.modules", {"sam3.model.sam3_video_predictor": Mock()}), \
                    patch("concepts.exemplar_discriminator.Sam3CropEncoder"), \
                    patch("concepts.exemplar_discriminator.file_sha256", return_value="hash"), \
                    patch("concepts.exemplar_discriminator.load_discriminator") as load, \
                    patch("concepts.exemplar_discriminator.filter_video_outputs", return_value=selected) as apply:
                processor.return_value.process_frame.return_value = (frame, [])
                cli.main()
                apply.assert_called_once_with(
                    [frame], original, load.return_value, root / "masks_exemplar_decisions.jsonl")
                exported = processor.return_value.save_mask_frames.call_args.args[0]
                np.testing.assert_array_equal(exported[0], selected[0]["out_binary_masks"])
