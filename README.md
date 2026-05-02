# segmentation_with_concepts

This workspace contains SAM3 together with the `concepts/` module, which adds concept-driven video segmentation tooling on top of SAM3.

## Contents

- `sam3/` — SAM3 source code and model files.
- `sam3/concepts/` — Concept segmentation module: configuration, chunk processing, mask rendering, binary mask tools.
- `sam3/run_concept_segmentation.py` — CLI for end-to-end concept-based video segmentation.
- `sam3/run_binary_mask_converter.py` — CLI for binary mask post-processing.
- `sam3/scripts/overlay_with_binary_selector.py` — CLI for compositing a mask video onto a base video using a binary selector.
- `SL30_image_examplar_*.webp` — Sample speed-limit-30 exemplar images used by the external-image prompting workflow.

## Installation

Dependencies and installation instructions are managed by SAM3:

- `sam3/pyproject.toml`
- `sam3/README.md`
- Upstream: https://github.com/facebookresearch/sam3

## Documentation

- `INTEGRATION_GUIDE.md` — Usage examples and CLI reference.
- `DEVELOPER_GUIDE.md` — Module structure and extension guide.
- `sam3/concepts/README.md` — `concepts/` module API reference.