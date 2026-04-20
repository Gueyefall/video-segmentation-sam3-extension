# segmentation_with_concepts

This repository packages an updated SAM3 codebase plus concept-driven segmentation workflows for research and engineering use.

## What This Repo Contains

- A vendored SAM3 codebase under `sam3/`.
- Concept-focused tooling for text-prompt video segmentation and mask post-processing.
- Project-level documentation comparing architecture and integration choices.

## How It Differs Globally From Original SAM3

Compared with the original SAM3 repository, this project is organized as a full workspace around SAM3, with additional concept-segmentation utilities and integration documentation:

- Adds a top-level project layer (documentation and orchestration) around the upstream SAM3 package.
- Includes concept modules in `sam3/concepts/` (configuration, mask processing, utilities, and strategy orchestration).
- Adds end-to-end runner scripts such as `sam3/run_concept_segmentation.py` and `sam3/run_binary_mask_converter.py`.
- Includes project documentation focused on architecture, integration, and developer usage.

## Dependency Note (Important)

Dependencies should match the original SAM3 repository baseline:

- Upstream SAM3 repository: https://github.com/facebookresearch/sam3
- Upstream install/dependency guidance: https://github.com/facebookresearch/sam3#installation

In this workspace, dependency declarations and installation context are primarily under:

- `sam3/pyproject.toml`
- `sam3/README.md`

## Documentation Map

For project-specific information, see:

- `ARCHITECTURE_COMPARISON.md`
- `INTEGRATION_GUIDE.md`
- `DEVELOPER_GUIDE.md`
- `sam3/concepts/README.md`
- `sam3/README.md`

## Scope

This repository is intended to be published as the whole project folder, not only the nested `sam3` package.