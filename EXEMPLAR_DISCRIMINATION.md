# Positive/negative exemplar discrimination

## Purpose and scope

A segmentation prompt can find a broad object category without distinguishing
visually similar subclasses. This optional **post-inference instance filter**
keeps only candidates whose appearance matches positive references more strongly
than hard-negative references. Existing segmentation behavior is unchanged unless
`--filter-exemplars` is supplied.

The method is generic: the application supplies the wanted and unwanted visual
classes. It does not contain a fixed vocabulary, object-specific rules, or
machine-specific configuration.

This is **not** SAM3's native visual-prompt API, a new trained classifier, OCR, a
vision-language model, or a guarantee of correct class identity. It reuses the
loaded SAM3 image backbone as a frozen feature extractor and makes an explicit,
auditable nearest-exemplar decision.

## Data flow

1. SAM3 generates and tracks candidate instances using the existing pipeline.
2. For each nonempty instance mask, take the tight axis-aligned bounding box from
   the original decoded BGR frame. Do not crop the colorized overlay.
3. If either crop dimension is below the configured minimum, abstain without
   encoding it. This prevents a tiny, unreadable symbol from being confidently
   classified merely by its background color.
4. Apply SAM3's standard image preprocessing: resize to 1008 by 1008, scale image
   values, and normalize with the existing processor. Extract the last
   `backbone_fpn` feature map from the loaded detector's image backbone.
5. Spatially average-pool this feature map to an **8 by 8 grid**, flatten all
   channels and cells, and L2-normalize the descriptor. Keeping spatial cells
   preserves more symbol/layout information than global average pooling.
6. Encode the positive and negative reference images once with exactly the same
   extractor. References should already be tightly cropped to the object face.
7. Compare each candidate descriptor with the reference descriptors, then retain
   or exclude the whole instance. Accepted mask pixels and instance IDs are
   unchanged; this stage does not dilate, erode, merge, or invent mask pixels.

For unit candidate descriptor `z`, positive bank `P`, and negative bank `N`:

```text
positive_similarity = max(dot(z, p) for p in P)
negative_similarity = max(dot(z, n) for n in N)
margin = positive_similarity - negative_similarity

accept iff positive_similarity >= min_similarity
       and margin >= min_margin
       and crop dimensions satisfy min_crop_side
```

Cosine similarities and their margin are **not probabilities**. A high absolute
similarity alone is insufficient: two subclasses may share shape, color, and
most of their visual structure.

### Decisions and abstention

Every candidate has an audit entry containing its object ID, decision reason,
best positive/negative reference IDs, both similarities, and the margin.

| Reason | Retained for editing? | Meaning |
|---|---|---|
| `accept` | Yes | Both configured appearance criteria pass |
| `review_small_crop` | No | Insufficient spatial detail |
| `review_empty_mask` | No | No visible candidate support |
| `review_low_similarity` | No | No sufficiently similar positive reference |
| `reject_negative_match` | No | Negative similarity is at least positive similarity |
| `review_insufficient_margin` | No | Positive preference is too weak |

Review outcomes are intentionally excluded, not silently treated as accepted
objects. Invalid images, nonbinary masks, nonfinite descriptors, missing output
frames, and incompatible configurations fail explicitly.

## Configuration and CLI

Create a JSON configuration outside the source checkout when using private data:

```json
{
  "schema_version": 1,
  "encoder_id": "sam3-last-fpn-spatial-8x8-v1",
  "checkpoint_sha256": "<SHA-256 of your exact checkpoint>",
  "positive": [{"id": "wanted-example", "image": "references/wanted.png"}],
  "negative": [
    {"id": "similar-but-unwanted", "image": "references/other.png"}
  ],
  "min_similarity": 0.90,
  "min_margin": 0.10,
  "min_crop_side": 32
}
```

**These threshold values are illustrative, not calibrated defaults.** All three
decision settings are required. Relative image paths resolve against the JSON
file's directory. Both banks must be nonempty and all reference IDs unique.
The loader rejects unknown configuration fields and a checkpoint/encoder mismatch.
Use a separate attribution manifest for source URLs, authors, licenses, and changes.

From the `sam3` directory:

```bash
python run_concept_segmentation.py \
  --video input.mp4 --concepts "broad object category" \
  --checkpoint /path/to/sam3.pt \
  --filter-exemplars /path/to/filter.json \
  --exemplar-report /path/to/decisions.jsonl \
  --output filtered.mp4
```

Without `--exemplar-report`, the report is written next to the requested output as
`OUTPUT_STEM_exemplar_decisions.jsonl`. Its first record contains configuration,
checkpoint, encoder, and reference-file hashes; subsequent records contain
per-frame decisions. An explicit local checkpoint is required so calibration
cannot silently be applied to different weights.
Reports must be new files: existing files are never overwritten, and the CLI
rejects report paths that overlap the output video or mask directory before
inference. Choose a new path for each run. A failed filtering run may leave an
incomplete report; treat it as incomplete and use a new path when retrying.

`--exemplar-image` is the older pseudo-frame prompting mechanism. It changes the
input to inference. `--filter-exemplars` is independent and filters inference
outputs. Do not conflate these two mechanisms or assume their effects are equivalent.

### Python API

```python
from pathlib import Path
from concepts.exemplar_discriminator import (
    Sam3CropEncoder, file_sha256, load_discriminator,
)

encoder = Sam3CropEncoder(
    predictor.model.detector, file_sha256(Path("sam3.pt"))
)
gate = load_discriminator(Path("filter.json"), encoder)
filtered_outputs, decisions = gate.filter_frame(original_frame, sam3_outputs)
```

The filter keeps all aligned per-instance output fields consistent. A custom
encoder can also be supplied to `ExemplarDiscriminator` directly, but thresholds
and references must be recalibrated for that encoder.

## Reference selection and calibration

1. **Define the visual distinction.** For example, selecting a particular printed
   label requires preserving the label's visible details, not just the object's
   general shape.
2. Collect positive exemplars and **hard negatives** that resemble the target.
   Include the closest confusable classes, not only unrelated background images.
3. Match reference styles across classes. Mixing photographic positives with only
   vector-art negatives can cause domain/style similarity to dominate class
   similarity. More positive images can therefore make a filter worse.
4. Prepare separate development and held-out examples using the exact candidate
   cropping path used in deployment. Annotate true class identity independently
   of the model's output. Avoid treating adjacent frames of one track as
   independent validation scenes.
5. On development data, inspect positive similarity, negative similarity, margin,
   crop dimensions, and failure cases. Choose explicit acceptance thresholds and
   a minimum readable crop size. If positives and hard negatives overlap in
   score space, do not claim a working discriminator by adjusting a threshold
   to hide the overlap.
6. Freeze the reference bank and thresholds **before** held-out evaluation.
   Report false accepts, false rejects, abstentions, target coverage, and scene
   counts. Report performance both per instance-frame and per independent track
   or scene; large frame counts alone can be misleading.
7. Recalibrate when changing the checkpoint, encoder, reference bank, preprocessing,
   precision/backend, or image domain. Hash checks establish provenance, not
   generalization or scientific validity.

The crop encoder uses the loaded detector's device. CUDA encoding explicitly uses
BF16 autocast; a separately loaded CPU detector uses FP32 and needs separate
validation. The repository's video predictor and crop-validation runner require
CUDA; encoder-level CPU support does not make the video pipeline CPU-compatible.
Candidate feature extraction adds inference cost; include it in runtime
comparisons rather than reporting segmentation time alone.

### Reproducible frozen-configuration validation

Prepare lossless candidate crops and a separate annotation file:

```json
{
  "split": "heldout",
  "examples": [
    {"id": "p1", "scene": "scene-a", "image": "crops/p1.png", "label": "positive"},
    {"id": "n1", "scene": "scene-b", "image": "crops/n1.png", "label": "negative"}
  ]
}
```

Paths resolve relative to this annotation file. Labels must come from independent
inspection, not the discriminator's own scores. Preserve the crop generation and
source-frame provenance privately when the inputs are confidential.

```bash
python scripts/validate_exemplar_discriminator.py \
  --checkpoint /path/to/sam3.pt --config /path/to/filter.json \
  --examples /path/to/heldout.json --report /path/to/validation.json
```

This CUDA validation runner evaluates the frozen configuration; it does not fit
thresholds on the held-out set. It reports accepted precision, false accepts,
positive acceptance, abstentions, scene counts, per-example scores/timing, and
image/configuration hashes. Use `split: "development"` for tuning experiments.
The validation report must also be a new file.
Do not silently re-label a failed held-out set as a successful validation after
tuning to it.

No private dataset or task-specific accuracy claim is included here. Unit tests
verify decision logic, field alignment, and failure handling, not real-world
classification accuracy.

## Combining appearance and temporal consistency

Appearance discrimination and forward/backward consistency solve different
problems. The first asks whether a candidate resembles the intended class; the
second asks whether independent temporal passes agree on its visible support.

For a conservative editing workflow:

1. Run independent forward and backward sessions over the same decoded frames.
2. Apply the same frozen exemplar filter to each direction.
3. Map both outputs to source frame indices; object IDs may differ across sessions.
4. Spatially match unambiguous instances and apply an explicit overlap threshold.
5. If accepted, intersect their masks; flag unmatched or ambiguous instances.
6. Keep the agreed visible mask separate from placement geometry, so disputed
   border pixels do not automatically rescale the inserted image.

The predictor's `propagation_direction="both"` is not this independent check.
Two directions can share a semantic error; temporal agreement is not a substitute
for class validation. Chunk resets do not themselves provide global identity or
long-horizon re-identification.

## Timing, reproducibility, and publication

Record source frame count/rate, checkpoint and reference hashes, code revision,
precision, chunk settings, prompts, device/software versions, and concurrent load.
Measure model loading, reference encoding, segmentation, crop discrimination,
temporal matching, rendering, and persisted-output validation separately. CUDA
work must be synchronized at timing boundaries. Also report end-to-end wall time.
Do not compare a cold run with a warm run without documenting that difference.

For replacement, validate the actual saved output, including frame completeness
and unchanged pixels outside the accepted mask. Lossy codecs cannot support an
exact unchanged-pixel claim.

Publish only code, synthetic tests, and assets whose reuse rights are verified.
Keep confidential media, candidate crops, annotations, calibration settings,
feature banks, absolute deployment paths, and runtime logs out of public commits.
Feature vectors are derived data and should not be assumed safe to publish.
Open model weights have their own licensing terms; this module does not relicense
them. Public image assets also retain their individual licenses and attribution
requirements; a source URL alone is not a reuse license.

## Limitations

- Candidate generation can miss targets; post-filtering cannot recover them.
- Cropped FPN descriptors are not trained or guaranteed to read small text or
  distinguish every symbol. Fine-grained numerals may require a separately
  validated OCR or specialized classifier.
- Blur, severe perspective, occlusion, lighting, background leakage, and
  distribution shifts can invalidate similarity thresholds.
- A similar but previously unseen negative class can still be accepted.
- This filter neither refines geometry nor reconstructs hidden object surfaces.
- Conservative abstention or mask intersection can reduce coverage and leave
  original border pixels visible. Report that trade-off rather than describing
  it as an unconditional quality improvement.
