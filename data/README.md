---
license: other
license_name: Mixed source-specific licenses
language: en
task_categories:
- text-classification
---
# OpenJev data v0.1

This workspace contains an actually prepared dataset for a small typed-decision
model. Public input sources and exact checksums are in SOURCE_LOCK.json.
Original texts retain the licenses and attributions in ../THIRD_PARTY.md.
The project code license does not replace dataset licenses.

Processed JSONL records contain a state, typed question, runtime candidate
descriptions, target distribution, split, source record ID, lineage group,
task and provenance. BANKING77 contributes choice and proposition views.
TweetEval sentiment contributes ordered score labels. CLINC is used only for
cross-dataset evaluation. A local urn generator provides known distributions.

The raw and processed files are excluded from Git and can be regenerated with
the pinned fetch/prepare command. Their computed counts and hashes are in
../reports/data_manifest.json. The 176 DeepSeek paraphrases are trained on only;
their teacher-reviewed labels are not independently human-verified.

Training, development, calibration and test source groups do not overlap after
normalized-text deduplication. Near-duplicate paraphrases and pretrained-model
corpus contamination are not ruled out. Unseen labels mean unseen during this
project's post-training, not unseen by the pretrained encoder.

The evaluation subsets and limitations are documented in ../reports/RESULTS.md.
Do not report the selected CLINC subset or BANKING77 candidate protocols as an
official full-benchmark score.
