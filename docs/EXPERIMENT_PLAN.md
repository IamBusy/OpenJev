# OpenJev v0.1 experiment plan

Status: designed before training or reading test results, 2026-09-19.

OpenJev is an independent Jev-inspired research implementation. No affiliation with
TypeSafe and no claim to reproduce its unpublished architecture or RLCD recipe.

## Research questions

1. Can a small pretrained text encoder plus a shared decision scorer improve
   dynamic-candidate decisions over the untrained semantic-similarity baseline?
2. Does fitting temperatures on a disjoint calibration split improve probability
   quality on untouched tests? Can it still fail after a distribution shift?
3. What is the measured local cost of returning complete typed distributions at
   different batch sizes and candidate counts, with all inference costs included?

## Initial implementation

Use an existing compact encoder and a shared candidate scoring function. The
candidate set is supplied by the caller; the output head has no per-label weights.
Choice returns a categorical distribution; Noul returns P(true); Score returns a
distribution over ordered rubric levels and its expected level. Type validity is
checked in ordinary code. It is not a guarantee of factual accuracy.

Start with a frozen encoder and trained residual scoring head so the entire
experiment can run on the available Mac. This is a learned decision model, with a
frozen pretrained backbone, not training a foundation model from scratch. A
fine-tuning extension is considered only after the baseline is reproducible.
Embeddings are batched; this does not imply Jev's proprietary shared-state sampler.

## Data protocol

Use clearly licensed public data first. Pin upstream revisions, save original
files with SHA-256 hashes, retain source IDs, and preserve official test splits.
Deduplicate normalized source texts across splits before creating any examples.
No paraphrases, alternative candidate sets, or related views of one source text
may cross between train, development, calibration, and test.

Use BANKING77 for dynamic intent choice and proposition verification. Derive
development and calibration only from its official training set. Predeclare a
subset of labels as unseen during training, development, and calibration; test
them separately. Use an independent public corpus as a cross-dataset transfer
test. Add a separate ordered-label task if its source permissions are clear.
These are post-training holdouts; pretrained-encoder corpus contamination cannot
be excluded and must be disclosed.

No closed-model teacher output is ground truth merely because it came from a
strong model. The optional synthesis pipeline must record provider/model,
parameters, prompt version, source seed, raw response, validation status, and
token usage. It uses training seeds only. Teacher API execution requires an
available authorized endpoint; public-data experiments do not depend on it.

## Training and selection

Select a checkpoint using development NLL only; fit temperatures on calibration
only; freeze choices before running test. Log seeds, hyperparameters, wall time,
hardware, dependency lock, model revision, data hashes, and checkpoint hashes.
Save both trainable weights and the exact frozen-backbone reference. Verify
save/reload prediction equivalence before reporting.

## Evaluation

- Accuracy and macro F1 where labels are comparable.
- NLL, multiclass Brier score, fixed-bin ECE and reliability-bin counts.
- Risk versus coverage and accuracy at stated retained coverage, with ties and
  sample counts reported. A confidence statistic is not automatically calibrated.
- In-domain, unseen-label and cross-dataset results separated by task and split.
- Candidate-order permutation consistency, candidate-count stress, unknown
  candidates, and isolation of independent questions.
- Latency p50/p95 with hardware, warmup, repeats, precision, input length,
  candidate count, batch size, and caching policy. Include tokenization, encoder,
  scorer and output construction; separate cold model load from warm inference.
- Compare semantic similarity, trained scorer before calibration, and calibrated
  scorer on exactly the same frozen records. Do not infer a speedup over Jev from
  comparing local timing to a vendor's network API number.

Failure to beat a baseline is a reportable research result. Success for this
first milestone is a complete, honestly measured and independently rerunnable
pipeline, not a predetermined metric or superiority claim.
