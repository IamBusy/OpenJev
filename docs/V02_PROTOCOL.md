# OpenJev v0.2 protocol

Declared before evaluating Qwen or fitting its adapters.

## Base model and inference contract

Use Qwen/Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca.
It fits a bounded local LoRA experiment on the available M3 Pro and carries the
Apache-2.0 license. Gemma remains an alternative; this experiment chooses Qwen.

LoRA updates attention projections in the language model. Read the next-token
logits for letter-coded candidates in a single prefill; normalize only over valid
labels. There is no autoregressive answer decoding in the decision path.
These are conditional label-token probabilities, not automatically calibrated
epistemic uncertainty. Distinct questions are batched independently; this does
not reproduce TypeSafe's proprietary shared-state sampler.

## Data quality

Preserve v0.1 artifacts and hashes. Its synthetic records are excluded from v0.2
training pending quality review: a spot check found phone-number-like text and
ambiguous sentiment labels despite teacher agreement. Source categories also
have mechanical descriptions and easy random negatives.

Use human-labeled public data for the first Qwen adapter comparison. Training
contains short dynamic-candidate intent choice, harder intent verification and
sentiment score records. Source groups cannot cross train/dev/calibration/test.
Use more semantically similar negative intents rather than only random negatives.
The exact urn generator remains a diagnostic, not the primary language-training
corpus.

## Frozen evaluation pack

1. Existing-bank and sentiment regression slices (explicitly previously analyzed).
2. Bank intents excluded from all prior post-training, using previously unused
   source records and matched candidate counts.
3. New task-family checks: ARC-Easy multiple choice and BoolQ passage-based truth
   judgments. Their labels come from public annotations, not a teacher model.
4. A rule-following score diagnostic in English and Chinese, with exact
   programmatic targets and disjoint template families.

Pin every source and record the final record IDs, candidate order, source group,
input-token length and target. Reject examples that exceed the shared context
budget rather than silently truncating. Lock the pack before model selection.
Do not describe these bounded local subsets as official leaderboard results.
Pretraining contamination cannot be excluded.

## Comparators

- Untuned Qwen3-0.6B using the same typed readout.
- Qwen3-0.6B with trained LoRA and separately fitted temperatures.
- The delivered MiniLM-based OpenJev v0.1 checkpoint on identical records.
- An independent DeepSeek reference on a predeclared smaller paired subset, with
  API latency and token usage kept separate from local inference speed.
- A standard supervised text classifier on supported seen-intent tasks; its
  inability to recognize new labels must not be disguised as an equivalent
  general decision model.

Report accuracy, macro F1 where meaningful, NLL, Brier, calibration, and per-task
sample counts. Measure local warm latency with identical inputs and candidate
counts; keep model load and remote-network time separate. Include failures and
do not replace a selected checkpoint based on final-test results.
