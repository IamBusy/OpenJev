# OpenJev v0.3 protocol

Declared before training the new candidate-scoring architecture or evaluating
its fresh holdout.

## Architecture

Qwen3-0.6B is the backbone. A state/system prefix is encoded once. Each independent
question/candidate suffix receives a fresh cache container referring to the same
prefix tensors. Candidates cannot attend to other candidates or questions.
Their positions depend on the prefix and their own suffix only.

A shared scalar head reads the last valid candidate hidden state. It starts from
the pretrained Yes-minus-No semantic direction and is then an independent
trainable head. The LM vocabulary projection and A–Z label mapping are absent
from prediction. Valid scores are normalized within each question.

The implementation must verify cached/recomputed forward and gradient equivalence,
prefix-cache immutability, permutation equivariance, independent-question
isolation, and large candidate sets. Inference caches live within one request.
Training caches live within one state group and never cross optimizer updates.
Training uses zero dropout to make sharing deterministic. This is experimental
differentiable prefix reuse, not untested reuse of a generation cache.

Shared prefix computation does not make suffix computation free. Benchmark cached
versus recomputed execution of the same model, same input and same batch sizes.
Report short- and long-state cases, compute counts, precision and numerical
differences. No speedup claim over Jev's remote API is implied.

## Data

Keep every earlier artifact unchanged. Use a bounded public-data mixture plus
software-state scenarios with several independent questions sharing a state.
Refund and incident scenarios are training families. Access and shipping scenarios
are held-out families. Labels come from executable rules over recorded facts.
Missing information is represented explicitly as an 'insufficient information'
choice; it is not assigned an arbitrary 0.5 probability.

DeepSeek may render at most 48 selected worlds and independently reconstruct their
facts for verification (at most 96 calls). It never creates the gold outcome.
Reject contradictory, contact-like or invalid text. Preserve raw outputs and
provenance. Deterministic templates remain a separately marked source.

Public BoolQ and ARC training records are newly allowed in this version, with their
source-specific licenses retained. Fresh held-out records must exclude v0.2 test
IDs and overlapping state hashes. Entire state groups stay in one split.

## Selection and evaluation

Predeclare the data and source hashes. Optimize per-question cross entropy;
select only by development NLL. Fit per-primitive temperatures on calibration.
This is not a claim to reproduce the unpublished RLCD algorithm.

Compare the untrained branch scorer, the development-selected branch scorer and
v0.2 Qwen on identical fresh and old-regression records. Keep old and fresh results
separate. A bounded 64-question DeepSeek reference may be used after the paired
subset is fixed. Do not use test results to select a checkpoint.

Report per-task accuracy, macro F1, NLL, Brier, calibration and known limitations.
Known limitations to investigate: independently scored 'none of the above' options
and genuinely set-relative instructions may need a set-aware scorer; deterministic
rule worlds do not establish real-world probability calibration.
