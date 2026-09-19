# Visual posterior experiment v0.1

Declared on 2026-09-19 before training or inspecting test results.

## Research question

Does retaining a joint scene posterior improve compound-event probabilities over
independent marginals, and how do learned posterior readout and explicit Bayesian
fusion generalize to an unseen dependency topology?

This is a controlled research substrate, not a claim of a novel architecture or
general image understanding. The first model operates on three fixed object slots
and two binary attributes per object (color and shape): 64 possible worlds.
Natural-language input uses a documented controlled grammar; compositional query
execution is symbolic. Neither free-form language understanding nor learned
open-vocabulary grounding is claimed.

## Observation model

A supplied, normalized world prior is generated from a six-variable Ising model.
All methods receive the same prior. Each true attribute is flipped independently
with probability 0.08 by the synthetic sensor. A rendered image depicts those
noisy observations, with selected entire objects hidden by opaque occluders.
The renderer takes only observed values, not the true world or the prior.
Appearance jitter is independent of the latent world conditional on observations.

The posterior target is the prior times the known observation likelihood,
normalized over all 64 worlds. Hidden attributes contribute likelihood 1.
It is NOT a one-hot label for the hidden true world. Rendering is designed to
make each observation symbol visually recoverable; the exact reference posterior
is conditional on that observation channel.

Each independent episode produces a partially occluded and a fully visible view
of the same noisy observation. They share a split and appearance seed. Train,
development, calibration, and test episodes have independent random streams.
Finite world assignments can recur; this is not an unseen-world-label experiment.

Training/development/calibration and ID test use local and chain dependency
families. The topology test uses cross-object edges absent in training. An
appearance test uses a separately declared rendering palette. No held-out
examples are used for checkpoint selection or hyperparameter adjustment.

## Models and controls

- Joint: learned visual encoder plus a prior-conditioned 64-way posterior head.
- Independent: same encoder family, prior-conditioned six Bernoulli marginals;
  the distribution over worlds factorizes.
- Evidence: learned per-object visual observation classifier (including hidden),
  followed by exact enumeration using the supplied prior and sensor channel.
  Soft perception predictions approximate likelihoods; exact fusion does not
  guarantee that the learned perception is correct.
- Joint factorized: product of the learned joint model's own marginals. This
  isolates the loss of dependencies while preserving all unary probabilities.
- Prior-only: no image evidence.
- Oracle: exact observation metadata; unavailable to normal inference.
- Oracle factorized: product of exact posterior marginals, a strong independence
  control. Oracle controls are not learned model results.

No model receives latent truth, visibility masks, or observation metadata at
inference. Evidence-model observation labels are privileged training supervision;
its comparison with posterior-supervised models is not a supervision-matched
architecture ablation. Joint versus joint-factorized IS a matched readout ablation.

## Training and selection

Use the checked-in pilot configuration unchanged. Three seeds, all reported.
Select each checkpoint using development posterior cross entropy; evidence
checkpoints are selected using the posterior after fusion, not observation accuracy.
Fit one temperature on calibration data for each joint/independent model.
Evidence fusion has no post-hoc temperature. Report raw and calibrated metrics.
Do not apply independent per-query calibration: it can break coherence.

## Evaluation

Report exact-target posterior KL, compound-event Brier divergence, unary error,
world log loss against sampled latent truth, visibility strata, and each test
distribution separately. Query families include unary predicates, conjunctions,
disjunctions, equality, implication, and count events. A query's labels and
ordering are not model inputs.

Check query-order isolation, probability complements/partitions, save/reload,
oracle independence from hidden truth, and case grouping. Finite enumeration
gives exact algebraic consistency under the model; this does not establish
real-world calibration. The independently factorized baseline is itself coherent,
but generally models correlations incorrectly.

Benchmark image preprocessing, encoder, posterior construction and query
evaluation together; separately report cached posterior query cost. Record
hardware, precision, shapes, batch size, warmups and repeats. No Jev speed claims.

Bootstrap compound-query errors by episode (all views and queries stay together),
and report every training seed. A confidence interval over episodes does not
replace training-seed replication.

## Publication boundary

Publish code, original generator, configuration, source/data/checkpoint hashes,
small illustrative images and machine-readable reports. Do not publish credentials,
machine-specific paths, downloaded third-party weights or raw private data.
The intended license for original code, synthetic examples and trained small
models is Apache-2.0.
