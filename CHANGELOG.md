# Changelog

## 0.3.2 — 2026-09-19

- Standardize the public model name as `OpenJev-0.6B`; track training and software
  versions separately from the model name. The trained tensor bytes are unchanged.
- Add the public `OpenJevModel` SDK export and `openjev-model` command, retaining
  existing implementation imports, old commands and local checkpoint fallback.
- Rename the Hugging Face repository with redirects and update model cards,
  examples, download metadata and the release bundle.

## 0.3.1 — 2026-09-19

- Publish the existing v0.3 trained adapter and scalar head on Hugging Face.
- Add `BranchDecision.from_pretrained` with pinned base download, file verification
  and automatic calibration; support the standard root-level PEFT layout.
- Preserve the original trained weights and the GitHub v0.3.0 release.

## 0.3.0 — 2026-09-19

First public experimental release.

- Shared Qwen state prefix with independent candidate branches and a trained
  scalar head; up to 255 candidates per choice, 512 branches per request.
- Local training, frozen data preparation, calibration, evaluation and HTTP API.
- Public checkpoint download with archive and per-file checksum verification.
- Exact v0.3 scenario reconstruction without private provider traces or API keys.
- Forward/gradient cache-equivalence tests, source checks and CPU CI.
- English/Chinese entry points, model card, attribution and reproduction guide.
- Historical v0.1/v0.2 experiments and measured regressions retained.

Known limits include short-input latency regressions, weaker results on some
banking and held-out rule tasks, one training seed, narrow calibration coverage,
and MPS/CPU-only execution. No RLCD reproduction or Jev parity is claimed.
