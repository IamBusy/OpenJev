# Reproduce OpenJev

Use a separate clone for new experiments: preparation and training write local
manifests and selection reports. Published reports describe the original run;
keep your regenerated reports separate when comparing results. Existing prepared
splits and run names are rejected rather than silently overwritten.

## Frozen v0.3 data, without a provider key

After installing with `uv sync --frozen --extra qwen --extra dev`, run in order:

```bash
uv run --no-sync openjev-model download --base-only
uv run --no-sync openjev prepare
uv run --no-sync openjev-qwen prepare
uv run --no-sync openjev-model prepare
```

The first data stage downloads pinned BANKING77, CLINC and TweetEval sources.
The second constructs the earlier evaluation pack and downloads BoolQ/ARC test
sources; it also needs the pinned MiniLM tokenizer. The third adds BoolQ/ARC
training sources and reconstructs the exact v0.3 scenarios from
`data/v03/WORLD_BALANCE_PLAN.json`. That snapshot contains fictional facts,
accepted generated narratives and deterministic templates, with lineage hashes.
It contains no credentials or raw provider requests. Scenario labels are computed
by the rule oracles in `src/openjev/data_v03.py`.

Compare the generated `data/v03/processed/manifest.json` with the published
`reports/v03/data_manifest.json` from an untouched checkout. Expected counts:
725 train, 176 development, 167 calibration, 405 fresh test, 60 reference.
The reference is an intentional subset of fresh test, not another held-out set.
The older 448-question regression pack stays separate.

`openjev-model prepare --replay-rendering` is an optional **local historical**
workflow requiring private saved provider responses. It is not needed for public
reproduction. `--live-rendering` explicitly calls DeepSeek, consumes provider
credits and may produce different narratives; it requires `DEEPSEEK_API_KEY`.
Never compare a live-rendered run as a byte-identical reproduction of this release.

## Training, selection and evaluation

The full cross-version evaluation needs both Qwen runs:

```bash
uv run --no-sync openjev-qwen train --run qwen-reproduction
uv run --no-sync openjev-model train --run branch-reproduction
uv run --no-sync openjev-model evaluate
uv run --no-sync python scripts/report_v03.py
uv run --no-sync openjev-model export --output artifacts/my-branch-model
```

Configuration files pin the model revision, seeds, optimization and context
limits. Training selects checkpoints by development NLL, including the initial
checkpoint. Evaluation fits temperatures on calibration data and then evaluates
fresh/regression tests. It also runs structural diagnostics and cache timing.
Reference results already recorded in the repository are retained; invoking
`openjev-model reference` makes new paid provider calls and is optional.

The original branch training took about 636 seconds on an M3 Pro with 36 GB RAM.
This is one measured run, not a minimum hardware requirement. CPU training is
slower. The code uses MPS when available and otherwise CPU. Floating-point,
backend and dependency differences can change training results; deterministic
split hashes are a stronger reproduction requirement than identical model bytes.

The matched three-path latency experiment additionally uses
`scripts/benchmark_v03_matched.py`. Its fixed workloads and raw repetitions are
in `reports/v03/matched_latency.json`. Do not generalize synthetic timing to a
production workload or compare it with a remote Jev service.

## Historical MiniLM experiment

```bash
uv run --no-sync openjev audit
uv run --no-sync python scripts/run_local.py --experiment public-reproduction
```

This uses the first prepared dataset and does not require provider synthesis.
The original augmentation arm used saved DeepSeek paraphrases that are excluded
from Git; its exact provider-trace replay is not distributed. Public-only training
remains reproducible. See `docs/EXPERIMENT_PLAN.md` and `reports/RESULTS.md`.

## Verification levels

1. CI: schema/metrics, deterministic rule cases, cache output/gradient equivalence,
   safe release extraction, source hygiene, package build and CLI help.
2. Prepared-data tests: split hashes, group isolation, holdout membership.
3. Full local integration: exported model inference, HTTP validation, and replay
   tests when their optional local artifacts exist.

CI skips the latter levels when their artifacts are absent. It does not silently
download multi-GB model weights or spend model-service credits.
