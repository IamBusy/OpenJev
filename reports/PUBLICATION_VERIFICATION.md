# Public release verification — 2026-09-19

These checks describe the source publication work after the recorded v0.3
training experiment. Historical training/source manifests remain historical;
packaging and documentation changes do not change the trained weights.

- Full local suite: 31 passed, two upstream deprecation warnings.
- Extracted source distribution, offline and without local data/weights:
  24 passed, seven explicit integration skips, two upstream warnings.
- Fresh workspace downloaded all public data sources and rebuilt v0.1/v0.2/v0.3
  splits. All 17 comparable public JSONL files matched the original SHA-256
  hashes; the complete v0.3 manifest also matched byte for byte.
- The fresh workspace reused only the pinned Qwen base directory to avoid a
  redundant 1.2 GB download. No `.env`, private teacher traces, processed splits,
  training runs or pre-existing OpenJev checkpoint were copied into it.
- The public model ZIP was installed into that workspace using its pinned hash.
  Default CLI inference then worked offline and encoded the three-question
  example's state once. Head/adapter bytes match the selected training checkpoint.
- Source hygiene and exact known-credential scans passed. Built source and wheel
  inventories exclude credentials, caches and training runs.
- Lint, formatting, local documentation links and package builds passed.

Public Linux CPU CI passed on the initial source commit:
[Checks run 35434980378](https://github.com/IamBusy/OpenJev/actions/runs/35434980378).
Release download checks are recorded with the release on GitHub.
