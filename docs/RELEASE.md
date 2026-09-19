# Release process

The initial public distribution is `IamBusy/OpenJev` on GitHub. Version v0.3.0 is
an experimental release with source archives, Python packages and the small
OpenJev-Branch adapter/head ZIP. Qwen base weights download separately from their
pinned upstream revision. No Hugging Face model/dataset publication is claimed.

Before a release:

1. Run lint, format checks, tests, source hygiene checks and package build.
2. Rebuild the frozen v0.3 splits in a clean workspace and compare their hashes.
3. Verify the exported head/adapter against the development-selected checkpoint.
4. Include the model card, LICENSE, attribution, results and file hashes in the
   model ZIP; update the pinned archive hash in `release_assets.json`.
5. Scan the source and built distributions for credentials and private paths.
6. Commit, tag and push the tested source; check public CI.
7. Publish an experimental GitHub Release and attach the model ZIP, wheel, source
   distribution and `SHA256SUMS`. Download through the public link and verify.

Do not bundle `.env`, model caches, raw provider traces, training runs or original
benchmark texts. Retain source-specific dataset terms. Report meaningful known
regressions in release notes. Keep historical result files identifiable; changes
to engineering after a training run do not retroactively change its provenance.

A future Hugging Face mirror requires an authorized account and a matching model
or dataset card. It must preserve exact hashes and the mixed source attribution.
