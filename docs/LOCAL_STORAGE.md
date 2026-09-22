# Local storage and recovery

Git contains the source, configuration, experiment reports, selected examples,
and public demo/media source. It is not a backup of the complete working directory.

Keep these separately backed up when preserving a research run:

- Native prepared training/evaluation data and source manifests.
- Training logs and all desired checkpoints, including unselected seeds/runs.
- Original model-service requests, responses and review records used for replay.
- Local configuration and credentials, stored privately.
- Unpublished operational notes and any original material excluded from Git.

The public vision dataset and selected models have their own Hugging Face
repositories. They do not include every local source cache or intermediate model.
See `reports/vision-v01/PUBLICATION.json` and the model/data guides for exact
published revisions and scope.

## Rebuild an environment

Removing the project's `.venv` does not remove source or trained weights. Restore
it from the repository root when local Python inference/training is needed:

```bash
uv sync --frozen --extra dev --extra qwen --extra vision
```

Do not commit virtual environments, downloaded base weights or local data caches.
The static browser demo can be served with a separate Python installation without
rebuilding the machine-learning environment.

## Restore pretrained backbones

For the Qwen text-decision model:

```bash
uv run --no-sync openjev-model download --base-only
```

For public-image vision inference, download the vision bundle and its backbone:

```bash
uv run --no-sync openjev-vision download --backbone
```

This places DINOv2 under `artifacts/openjev-vision-v0.1/backbone`. Pass that path
with `--backbone` to `openjev-vision public-predict`. Public training can recreate
its default backbone cache through `openjev-vision public-train`.

The historical MiniLM loader uses the pinned model identity in the checkpoint's
`config.json` when the local `backbone/` directory is absent, downloading it via
Sentence Transformers as needed. Its trained scoring head remains separate.

Synthetic vision CNN checkpoints already contain their trained network and need
no Qwen, MiniLM or DINOv2 base weights. Removing a pretrained backbone does not
mean deleting these trained CNNs, LoRA adapters or scoring heads.

Global package/model caches outside this checkout are shared resources and are
not included in a cleanup of this project directory.
