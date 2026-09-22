# OpenJev-Vision launch materials — 2026-09-19

Source, graphics and dated drafts for the visual probability research demo.

## Contents

| Directory | Contents |
| --- | --- |
| `space-static/` | Complete browser demo, HTML/CSS/JavaScript and a small ONNX runtime model |
| `space/` | Optional local Gradio demo with hash-verified downloads of the published model |
| `assets/` | Original covers, diagrams, fixed-example GIFs and a 42-second video |
| `tools/` | Media generation and PyTorch-to-ONNX export scripts |
| `copy/` | Dated publication drafts and a factual briefing for human authors |

[Live demo](https://huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo) ·
[Research code](https://github.com/IamBusy/OpenJev-Vision) ·
[Published weights](https://huggingface.co/IamBusy/OpenJev-Vision) ·
[Published dataset](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1)

## Run the static demo

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 \
  --directory marketing/launch-2026-09-19/space-static
```

Open `http://127.0.0.1:8000`. The browser downloads ONNX Runtime Web 1.22.0 from
jsDelivr and performs inference locally using the included synthetic-scene model.
The full model scope, licenses and provenance are in `space-static/README.md`.
Use an HTTP server; directly opening `index.html` from disk can block model fetches.

The Gradio version has its own [local instructions](space/README.md).
It downloads the selected CNN; it does not require Qwen or DINOv2.

## Regenerate graphics and the browser model

The graphic scripts were authored on macOS and use its Arial, Arial Unicode and
STHeiti fonts. Equivalent fonts must be configured before rendering on another OS.
The supplied PNG/SVG/GIF/MP4 files can be used without recreating the environment.

```bash
uv sync --frozen --extra dev --extra vision
uv run --no-sync python marketing/launch-2026-09-19/tools/build_media.py
uv run --no-sync openjev-vision download
uv run --frozen --extra vision --with onnx --with onnxruntime \
  python marketing/launch-2026-09-19/tools/export_browser_model.py
```

The export script compares ONNX outputs with the published PyTorch checkpoint
on two fixed development views and writes the model hash and maximum differences.
See `assets/provenance.json` and `space-static/provenance.json`. The recorded
source revision identifies the experiment used to create these archived assets;
formatting or documentation updates do not change the recorded measurements.
`MANIFEST.json` lists hashes of the distributable files in this directory.

## Research scope

The demo has three positions, two colors and two shapes, and uses an explicit
64-world prior. It supports declared event predicates, not arbitrary photos or
free-form scene understanding. Figures use fixed development examples and are
not evidence of full-benchmark performance. Public-image experiments and negative
results are documented separately in the main research reports.
