---
title: OpenJev Vision
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: One image encoding, several compositional probability queries
models:
- IamBusy/OpenJev-Vision
datasets:
- IamBusy/OpenJev-Vision-Research-v0.1
tags:
- computer-vision
- uncertainty
- probabilistic-reasoning
- openjev
---

# OpenJev-Vision: live synthetic-scene demo

Run the published small CNN on two fixed development views. It reads a 64×192
synthetic image and its supplied 64-world prior, then returns a distribution.
Compose additional event queries without rerunning the image encoder.

This is live CPU inference, not a recorded output replay. The model is downloaded
from `IamBusy/OpenJev-Vision` at the pinned revision
`8cf6cbd39a7dc72840c9b52810a3f672786566b0`; hashes are checked on load.

The copied model, query, world, data and render modules come from source commit
`4fa973e5212881d704def6818b0def190b094f12`, with local import paths adjusted.
The examples originate in `examples/vision` at that same commit.

Scope: three object positions, two colors, two shapes, explicit event predicates.
This is not a general visual assistant. Probabilistic consistency does not establish
perceptual correctness or calibration on arbitrary real images.

The project includes separate experiments with DINOv2 features, Pets and CLEVR-4,
alongside three-seed evaluations and negative findings. Those experiments are not
the model served by this Space.

[Code](https://github.com/IamBusy/OpenJev-Vision) ·
[Weights](https://huggingface.co/IamBusy/OpenJev-Vision) ·
[Data](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) ·
[Results](https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md)

Independent project inspired by TypeSafe's Jev; not affiliated with TypeSafe.

## Run locally

```sh
python -m pip install -r requirements.txt gradio==5.49.1
python app.py
```

Code, original synthetic examples, and the served synthetic model: Apache-2.0.
