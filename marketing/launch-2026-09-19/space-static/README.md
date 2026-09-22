---
title: OpenJev Vision
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Live browser inference for compositional probability queries
models:
- IamBusy/OpenJev-Vision
datasets:
- IamBusy/OpenJev-Vision-Research-v0.1
tags:
- computer-vision
- uncertainty
- onnx
- openjev
---

# OpenJev-Vision: live browser demo

The published small synthetic-scene CNN runs **inside your browser** with ONNX
Runtime Web. This is live inference, not recorded output replay. It reads a
64×192 image and its supplied prior once, and returns a distribution over 64
possible scenes. Additional event queries reuse that distribution.

The two images come from the same fixed development episode. The supported
vocabulary is three positions, two colors, and two shapes. Event composition
uses explicit predicates; this demo is not a general-purpose visual assistant.

`model.onnx` is a float32 export of synthetic-joint seed17 with the published
temperature calibration. Model revision:
`8cf6cbd39a7dc72840c9b52810a3f672786566b0`.
Source commit: `4fa973e5212881d704def6818b0def190b094f12`.
See `provenance.json` for hashes and numerical comparison with the PyTorch model.

Code, original synthetic examples, and the served model are Apache-2.0.
ONNX Runtime Web 1.22.0 is loaded from jsDelivr under its MIT license.

The full research project includes separate DINOv2-based Pets/CLEVR-4
experiments, three-seed evaluations and negative findings. Those public-image
models are not the model served by this demo.

[Code and quick start](https://github.com/IamBusy/OpenJev-Vision) ·
[Weights](https://huggingface.co/IamBusy/OpenJev-Vision) ·
[Dataset](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) ·
[Results](https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md)

Independent project inspired by TypeSafe's Jev; not affiliated with TypeSafe.
