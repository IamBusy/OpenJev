# Live browser demo: one image encoding, several probability queries

Published: https://huggingface.co/IamBusy/OpenJev-Vision/discussions/1

The OpenJev-Vision live demo is now available:

https://huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo

![Two fixed development views and recorded model estimates](https://huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo/resolve/main/demo-en.gif)

The demo runs the published synthetic-joint seed-17 CNN inside your browser. Reveal more visual evidence, watch the estimates change, then ask another event question without rerunning the image encoder. No account or model-service key is needed.

The ONNX export includes the published temperature calibration. Its predictions were checked against the PyTorch model on both bundled views; the maximum absolute posterior difference was below 0.000001. The examples come from one fixed development episode.

The full release also includes separate DINOv2-based Pets and CLEVR-4 experiments, downloadable weights, data and three-seed evaluations. The failures are included: the direct joint predictor degrades on unseen dependencies, and independent attributes beat the tested joint and binding heads on unseen color/shape pairs.

**Scope:** fixed vocabularies and explicit event semantics. This is an experimental toolkit, not a general-purpose VLM. It is an independent project inspired by Jev and is not affiliated with TypeSafe.

- [Code and quick start](https://github.com/IamBusy/OpenJev-Vision)
- [Data](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1)
- [Full results and limitations](https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md)

I’d especially welcome reproducible failure cases and suggestions for evaluating generalization to new event structures.
