OpenJev-Vision: one image encoding, several probability queries.

I’m sharing an open research toolkit that keeps a distribution over a defined scene vocabulary and answers compositional questions from it.

The live demo runs a small trained CNN in your browser on two fixed development views. Reveal more visual evidence, watch the estimates change, then ask another event question without rerunning the image encoder.

The release also includes separate DINOv2-based Pets and CLEVR-4 experiments, downloadable weights, data and three-seed evaluations. The failures are included: the direct joint predictor degrades on unseen dependencies, and independent attributes beat the tested joint and binding heads on unseen color/shape pairs.

Scope: fixed vocabularies and explicit event semantics. This is an experimental toolkit, not a general-purpose VLM. It is an independent project inspired by Jev and is not affiliated with TypeSafe.

Try the live demo: https://huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo
Weights: https://huggingface.co/IamBusy/OpenJev-Vision
Data: https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1
Code and results: https://github.com/IamBusy/OpenJev-Vision

I’d especially welcome reproducible failure cases and suggestions for evaluating generalization to new event structures.
