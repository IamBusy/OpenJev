# OpenJev visual posterior research

[中文](VISION.zh-CN.md) ·
[Data](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) ·
[Weights](https://huggingface.co/IamBusy/OpenJev-Vision) ·
[Results](../reports/vision-v01/RESULTS.md)

**Encode an image once, retain a distribution, ask several compositional questions.**

This is OpenJev's experimental multimodal research track. v0.1 establishes data,
reference posteriors, trainable visual models, strong controls and reproducible
negative results. It does not claim to have solved general multimodal reasoning
or introduced a novel state-of-the-art architecture.

![A shared visual posterior](../reports/vision-v01/demo.png)

## Two complementary experiments

| Track | Input and model | What it tests |
|---|---|---|
| Controlled scenes | 64×192 pixels + supplied 64-world prior; small trained CNN | Retaining correlations, exact probability targets, occlusion, unseen dependencies |
| Public images | Frozen DINOv2-small + trained readouts | Real photographs, color/shape composition, shared categorical probabilities |

The synthetic CNN is not a model for natural photos. The public heads have fixed
Pets/CLEVR-4 ontologies. Query execution is symbolic; controlled English is a
documented convenience parser, not learned general language understanding.

## Quick start

Use a fresh checkout of the visual research branch until it is merged:

~~~bash
git clone --branch research/vision-posterior https://github.com/IamBusy/OpenJev.git
cd OpenJev
uv sync --frozen --extra vision --extra dev
uv run --no-sync openjev-vision download
uv run --no-sync openjev-vision predict \
  --checkpoint artifacts/openjev-vision-v0.1/synthetic-joint \
  --image examples/vision/scene-0.png \
  --prior examples/vision/scene-0-prior.json \
  --questions examples/vision/questions.json
~~~

The bundled examples are a fixed development episode, not cherry-picked test
successes. Compare scene-0 (occluded) with scene-1 (more visible evidence).
The download verifies the bundle's file hashes and uses safetensors.

~~~python
from PIL import Image
import json
from openjev.vision.model import SceneModel
from openjev.vision.query import answer_questions

model = SceneModel("artifacts/openjev-vision-v0.1/synthetic-joint")
prior = json.load(open("examples/vision/scene-0-prior.json"))
p = model.posterior(Image.open("examples/vision/scene-0.png"), prior)
answers = answer_questions(p, {
    "red": "Is the left object red?",
    "both": {"type": "noul", "event": "red(left) and square(right)"},
    "count": "How many objects are red?",
})
~~~

One posterior supports all questions. Choice events must form a mutually
exclusive, exhaustive partition; overlapping alternatives fail explicitly.
Event expressions are parsed with a restricted AST, never executed as Python.

## Public-image inference

~~~bash
uv run --no-sync openjev-vision download --backbone
~~~

This additionally downloads the pinned DINOv2-small backbone (about 88 MB).

~~~python
from PIL import Image
from openjev.vision.public_inference import PublicVisionModel

root = "artifacts/openjev-vision-v0.1"
model = PublicVisionModel(
    f"{root}/pets-joint",
    f"{root}/backbone",
    f"{root}/pets-joint/ontology.json",
)
result = model.predict(Image.open("your-pet-photo.jpg"), {
    "cat": {"type": "noul", "event": {"species": "cat"}},
    "breed": {"type": "noul", "event": {"breed": "persian"}},
    "species": {
        "type": "choice",
        "criteria": {"cat": {"species": "cat"}, "dog": {"species": "dog"}},
    },
})
~~~

For CLEVR-4, choose clevr4-independent, clevr4-binding or clevr4-joint and its
matching ontology. Events use color/shape values from the ontology, for example
an event with color=red and shape=cube. The breed/species or color/shape answers
are derived from one shared categorical distribution. The public head does not
recognize every scene object or support arbitrary new labels.

## Reproduce the experiments

Run from the checkout. Commands reject existing experiment output directories,
so earlier runs are not silently overwritten.

~~~bash
uv run --no-sync openjev-vision generate
uv run --no-sync openjev-vision train
uv run --no-sync openjev-vision evaluate
uv run --no-sync openjev-vision benchmark
uv run --no-sync openjev-vision prepare-public
uv run --no-sync openjev-vision train-public --dataset pets
uv run --no-sync openjev-vision train-public --dataset clevr4
uv run --no-sync python scripts/report_vision.py
uv run --no-sync openjev-vision export-data
~~~

Public preparation fetches a pinned Pet mirror, official annotations and selected
CLEVR-4 ZIP members; it does not need the complete 3.8 GB CLEVR archive.
Allow several GB of working disk space. Hugging Face's public sources require no
model-service key. MPS is the measured image-training device; CPU and CUDA code
paths are available but their performance was not benchmarked in this release.
Run timing separately from other GPU workloads.

The full synthetic experiment uses 3 variants × 3 seeds. The public experiment
uses 1 Pet readout and 3 CLEVR-4 readouts × 3 seeds. The fixed seed-17 bundles
are for inference convenience; the results report every seed.

## What the first experiment established

- Keeping the learned joint model's correlations improves compound probabilities
  in distribution compared with discarding those same correlations.
- The direct posterior predictor fails to generalize reliably to unseen
  dependency structures.
- A learned visual observation model plus the known observation process and
  supplied prior is a strong control, with different training supervision.
- A flat 100-way color/shape classifier fails on withheld pair labels.
  Independent attributes generalize better; the low-rank binding head does not
  beat that baseline.

These findings motivate structure-learning research. They do not prove that
finite enumeration, a probability head, or a bilinear interaction is new.

## Research roadmap

1. Learn scene dependencies while preserving query meaning across refinements.
2. Preserve useful posterior information under a visual-memory budget.
3. Replace fixed slots and taxonomies with learned grounding and variable objects.
4. Extend to time, distinguishing new evidence from correlated repeated views.

These items are research proposals, not delivered capabilities. The current
release makes their baselines and failure cases concrete.

## Validation and provenance

[Synthetic protocol](VISION_PROTOCOL.md), [public protocol](VISION_PUBLIC_PROTOCOL.md),
[data provenance](VISION_DATA.md), [full results](../reports/vision-v01/RESULTS.md).
Model selection uses development NLL, calibration uses a separate split, and
tests remain distinct. Reused source images/paired views stay grouped.
Probability identities are mathematical properties of the represented distribution;
they do not certify correct perception or calibrated probabilities on new domains.
