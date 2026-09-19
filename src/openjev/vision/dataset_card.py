"""Source-specific attribution and machine-readable Hub configuration."""

import json
from pathlib import Path


def write_card(output, apache_license):
    output = Path(output)
    manifest = json.loads((output / "EXPORT_MANIFEST.json").read_text())
    rows = sum(item["rows"] for item in manifest["files"].values())
    yaml = [
        "---",
        "pretty_name: OpenJev Vision Research v0.1",
        "license: other",
        "license_name: source-specific-open-licenses",
        "license_link: https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1/blob/main/LICENSES.md",
        "language:",
        "- en",
        "task_categories:",
        "- image-classification",
        "- visual-question-answering",
        "size_categories:",
        "- 10K<n<100K",
        "tags:",
        "- openjev",
        "- probabilistic-reasoning",
        "- compositional-generalization",
        "- uncertainty",
        "- synthetic-data",
        "configs:",
    ]
    for name, splits in manifest["configs"].items():
        yaml.extend([f"- config_name: {name}", "  data_files:"])
        for split in splits:
            yaml.extend([f"  - split: {split}", f"    path: {name}-{split}.parquet"])
    yaml.append("---")
    body = f"""
# OpenJev Vision Research v0.1

**{rows:,} image records, with public provenance, original synthetic scenes,
and programmatically derived decision questions.**

This is an experimental research dataset for visual posterior learning and
compositional decisions, released with [OpenJev](https://github.com/IamBusy/OpenJev).
It is not a reproduction of TypeSafe's proprietary Jev model or training method.

## Three separate configurations

| Config | Images | What the labels mean | License |
|---|---:|---|---|
| synthetic | 8,192 | Exact posterior over 64 worlds under a specified noisy sensor and supplied prior | Apache-2.0 |
| pets | 2,960 | Original breed/species annotations of real photographs | CC BY-SA 4.0 |
| clevr4 | 1,680 | Original attributes of public rendered scenes; 20 color-shape combinations withheld from training | CC BY 4.0 |

Do not treat class labels in pets/clevr4 as ground-truth probability distributions.
Do not treat the synthetic posterior as a real-world calibration guarantee.
No closed-model teacher was used to assign any released label.

## Loading

~~~python
from datasets import load_dataset
import json

data = load_dataset("IamBusy/OpenJev-Vision-Research-v0.1", "synthetic")
example = data["train"][0]
image = example["image"]
questions = json.loads(example["questions_json"])
targets = json.loads(example["target_json"])
~~~

Choose pets or clevr4 for the public-image configurations.

## Schema

Each record contains a decoded image, unique id, group_id, source,
source_id, split, license, and three JSON strings:

- questions_json: typed questions, executable event semantics and derived targets.
- target_json: synthetic prior/posterior and latent-world annotation, or public class labels.
- provenance_json: generation parameters or original source hashes and transformation records.

Synthetic observed metadata and latent truth are supervision/oracle fields;
the vision models must not read them at inference. Model inputs are image pixels
and the explicitly supplied prior. Event execution uses a fixed, documented ontology.
Descriptive instructions are not evidence of learned free-form language understanding.

## Synthetic construction

Three fixed slots (left, center, right), each with blue/red color and circle/square
shape, define 64 worlds. A declared six-variable Ising prior is supplied.
Each attribute is flipped independently with probability 0.08 by a synthetic
sensor. Images render observed attributes; whole objects may be occluded.
The exact posterior is computed from observable sensor values, not set to the
hidden world's one-hot label.

Each episode has one occluded and one fully visible view of the same sensor
observation. Their split and appearance seed remain shared. The 36 Boolean
questions per image yield 294,912 derived targets, not independent samples.
Pixel texture is independent appearance noise. Training priors use local/chain
edges; test_topology changes dependency structure; test_appearance changes
the rendering palette. Finite world assignments can repeat across splits.

## Public source selection

**Oxford-IIIT Pet:** original data by Omkar M. Parkhi, Andrea Vedaldi, Andrew
Zisserman and C. V. Jawahar, Cats and Dogs, CVPR 2012.
[Official dataset and license](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Downloaded through timm/oxford-iiit-pet at revision
089695c834a7deb60505b7cc506672db1c31a6aa; original IDs, breed names,
species and splits were checked against the original Oxford annotation archive.
Per breed: 40 train, 10 dev, 10 calibration from official trainval; 20 official
test images for test. Original image owners retain copyright.

**CLEVR-4:** original data by Sagar Vaze, Andrea Vedaldi and Andrew Zisserman,
No Representation Rules Them All in Category Discovery, NeurIPS 2023.
[Official dataset and license](https://www.robots.ox.ac.uk/~vgg/data/clevr4/).
Selected from the official 10k v1 archive. Colors and shapes are sorted lexically.
Pairs for which (color_index + shape_index) modulo 5 equals zero are absent from
train/dev/calibration. For the other 80 pairs, select 12/2/2 original train images.
Test has four original validation images for each of all 100 pairs.
This is not the full official benchmark.

Selection uses a fixed SHA256 ordering of original IDs. Public images are converted
to RGB, resized so the longest side is at most 256 pixels using Lanczos, and saved
as lossless PNG. Original IDs and hashes remain in every row. Questions are newly
derived from existing annotations using documented templates; no new visual
attributes were hallucinated or inferred from a language model.

## Splits and leakage boundaries

| Config | Train | Dev | Calibration | Test |
|---|---:|---:|---:|---:|
| synthetic | 4,096 | 512 | 512 | 1,024 each: ID, topology, appearance |
| pets | 1,480 | 370 | 370 | 740 |
| clevr4 | 960 | 160 | 160 | 400 |

All derived views/questions for an original source image or synthetic episode stay
in one split. Exact selected-image duplicates across splits were checked.
No claim is made about eliminating all near duplicates or foundation-model
pretraining contamination. Synthetic and public results must be reported separately.

## Reproducibility

The three source manifests and EXPORT_MANIFEST.json provide source revisions,
selection rules and SHA256 checksums for the Parquet files. Code and protocols are
in OpenJev's visual research track. Evaluate compound probabilities and held-out
compositions, not just format validity or in-distribution top-1 accuracy.

## Limitations

Fixed small ontologies, synthetic sensor assumptions, and bounded public subsets
limit conclusions. CLEVR-4 objects within an image share color and shape, so it
does not establish arbitrary object-binding ability. Photo labels describe the
annotated pet, not every object in a scene. Some probabilities can be confidently
wrong. This dataset is not a safety certification or evidence of frontier capability.

## Licensing

**Licenses are source-specific.** Oxford-derived images and their adaptations
remain CC BY-SA 4.0. CLEVR-4-derived images remain CC BY 4.0. Original synthetic
scenes and original generator code are Apache-2.0. See LICENSES.md; do not
relabel the whole mixture Apache-2.0.
"""
    (output / "README.md").write_text("\n".join(yaml) + "\n" + body)
    (output / "LICENSES.md").write_text("""# Licenses and attribution

## Original synthetic configuration

Original generator, rendered images, exact posterior annotations and event queries:
Copyright 2026 OpenJev contributors. Apache License 2.0; see LICENSE-APACHE-2.0.txt.

## Oxford-IIIT Pet configuration

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
Authors: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar.
Publication: Cats and Dogs, IEEE CVPR 2012.
License: Creative Commons Attribution-ShareAlike 4.0 International:
https://creativecommons.org/licenses/by-sa/4.0/
Original image owners retain copyright. Original image IDs are preserved per record.
Changes: subset selection, RGB conversion, resize to longest side <=256, PNG encoding,
and annotation-derived typed questions. These adaptations retain CC BY-SA 4.0.

## CLEVR-4 configuration

Source: https://www.robots.ox.ac.uk/~vgg/data/clevr4/
Authors: Sagar Vaze, Andrea Vedaldi and Andrew Zisserman.
Publication: No Representation Rules Them All in Category Discovery, NeurIPS 2023.
License: Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/
Changes: subset selection, held-out-composition partition, RGB conversion, resize
to longest side <=256, PNG encoding, and annotation-derived typed questions.
Original image IDs and labels are retained; these adaptations remain CC BY 4.0.

No source author endorses this derived research dataset. Licenses apply separately
to their corresponding configurations; the mixture has no single permissive license.
""")
    (output / "LICENSE-APACHE-2.0.txt").write_text(Path(apache_license).read_text())
    (output / "LICENSE").write_text((output / "LICENSES.md").read_text())
