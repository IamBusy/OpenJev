"""Portable experimental weights, hashes, and optional Hub download."""

import json
import shutil
from pathlib import Path

from .data import sha256, write_json

DEFAULT_REPO = "IamBusy/OpenJev-Vision"
DEFAULT_REVISION = "8cf6cbd39a7dc72840c9b52810a3f672786566b0"


def export_models(runs, public_data, output, license_path):
    runs, output = Path(runs), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Model export directory is not empty")
    output.mkdir(parents=True)
    models = {}
    for variant in ("joint", "independent", "evidence"):
        key = f"synthetic-{variant}"
        path = runs / f"{variant}-17" / "checkpoint"
        for source_name, suffix in (
            ("model.safetensors", ".safetensors"),
            ("config.json", ".config.json"),
            ("calibration.json", ".calibration.json"),
        ):
            shutil.copyfile(path / source_name, output / (key + suffix))
        models[key] = {"kind": "synthetic", "seed": 17}
    for dataset, variants in (
        ("pets", ("joint",)),
        ("clevr4", ("joint", "independent", "binding")),
    ):
        for variant in variants:
            key = f"{dataset}-{variant}"
            path = runs / f"public-{dataset}" / f"{variant}-17"
            for source_name, suffix in (
                ("head.safetensors", ".safetensors"),
                ("config.json", ".config.json"),
                ("calibration.json", ".calibration.json"),
            ):
                shutil.copyfile(path / source_name, output / (key + suffix))
            models[key] = {"kind": "public", "dataset": dataset, "seed": 17}
        shutil.copyfile(
            Path(public_data) / f"{dataset}_manifest.json", output / f"{dataset}.ontology.json"
        )
    write_json(
        output / "model_index.json",
        {
            "format_version": 1,
            "models": models,
            "release_rule": "Fixed seed 17 for inference bundles; all three seeds remain reported.",
            "backbone": "facebook/dinov2-small",
            "backbone_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        },
    )
    shutil.copyfile(license_path, output / "LICENSE-APACHE-2.0.txt")
    notices = """# Released weight licenses

Original synthetic-scene CNN weights: Apache-2.0, see LICENSE-APACHE-2.0.txt.
Oxford-IIIT Pet readout weights: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
CLEVR-4 readout weights: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Copyright 2026 OpenJev contributors for these trained readouts.

Training image sources and authors:
Oxford-IIIT Pet, Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, C. V. Jawahar:
https://www.robots.ox.ac.uk/~vgg/data/pets/
CLEVR-4, Sagar Vaze, Andrea Vedaldi, Andrew Zisserman:
https://www.robots.ox.ac.uk/~vgg/data/clevr4/
Original data retain their source-specific licenses; original image owners retain copyright.
The separate DINOv2 backbone is not redistributed here and retains its own Apache-2.0 license.
"""
    (output / "LICENSE").write_text(notices)
    (output / "LICENSES.md").write_text(notices)
    (output / "README.md").write_text("""---
license: other
license_name: source-specific-research-licenses
license_link: https://huggingface.co/IamBusy/OpenJev-Vision/blob/main/LICENSES.md
library_name: openjev
tags:
- openjev
- image-classification
- uncertainty
- research
datasets:
- IamBusy/OpenJev-Vision-Research-v0.1
---
# OpenJev Vision v0.1 — experimental research weights

Small trained readouts for [OpenJev's visual research track](https://github.com/IamBusy/OpenJev-Vision).
The [dataset](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1)
contains original synthetic scenes and attributed public-image subsets.

## Contents

Seven fixed-seed-17 checkpoints: three synthetic-scene CNNs (joint, independent,
and learned visual evidence with exact finite fusion), one Pets breed head,
and three CLEVR-4 color/shape heads (joint, independent and low-rank binding).
Every checkpoint was selected by development NLL and calibrated on a disjoint
calibration split. All three training seeds are reported in the code repository.

Synthetic checkpoints include their complete small CNN. Public-image checkpoints
contain only trained heads and train-fitted feature normalization. They require
the separately downloaded, frozen facebook/dinov2-small backbone at revision
ed25f3a31f01632728cabb09d1542f84ab7b0056, and its exact image processor.
They are not Transformers AutoModel checkpoints or general-purpose VLMs.

## Load

Install OpenJev with its vision extra, then:

~~~python
from openjev.vision.hub import download_models
paths = download_models("artifacts/openjev-vision-v0.1")
~~~

The returned directories can be used with SceneModel or PublicVisionModel.
See the repository's visual research guide for complete image/question examples.
Model file SHA256 hashes are recorded in FILE_MANIFEST.json.

## Measured behavior and limitations

The learned joint synthetic model improves compound probabilities over its own
factorized marginals in distribution, but its posterior prediction degrades
substantially on unseen dependency topologies. The learned-evidence/exact-fusion
control is much stronger in the known simulator; it also has privileged
observation-category training supervision.

On the 740-image balanced Pets subset, the seed-17 frozen-feature breed head
scores 93.24% accuracy; this is not the official full-dataset benchmark.
On 80 held-out color/shape-composition images in CLEVR-4, seed-17 joint,
independent and binding heads score 0%, 63.75% and 56.25%, respectively.
The low-rank binding head does not outperform the independent baseline.
These negative results are part of the release.

Query composition is exact probability arithmetic over a fixed ontology.
Free-form language understanding, general scene grounding, and arbitrary
real-world uncertainty calibration have not been established.
Synthetic inputs are 64x192 three-slot images with a supplied 64-world prior.
Pets/CLEVR-4 heads operate on their separate declared taxonomies.
DINOv2 pretraining overlap with the public datasets cannot be excluded.

## Training and licenses

Synthetic CNNs use original Apache-2.0 rendered data and exact observation-model
targets. Public heads use resized Oxford-IIIT Pet (CC BY-SA 4.0) and CLEVR-4
(CC BY 4.0) subsets, with attribution and license notices in the dataset card.
Original source data retain their respective licenses. We distribute the
original synthetic CNN weights under Apache-2.0, the Pets heads under CC BY-SA 4.0,
and the CLEVR-4 heads under CC BY 4.0. See LICENSES.md. No source images or
third-party backbone weights are included. DINOv2's own license applies to its
separate backbone. Do not use this model bundle to relabel the source data.
""")
    write_json(
        output / "FILE_MANIFEST.json",
        {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    )
    return output


def download_models(output, repo_id=DEFAULT_REPO, revision=None, *, backbone=False):
    from huggingface_hub import snapshot_download

    output = Path(output)
    if revision is None and repo_id == DEFAULT_REPO:
        revision = DEFAULT_REVISION
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.md", "*.txt", "LICENSE"],
        )
    )
    manifest = json.loads((snapshot / "FILE_MANIFEST.json").read_text())
    for name, expected in manifest.items():
        if Path(name).name != name:
            raise ValueError("Unsafe model manifest path")
        if sha256(snapshot / name) != expected:
            raise ValueError(f"Model bundle hash mismatch: {name}")
    index = json.loads((snapshot / "model_index.json").read_text())
    if index.get("format_version") != 1:
        raise ValueError("Unsupported visual model bundle format")
    paths = {}
    for key, entry in index["models"].items():
        if Path(key).name != key:
            raise ValueError("Unsafe model key")
        if entry["kind"] not in {"synthetic", "public"}:
            raise ValueError("Unsupported visual model kind")
        destination = output / key
        destination.mkdir(parents=True, exist_ok=True)
        for suffix, filename in (
            (
                ".safetensors",
                "model.safetensors" if entry["kind"] == "synthetic" else "head.safetensors",
            ),
            (".config.json", "config.json"),
            (".calibration.json", "calibration.json"),
        ):
            source_name = key + suffix
            if source_name not in manifest:
                raise ValueError("Unverified checkpoint component")
            target = destination / filename
            if target.exists() and sha256(target) != manifest[source_name]:
                raise FileExistsError(f"Refusing to replace a different checkpoint file: {target}")
            shutil.copyfile(snapshot / source_name, target)
        if entry["kind"] == "public":
            if entry["dataset"] not in {"pets", "clevr4"}:
                raise ValueError("Unsupported public ontology")
            shutil.copyfile(
                snapshot / f"{entry['dataset']}.ontology.json", destination / "ontology.json"
            )
        paths[key] = str(destination)
    if backbone:
        path = snapshot_download(
            index["backbone"],
            revision=index["backbone_revision"],
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "README.md",
            ],
            local_dir=output / "backbone",
        )
        paths["backbone"] = str(path)
    return paths
