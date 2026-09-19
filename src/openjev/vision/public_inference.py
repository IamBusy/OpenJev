"""One visual encoding, one categorical posterior, many declared event queries."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .model import device_for
from .public_model import AttributeHead


def compile_public_event(spec, worlds):
    if not isinstance(spec, dict) or not spec:
        raise ValueError("Public event must be a nonempty mapping")
    if len(spec) == 1 and "not" in spec:
        return ~compile_public_event(spec["not"], worlds)
    for operator, combine in (("all", np.logical_and), ("any", np.logical_or)):
        if len(spec) == 1 and operator in spec:
            parts = spec[operator]
            if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
                raise ValueError("A composed event needs 1–32 children")
            return combine.reduce([compile_public_event(part, worlds) for part in parts])
    mask = np.ones(len(worlds), dtype=bool)
    for attribute, value in spec.items():
        if attribute not in worlds[0] or value not in {world[attribute] for world in worlds}:
            raise ValueError(f"Unknown ontology value for {attribute}")
        mask &= np.asarray([world[attribute] == value for world in worlds])
    return mask


def answer_public(posterior, questions, worlds):
    p = np.asarray(posterior, dtype=float)
    if p.shape != (len(worlds),) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid public posterior")
    p = p / p.sum()
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 256:
        raise ValueError("Provide 1–256 questions")
    result = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            result[name] = {
                "type": "noul",
                "noul": float(p @ compile_public_event(q["event"], worlds)),
            }
        elif q["type"] == "choice":
            criteria = q["criteria"]
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("Choice requires 2–255 alternatives")
            masks = np.stack([compile_public_event(e, worlds) for e in criteria.values()])
            if not np.all(masks.sum(0) == 1):
                raise ValueError("Choice candidates must partition the model's ontology")
            probabilities = masks @ p
            result[name] = {
                "type": "choice",
                "choice": list(criteria)[int(probabilities.argmax())],
                "probabilities": dict(zip(criteria, map(float, probabilities), strict=True)),
            }
        else:
            raise ValueError(
                "Public v0.1 supports noul/choice; synthetic count scores are separate"
            )
    return result


class PublicVisionModel:
    def __init__(self, checkpoint, backbone_path, ontology_path, device="auto"):
        from transformers import AutoImageProcessor, AutoModel

        self.checkpoint = Path(checkpoint)
        self.config = json.loads((self.checkpoint / "config.json").read_text())
        self.device = device_for(device)
        self.backbone = (
            AutoModel.from_pretrained(backbone_path, local_files_only=True).to(self.device).eval()
        )
        self.processor = AutoImageProcessor.from_pretrained(
            backbone_path, local_files_only=True, use_fast=False
        )
        self.head = (
            AttributeHead(
                self.config["dimensions"],
                self.config["variant"],
                self.config["feature_dim"],
                self.config["rank"],
            )
            .to(self.device)
            .eval()
        )
        self.head.load_state_dict(load_file(str(self.checkpoint / "head.safetensors")))
        self.temperature = json.loads((self.checkpoint / "calibration.json").read_text())[
            "temperature"
        ]
        ontology = json.loads(Path(ontology_path).read_text())
        if self.config["dataset"] == "pets":
            self.worlds = [
                {"breed": name, "species": ("cat", "dog")[ontology["species"][i]]}
                for i, name in enumerate(ontology["classes"])
            ]
        else:
            self.worlds = [
                {"color": color, "shape": shape}
                for color, shape in product(ontology["colors"], ontology["shapes"])
            ]

    @torch.inference_mode()
    def posterior(self, image):
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        features = self.backbone(pixel_values=pixels).last_hidden_state[:, 0]
        return self.head(features, self.temperature).exp().cpu().numpy()[0]

    def predict(self, image, questions):
        p = self.posterior(image)
        return {
            "model": f"OpenJev-Vision/{self.config['dataset']}/{self.config['variant']}",
            "model_version": "0.1.0",
            "answers": answer_public(p, questions, self.worlds),
            "posterior": p.tolist(),
            "worlds": self.worlds,
            "visual_encodings": 1,
            "scope": "Fixed declared ontology; query instructions are descriptive, event semantics are executable.",
        }
