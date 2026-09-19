"""Frozen public visual features and transparent finite-distribution readouts."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from scipy.optimize import minimize_scalar
from torch import nn
from torch.nn import functional as F

from .data import sha256, write_json
from .model import device_for
from .train import hardware, seed_all


def records_for(data_dir, dataset):
    if dataset not in {"pets", "clevr4"}:
        raise ValueError("dataset must be pets or clevr4")
    return [
        json.loads(line) for line in (Path(data_dir) / f"{dataset}.jsonl").read_text().splitlines()
    ]


def cache_features(data_dir, dataset, backbone_path, output, config, device="auto"):
    from transformers import AutoImageProcessor, AutoModel

    output = Path(output)
    rows = records_for(data_dir, dataset)
    signature = {
        "records_sha256": sha256(Path(data_dir) / f"{dataset}.jsonl"),
        "backbone": config["backbone"],
        "revision": config["revision"],
        "weights_sha256": sha256(Path(backbone_path) / "model.safetensors"),
        "processor_sha256": sha256(Path(backbone_path) / "preprocessor_config.json"),
    }
    signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if str(cached["signature"]) == signature_hash:
                return output
        raise ValueError("Existing feature cache does not match image/model signature")
    dev = device_for(device)
    model = AutoModel.from_pretrained(backbone_path, local_files_only=True).to(dev).eval()
    processor = AutoImageProcessor.from_pretrained(
        backbone_path, local_files_only=True, use_fast=False
    )
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(rows), 32):
        images = []
        for row in rows[start : start + 32]:
            path = Path(data_dir) / row["image"]
            if sha256(path) != row["image_sha256"]:
                raise ValueError("Image changed since data preparation")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(dev)
        with torch.inference_mode():
            features = model(pixel_values=pixels).last_hidden_state[:, 0]
        vectors.append(features.cpu().numpy())
        if start % 320 == 0:
            print(
                json.dumps(
                    {"features": dataset, "completed": start + len(images), "total": len(rows)}
                ),
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vectors=np.concatenate(vectors),
        ids=np.asarray([r["source_id"] for r in rows]),
        signature=signature_hash,
    )
    write_json(
        output.with_suffix(".json"),
        {
            **signature,
            "signature": signature_hash,
            "images": len(rows),
            "seconds": time.perf_counter() - started,
            "device": str(dev),
            "precision": "float32",
            "hardware": hardware(),
            "feature": "DINOv2 final normalized CLS token, frozen",
        },
    )
    return output


class AttributeHead(nn.Module):
    def __init__(self, dimensions, variant="joint", feature_dim=384, rank=4):
        super().__init__()
        self.dimensions = tuple(dimensions)
        self.variant = variant
        self.rank = rank
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("scale", torch.ones(feature_dim))
        if len(dimensions) == 1 or variant == "joint":
            self.output = nn.Linear(feature_dim, int(np.prod(dimensions)))
        elif variant in {"independent", "binding"}:
            self.output = nn.Linear(feature_dim, sum(dimensions))
            if variant == "binding":
                self.left = nn.Parameter(torch.randn(dimensions[0], rank) * 0.1)
                self.right = nn.Parameter(torch.randn(dimensions[1], rank) * 0.1)
                self.context = nn.Linear(feature_dim, rank)
                nn.init.zeros_(self.context.weight)
                nn.init.zeros_(self.context.bias)
        else:
            raise ValueError("Unsupported attribute head")

    def logits(self, features):
        x = (features - self.mean) / self.scale
        unary = self.output(x)
        if len(self.dimensions) == 1 or self.variant == "joint":
            return unary
        left, right = unary.split(self.dimensions, dim=-1)
        joint = left[:, :, None] + right[:, None, :]
        if self.variant == "binding":
            joint = joint + torch.einsum(
                "br,cr,sr->bcs", self.context(x).tanh(), self.left, self.right
            )
        return joint.flatten(1)

    def forward(self, features, temperature=1.0):
        return F.log_softmax(self.logits(features) / temperature, dim=-1)


def classification_metrics(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.maximum(p, 1e-15)
    p /= p.sum(-1, keepdims=True)
    y = np.asarray(labels)
    correct = p.argmax(-1) == y
    conf = p.max(-1)
    bins, ece = [], 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        chosen = (conf >= lower) & (conf <= upper if upper == 1 else conf < upper)
        n = int(chosen.sum())
        if n:
            accuracy, confidence = float(correct[chosen].mean()), float(conf[chosen].mean())
            ece += n / len(y) * abs(accuracy - confidence)
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "n": n,
                    "accuracy": accuracy,
                    "confidence": confidence,
                }
            )
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(-1).mean()),
        "ece": float(ece),
        "reliability": bins,
    }


def train_public(data_dir, dataset, feature_path, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Public run exists: {output}")
    output.mkdir(parents=True)
    rows = records_for(data_dir, dataset)
    manifest = json.loads((Path(data_dir) / f"{dataset}_manifest.json").read_text())
    with np.load(feature_path, allow_pickle=False) as cache:
        if cache["ids"].tolist() != [r["source_id"] for r in rows]:
            raise ValueError("Features do not align with records")
        features = torch.tensor(cache["vectors"], dtype=torch.float32)
    labels = torch.tensor([r["labels"]["breed" if dataset == "pets" else "joint"] for r in rows])
    split = np.asarray([r["split"] for r in rows])
    subsets = {
        s: torch.tensor(np.flatnonzero(split == s)) for s in ("train", "dev", "calibration", "test")
    }
    dims = (
        [len(manifest["classes"])]
        if dataset == "pets"
        else [len(manifest["colors"]), len(manifest["shapes"])]
    )
    variants = ["joint"] if dataset == "pets" else ["joint", "independent", "binding"]
    source = Path(__file__)
    report = {
        "dataset": dataset,
        "config": config,
        "features_sha256": sha256(feature_path),
        "source_sha256": sha256(source),
        "data_manifest_sha256": sha256(Path(data_dir) / f"{dataset}_manifest.json"),
        "hardware": hardware(),
        "head_device": "cpu",
        "results": {},
        "limitations": [
            "Frozen DINOv2 with small supervised readouts; not a general VLM.",
            "Pretraining overlap is unknown; official splits apply to head training only.",
            "Uncertainty targets on public images are observed class labels, not oracle distributions.",
        ],
    }
    for seed in config["seeds"]:
        for variant in variants:
            seed_all(seed)
            model = AttributeHead(dims, variant, features.shape[1], config["binding_rank"])
            with torch.no_grad():
                model.mean.copy_(features[subsets["train"]].mean(0))
                model.scale.copy_(features[subsets["train"]].std(0).clamp_min(1e-4))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
            )
            best, history = float("inf"), []
            rng = np.random.default_rng(seed)
            path = output / f"{variant}-{seed}"
            path.mkdir()
            started = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                order = rng.permutation(subsets["train"].numpy())
                for start in range(0, len(order), config["batch_size"]):
                    indices = torch.tensor(order[start : start + config["batch_size"]])
                    loss = F.nll_loss(model(features[indices]), labels[indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    dev_nll = float(
                        F.nll_loss(model(features[subsets["dev"]]), labels[subsets["dev"]])
                    )
                history.append({"epoch": epoch, "dev_nll": dev_nll})
                if dev_nll < best:
                    best = dev_nll
                    save_file(
                        {k: v.contiguous() for k, v in model.state_dict().items()},
                        str(path / "head.safetensors"),
                    )
                    write_json(
                        path / "config.json",
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "dimensions": dims,
                            "rank": config["binding_rank"],
                            "feature_dim": features.shape[1],
                            "seed": seed,
                            "selected_epoch": epoch,
                            "dev_nll": dev_nll,
                            "selection_split": "dev",
                            "backbone": config["backbone"],
                            "revision": config["revision"],
                        },
                    )
            model.load_state_dict(load_file(str(path / "head.safetensors")))
            with torch.inference_mode():
                calibration_logits = model.logits(features[subsets["calibration"]]).double()
            calibration_labels = labels[subsets["calibration"]]
            fit = minimize_scalar(
                lambda logt: float(
                    F.cross_entropy(calibration_logits / np.exp(logt), calibration_labels)
                ),
                bounds=(np.log(0.2), np.log(5)),
                method="bounded",
            )
            temperature = float(np.exp(fit.x))
            write_json(
                path / "calibration.json", {"temperature": temperature, "fit_split": "calibration"}
            )
            write_json(path / "history.json", history)
            with torch.inference_mode():
                raw = model(features[subsets["test"]]).exp().numpy()
                p = model(features[subsets["test"]], temperature).exp().numpy()
            y = labels[subsets["test"]].numpy()
            result = {
                "raw": classification_metrics(raw, y),
                "calibrated": classification_metrics(p, y),
                "temperature": temperature,
                "best_dev_nll": best,
                "training_seconds": time.perf_counter() - started,
                "head_parameters": sum(p.numel() for p in model.parameters()),
                "weights_sha256": sha256(path / "head.safetensors"),
            }
            if dataset == "pets":
                species = np.asarray(manifest["species"])
                species_p = np.stack([p[:, species == i].sum(-1) for i in (0, 1)], axis=-1)
                result["species"] = classification_metrics(species_p, species[y])
            else:
                table = p.reshape(-1, *dims)
                result["color"] = classification_metrics(table.sum(2), y // dims[1])
                result["shape"] = classification_metrics(table.sum(1), y % dims[1])
                test_rows = [rows[i] for i in subsets["test"].tolist()]
                unseen = np.asarray([r["unseen_composition"] for r in test_rows])
                result["seen"] = classification_metrics(p[~unseen], y[~unseen])
                result["unseen"] = classification_metrics(p[unseen], y[unseen])
            report["results"][f"{variant}-{seed}"] = result
            np.savez_compressed(
                path / "test_predictions.npz",
                raw=raw,
                calibrated=p,
                labels=y,
                ids=np.asarray([rows[i]["source_id"] for i in subsets["test"].tolist()]),
            )
            write_json(output / "results.json", report)
            print(
                json.dumps(
                    {
                        "trained": dataset,
                        "variant": variant,
                        "seed": seed,
                        "test_accuracy": result["calibrated"]["accuracy"],
                    }
                ),
                flush=True,
            )
    return report
