"""Frozen-test evaluation, matched marginal controls, and episode bootstrap."""

import json
import time
from pathlib import Path

import numpy as np

from .data import load_split, sha256, write_json
from .model import SceneModel
from .query import answer_questions, evaluation_queries
from .train import hardware, predict_arrays
from .world import factorize


def measure(p, target, truth, matrix, categories):
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-15)
    p /= p.sum(-1, keepdims=True)
    target = np.asarray(target, dtype=np.float64)
    target /= target.sum(-1, keepdims=True)
    divergence = np.where(
        target > 0, target * (np.log(np.maximum(target, 1e-30)) - np.log(p)), 0
    ).sum(-1)
    estimates, exact = p @ matrix.T, target @ matrix.T
    errors = (estimates - exact) ** 2
    compound = np.asarray(categories) == "compound"
    outcomes = matrix[:, truth].T
    return {
        "n": len(p),
        "posterior_kl": float(divergence.mean()),
        "unary_probability_mse": float(errors[:, ~compound].mean()),
        "compound_probability_mse": float(errors[:, compound].mean()),
        "event_brier_observed": float(((estimates - outcomes) ** 2).mean()),
        "world_nll_observed": float(-np.log(p[np.arange(len(p)), truth]).mean()),
        "world_accuracy": float((p.argmax(-1) == truth).mean()),
    }, errors[:, compound].mean(-1)


def episode_bootstrap(differences, groups, seed=9001, repeats=2000):
    identifiers, inverse = np.unique(groups, return_inverse=True)
    means = np.bincount(inverse, weights=differences) / np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), size=(repeats, len(means)))].mean(-1)
    return {
        "episodes": len(identifiers),
        "difference": float(means.mean()),
        "bootstrap_95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "direction": "factorized minus joint probability MSE; positive favors joint",
        "note": "Resampling episodes, retaining their views and queries; not seed uncertainty.",
    }


def evaluate(data_dir, runs_dir, output, config, device="auto"):
    output = Path(output)
    if (output / "metrics.json").exists():
        raise FileExistsError("Evaluation report already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    specs, matrix = evaluation_queries()
    categories = [kind for kind, _ in specs]
    report = {
        "protocol": "docs/VISION_PROTOCOL.md",
        "queries": [{"family": kind, "event": event} for kind, event in specs],
        "hardware": hardware(),
        "seeds": config["training_seeds"],
        "data_manifest_sha256": sha256(Path(data_dir) / "manifest.json"),
        "groups": {},
        "matched_ablation": {},
        "limitations": [
            "Controlled six-bit scenes, fixed slots and exact supplied priors.",
            "Query semantics are symbolic; no general language understanding.",
            "Coherence is exact finite probability arithmetic, not learned factual correctness.",
            "Evidence model receives observation labels during training.",
            "This study does not establish performance on natural images or Jev.",
        ],
    }
    for split in ("test_id", "test_topology", "test_appearance"):
        data = load_split(data_dir, split)
        masks = {"all": np.ones(len(data["images"]), dtype=bool)}
        masks.update({f"visible_{i}": data["visibility"] == i for i in range(4)})
        for name, p in (
            ("prior_only", data["priors"]),
            ("oracle", data["targets"]),
            ("oracle_factorized", factorize(data["targets"])),
        ):
            report["groups"][f"{split}/{name}"] = {
                key: measure(
                    p[mask], data["targets"][mask], data["truth"][mask], matrix, categories
                )[0]
                for key, mask in masks.items()
                if mask.any()
            }
        for seed in config["training_seeds"]:
            for variant in config["variants"]:
                path = Path(runs_dir) / f"{variant}-{seed}" / "checkpoint"
                predictor = SceneModel(path, device)
                raw, logp = predict_arrays(predictor.model, data, config["batch_size"])
                _, calibrated = predict_arrays(
                    predictor.model, data, config["batch_size"], predictor.temperature
                )
                distributions = {"raw": np.exp(logp), "calibrated": np.exp(calibrated)}
                if variant == "joint":
                    distributions["factorized"] = factorize(distributions["calibrated"])
                per_image = {}
                for mode, p in distributions.items():
                    key = f"{split}/{variant}-{seed}/{mode}"
                    report["groups"][key] = {
                        group: measure(
                            p[mask], data["targets"][mask], data["truth"][mask], matrix, categories
                        )[0]
                        for group, mask in masks.items()
                        if mask.any()
                    }
                    _, per_image[mode] = measure(
                        p, data["targets"], data["truth"], matrix, categories
                    )
                if variant == "joint":
                    report["matched_ablation"][f"{split}/seed-{seed}"] = episode_bootstrap(
                        per_image["factorized"] - per_image["calibrated"], data["groups"]
                    )
                np.savez_compressed(
                    output / f"{split}-{variant}-{seed}.npz",
                    raw_outputs=raw,
                    **distributions,
                    groups=data["groups"],
                    truth=data["truth"],
                )
                print(
                    json.dumps({"evaluated": split, "variant": variant, "seed": seed}), flush=True
                )
    write_json(output / "metrics.json", report)
    return report


def benchmark(checkpoint, data_dir, output, device="auto"):
    model = SceneModel(checkpoint, device)
    example = load_split(data_dir, "dev")
    image, prior = example["images"][0], example["priors"][0]
    specs, _ = evaluation_queries()
    results = []
    for count in (1, 8, 32):
        questions = {
            f"q{i}": {"type": "noul", "event": specs[i % len(specs)][1]} for i in range(count)
        }
        timings = {}
        predictions = {}
        for mode in ("shared", "reencoded"):
            elapsed = []
            for iteration in range(23):
                started = time.perf_counter()
                if mode == "shared":
                    p = model.posterior(image, prior)
                    result = answer_questions(p, questions)
                else:
                    result = {}
                    for key, q in questions.items():
                        result.update(answer_questions(model.posterior(image, prior), {key: q}))
                duration = time.perf_counter() - started
                if iteration >= 3:
                    elapsed.append(duration * 1000)
            predictions[mode] = result
            timings[mode] = {
                "p50_ms": float(np.median(elapsed)),
                "p95_ms": float(np.quantile(elapsed, 0.95)),
                "raw_ms": elapsed,
            }
        error = max(
            abs(predictions["shared"][k]["noul"] - predictions["reencoded"][k]["noul"])
            for k in questions
        )
        if error > 1e-6:
            raise AssertionError("Shared and recomputed query paths differ")
        results.append(
            {"questions": count, "timings": timings, "max_probability_difference": error}
        )
    report = {
        "hardware": hardware(),
        "device": str(model.device),
        "precision": "float32",
        "image_shape": [64, 192, 3],
        "model": model.config,
        "boundary": "Pixels to tensor, vision encoder, posterior, event compilation and answer construction",
        "cache": "Model remains loaded; shared encodes once per request; no cross-request image cache.",
        "warmups": 3,
        "repeats": 20,
        "results": results,
    }
    write_json(output, report)
    return report
