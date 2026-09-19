"""Reproducible supervised experiments; development selection, separate calibration."""

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F

from .data import load_split, sha256, write_json
from .model import SceneModel, VisualPosterior, device_for, image_tensor, save_checkpoint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def hardware():
    return {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def tensor_data(data, device):
    return {
        "images": image_tensor(data["images"], device),
        "priors": torch.tensor(data["priors"], device=device),
        "targets": torch.tensor(data["targets"], device=device),
        "observed": torch.tensor(
            np.where(data["observed"] < 0, 2, data["observed"]), device=device, dtype=torch.long
        ),
    }


@torch.inference_mode()
def predict_arrays(model, data, batch_size=128, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    outputs, logps = [], []
    for start in range(0, len(data["images"]), batch_size):
        images = image_tensor(data["images"][start : start + batch_size], device)
        prior = torch.tensor(data["priors"][start : start + batch_size], device=device)
        out = model(images, prior)
        logp = model.log_posterior(out, prior, temperature)
        outputs.append(out.cpu().numpy())
        logps.append(logp.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(logps)


def fit(data_dir, run_dir, config, variant, seed, device="auto"):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    train = load_split(data_dir, "train")
    dev = load_split(data_dir, "dev")
    if set(train["groups"]) & set(dev["groups"]):
        raise ValueError("Training/development group overlap")
    seed_all(seed)
    devicename = device_for(device)
    model = VisualPosterior(variant, config["hidden_dim"], config["sensor_error"]).to(devicename)
    tensors = tensor_data(train, devicename)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    source = Path(__file__).parent
    manifest = {
        "variant": variant,
        "seed": seed,
        "config": config,
        "hardware": hardware(),
        "device": str(devicename),
        "precision": "float32",
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
        "data_manifest_sha256": sha256(Path(data_dir) / "manifest.json"),
        "selection": "lowest development posterior cross entropy",
    }
    write_json(run_dir / "experiment.json", manifest)
    started = time.perf_counter()
    history, best = [], float("inf")
    rng = np.random.default_rng(seed)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train["images"]))
        total = 0.0
        for start in range(0, len(order), config["batch_size"]):
            indices = torch.tensor(order[start : start + config["batch_size"]], device=devicename)
            outputs = model(tensors["images"][indices], tensors["priors"][indices])
            if variant == "evidence":
                loss = F.cross_entropy(
                    outputs.reshape(-1, 3), tensors["observed"][indices].flatten()
                )
            elif variant == "independent":
                marginal = tensors["targets"][indices] @ model.worlds
                loss = F.binary_cross_entropy_with_logits(outputs, marginal)
            else:
                loss = -(tensors["targets"][indices] * F.log_softmax(outputs, -1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        _, logp = predict_arrays(model, dev, config["batch_size"])
        dev_nll = float(-(dev["targets"] * logp).sum(-1).mean())
        row = {
            "epoch": epoch,
            "train_loss": total / len(order),
            "dev_nll": dev_nll,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if dev_nll < best:
            best = dev_nll
            save_checkpoint(
                run_dir / "checkpoint",
                model,
                {
                    "seed": seed,
                    "selected_epoch": epoch,
                    "dev_nll": dev_nll,
                    "selection_split": "dev",
                },
            )
        write_json(run_dir / "history.json", history)
        print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
    manifest["training_seconds"] = time.perf_counter() - started
    manifest["best_dev_nll"] = best
    write_json(run_dir / "experiment.json", manifest)
    calibrate(data_dir, run_dir / "checkpoint", device)
    return manifest


def calibrate(data_dir, checkpoint, device="auto"):
    predictor = SceneModel(checkpoint, device)
    calibration = load_split(data_dir, "calibration")
    raw, _ = predict_arrays(predictor.model, calibration)
    # Fit on CPU double precision; test data is not loaded here.
    prior = torch.tensor(calibration["priors"], dtype=torch.float64)
    outputs = torch.tensor(raw, dtype=torch.float64)
    cpu_model = VisualPosterior(
        predictor.config["variant"],
        predictor.config["hidden_dim"],
        predictor.config["sensor_error"],
    ).double()
    target = calibration["targets"].astype(np.float64)

    def nll(log_temperature):
        with torch.inference_mode():
            logp = cpu_model.log_posterior(outputs, prior, float(np.exp(log_temperature))).numpy()
        return float(-(target * logp).sum(-1).mean())

    temperature = 1.0
    if predictor.config["variant"] != "evidence":
        fitted = minimize_scalar(nll, bounds=(np.log(0.2), np.log(5)), method="bounded")
        temperature = float(np.exp(fitted.x))
    result = {
        "temperature": temperature,
        "fit_split": "calibration",
        "raw_nll": nll(0),
        "calibrated_nll": nll(np.log(temperature)),
        "data_sha256": sha256(Path(data_dir) / "calibration.npz"),
        "scope": "This synthetic observation model and prior mixture only.",
    }
    write_json(Path(checkpoint) / "calibration.json", result)
    return result
