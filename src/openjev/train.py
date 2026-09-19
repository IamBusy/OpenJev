import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.optimize import minimize_scalar

from .io import hardware, seed_everything, sha256, write_json
from .metrics import nll
from .model import TYPE_INDEX, batch, cache_features, new_head, predict_logits, save_head, tensors


def train(root: Path, config_path: Path, run_dir: Path, augmented=False, device="auto"):
    config = json.loads(config_path.read_text())
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "source_manifest.json",
        {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src/openjev").glob("*.py"))},
    )
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    data_dir = root / "data/processed"
    paths = [data_dir / "train.jsonl"]
    if augmented:
        paths.append(data_dir / "synthetic_train.jsonl")
    cache_dir = root / "artifacts/features"
    train_arrays, train_records, train_cache = cache_features(paths, config, cache_dir, device)
    dev_arrays, dev_records, dev_cache = cache_features(
        [data_dir / "dev.jsonl"], config, cache_dir, device
    )
    if {r.group_id for r in train_records} & {r.group_id for r in dev_records}:
        raise ValueError("Training/development group leakage")
    if any(r.split != "train" for r in train_records) or any(r.split != "dev" for r in dev_records):
        raise ValueError("Wrong split supplied to training/selection")
    # Loading a pretrained encoder on a cache miss consumes torch RNG state.
    # Initialize training RNG only after all feature-cache work so a clean run
    # and a cached rerun start from identical head weights and dropout streams.
    seed_everything(config["seed"])
    dim = train_arrays["embeddings"].shape[1]
    head = new_head(config, dim)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    features = tensors(train_arrays)
    rng = np.random.default_rng(config["seed"])
    best, best_epoch, stale = float("inf"), 0, 0
    initial = predict_logits(head, dev_arrays, baseline=True)
    history = [
        {"epoch": 0, "dev_nll": nll(initial, dev_arrays["targets"]), "variant": "semantic_baseline"}
    ]
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        head.train()
        permutation = rng.permutation(len(train_records))
        loss_sum = 0.0
        for start in range(0, len(permutation), config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]]
            logits = head(*batch(features, indices))
            target = features["targets"][indices]
            loss = -(target * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        dev_logits = predict_logits(head, dev_arrays)
        dev_loss = nll(dev_logits, dev_arrays["targets"])
        row = {
            "epoch": epoch,
            "train_nll": loss_sum / len(train_records),
            "dev_nll": dev_loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_loss < best - 1e-5:
            best, best_epoch, stale = dev_loss, epoch, 0
            save_head(
                run_dir / "checkpoint",
                head,
                config,
                embedding_dimension=int(dim),
                augmented=augmented,
                selected_epoch=epoch,
                selection_metric="development NLL only",
            )
        else:
            stale += 1
        write_json(run_dir / "training_history.json", history)
        if stale >= config["patience"]:
            break
    head.load_state_dict(load_file(str(run_dir / "checkpoint/head.safetensors")))
    head.eval()
    before = predict_logits(head, dev_arrays)[0:8]
    reloaded = new_head(config, dim)
    reloaded.load_state_dict(load_file(str(run_dir / "checkpoint/head.safetensors")))
    after = predict_logits(reloaded, dev_arrays)[0:8]
    reload_error = float(np.max(np.abs(before - after)))
    if reload_error > 1e-6:
        raise AssertionError("Checkpoint reload changed predictions")
    metadata = {
        "config": config,
        "augmented": augmented,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "best_epoch": best_epoch,
        "best_dev_nll": best,
        "training_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "backbone_training": "frozen pretrained encoder; trainable shared residual scorer",
        "head_training_device": "cpu",
        "hardware": hardware(),
        "train_files": {str(p.relative_to(root)): sha256(p) for p in paths},
        "dev_sha256": sha256(data_dir / "dev.jsonl"),
        "feature_caches": [str(train_cache.relative_to(root)), str(dev_cache.relative_to(root))],
        "checkpoint_sha256": sha256(run_dir / "checkpoint/head.safetensors"),
        "dependency_lock_sha256": sha256(root / "uv.lock"),
        "reload_max_logit_error": reload_error,
    }
    write_json(run_dir / "run.json", metadata)
    return metadata


def calibrate(root: Path, checkpoint: Path, device="auto"):
    config = json.loads((checkpoint / "config.json").read_text())
    path = root / "data/processed/calibration.jsonl"
    arrays, records, _ = cache_features([path], config, root / "artifacts/features", device)
    if any(r.split != "calibration" for r in records):
        raise ValueError("Temperature fitting requires calibration records only")
    head = new_head(config, arrays["embeddings"].shape[1])
    head.load_state_dict(load_file(str(checkpoint / "head.safetensors")))
    result = {
        "fit_split": "calibration",
        "data_sha256": sha256(path),
        "method": "one scalar temperature per question type, bounded [0.05, 20]",
        "checkpoint_sha256": sha256(checkpoint / "head.safetensors"),
    }
    for variant, baseline in [("baseline", True), ("trained", False)]:
        logits = predict_logits(head, arrays, baseline=baseline)
        result[variant] = {}
        for name, idx in TYPE_INDEX.items():
            chosen = arrays["types"] == idx
            x, y = logits[chosen], arrays["targets"][chosen]
            fit = minimize_scalar(
                lambda logt: nll(x, y, np.exp(logt)),
                bounds=(np.log(0.05), np.log(20.0)),
                method="bounded",
            )
            result[variant][name] = {
                "temperature": float(np.exp(fit.x)),
                "n_records": int(chosen.sum()),
                "nll_before": nll(x, y),
                "nll_after": float(fit.fun),
                "sources": sorted(
                    {r.source for r, yes in zip(records, chosen, strict=True) if yes}
                ),
            }
    write_json(checkpoint / "calibration.json", result)
    return result
