"""Bounded local LoRA training with development-only checkpoint selection."""

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .io import hardware, read_records, sha256, write_json
from .metrics import nll
from .qwen import QwenDecision


def target_array(records, k=None):
    k = k or max(len(r.target) for r in records)
    y = np.zeros((len(records), k), dtype=np.float32)
    mask = np.zeros_like(y, dtype=bool)
    for i, r in enumerate(records):
        y[i, : len(r.target)] = r.target
        mask[i, : len(r.target)] = True
    return y, mask


def train_qwen(root: Path, name="qwen-v02-seed29"):
    cfg = json.loads((root / "configs/qwen-v02.json").read_text())
    data = root / "data/v02/processed"
    manifest = json.loads((data / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if sha256(data / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("Frozen data changed")
    train = read_records(data / "train.jsonl")
    dev = read_records(data / "dev.jsonl")
    if {r.group_id for r in train} & {r.group_id for r in dev}:
        raise ValueError("Source lineage leakage")
    run = root / "runs" / name
    if run.exists():
        raise FileExistsError("Use a new run name")
    run.mkdir(parents=True)
    write_json(run / "config.json", cfg)
    write_json(
        run / "source_manifest.json",
        {str(p.relative_to(root)): sha256(p) for p in (root / "src/openjev").glob("*.py")},
    )
    runner = QwenDecision(root, cfg, training=True)
    parameters = [p for p in runner.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    dev_targets, _ = target_array(dev)
    native, _ = runner.predict_records(dev)
    best = nll(native, dev_targets)
    selected = 0
    initial = run / "checkpoint-0"
    runner.model.save_pretrained(initial, safe_serialization=True)
    history = [{"update": 0, "dev_nll": best, "phase": "untrained adapter baseline"}]
    print(json.dumps(history[-1]), flush=True)
    rng = np.random.default_rng(cfg["seed"])
    accumulation = cfg["gradient_accumulation_steps"]
    microbatch = cfg["train_batch_size"]
    total_micro = math.ceil(len(train) / microbatch) * cfg["epochs"]
    total_updates = math.ceil(total_micro / accumulation)
    interval = max(1, total_updates // 4)
    started = time.perf_counter()
    step, update = 0, 0
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["epochs"]):
        permutation = rng.permutation(len(train))
        for start in range(0, len(train), microbatch):
            records = [train[i] for i in permutation[start : start + microbatch]]
            runner.model.train()
            scores = runner.logits(records)
            y, _ = target_array(records, scores.shape[1])
            target = torch.from_numpy(y).to(runner.device)
            raw_loss = -(target * scores.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(raw_loss):
                raise FloatingPointError("Non-finite Qwen loss")
            (raw_loss / accumulation).backward()
            losses.append(float(raw_loss.detach().cpu()))
            step += 1
            if step % accumulation and step < total_micro:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update % 8 == 0:
                print(
                    json.dumps(
                        {
                            "update": update,
                            "total_updates": total_updates,
                            "train_nll": float(np.mean(losses[-32:])),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if update % interval == 0 or step == total_micro:
                dev_scores, _ = runner.predict_records(dev)
                loss = nll(dev_scores, dev_targets)
                checkpoint = run / f"checkpoint-{update}"
                runner.model.save_pretrained(checkpoint, safe_serialization=True)
                if loss < best:
                    best, selected = loss, update
                row = {
                    "update": update,
                    "dev_nll": loss,
                    "best_update": selected,
                    "seconds": time.perf_counter() - started,
                }
                history.append(row)
                write_json(run / "history.json", history)
                print("DEV", json.dumps(row), flush=True)
    meta = {
        "model_name": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "config": cfg,
        "train_records": len(train),
        "dev_records": len(dev),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters_including_lora": sum(p.numel() for p in runner.model.parameters()),
        "selected_update": selected,
        "selected_adapter": str((run / f"checkpoint-{selected}").relative_to(root)),
        "selection_rule": "minimum development NLL including the untrained baseline",
        "best_dev_nll": best,
        "training_seconds": time.perf_counter() - started,
        "precision": "bfloat16 base; PEFT adapter precision as configured by library",
        "hardware": hardware(),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "adapter_sha256": sha256(run / f"checkpoint-{selected}/adapter_model.safetensors"),
    }
    write_json(run / "run.json", meta)
    write_json(root / "reports/v02/TRAINING_SELECTION.json", meta)
    del optimizer, runner
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return meta
