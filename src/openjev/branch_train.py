"""Train the branch scorer with differentiable, request-local prefix reuse."""

import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .branch_model import BranchDecision
from .io import hardware, read_records, sha256, write_json
from .metrics import nll
from .qwen_train import target_array


def state_groups(records):
    groups = defaultdict(list)
    for r in records:
        groups[r.group_id].append(r)
    result = list(groups.values())
    for rows in result:
        if len({r.state for r in rows}) != 1:
            raise ValueError("A state group contains differing encoded states")
    return result


def branch_logits(model, records, cached=True):
    model.set_training(False)
    width = max(len(r.target) for r in records)
    logits = np.full((len(records), width), -1e9, dtype=np.float32)
    index = {r.id: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("Duplicate record IDs")
    usage = []
    with torch.inference_mode():
        for rows in state_groups(records):
            scores, stats = model.score(rows[0].state, [r.question for r in rows], cached)
            for r, s in zip(rows, scores, strict=True):
                logits[index[r.id], : len(r.target)] = s.detach().cpu().numpy()
            usage.append(stats)
    return logits, usage


def train_branch(root: Path, name="branch-v03-seed43"):
    cfg = json.loads((root / "configs/branch-v03.json").read_text())
    data = root / "data/v03/processed"
    manifest = json.loads((data / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if sha256(data / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("Data changed after freeze")
    train = read_records(data / "train.jsonl")
    dev = read_records(data / "dev.jsonl")
    if {r.group_id for r in train} & {r.group_id for r in dev}:
        raise ValueError("Split leakage")
    groups = state_groups(train)
    run = root / "runs" / name
    if run.exists():
        raise FileExistsError("Use a new training run name")
    run.mkdir(parents=True)
    write_json(
        run / "source_manifest.json",
        {str(p.relative_to(root)): sha256(p) for p in (root / "src/openjev").glob("*.py")},
    )
    model = BranchDecision(root, cfg, training=True)
    adapter_parameters = [p for p in model.lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": cfg["lora_learning_rate"]},
            {"params": list(model.head.parameters()), "lr": cfg["head_learning_rate"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    targets, _ = target_array(dev)
    initial, _ = branch_logits(model, dev)
    best = nll(initial, targets)
    selected = 0
    model.save(run / "checkpoint-0", selected_update=0)
    history = [{"update": 0, "dev_nll": best, "phase": "untrained branch scorer"}]
    write_json(run / "history.json", history)
    print("INITIAL", json.dumps(history[-1]), flush=True)
    accum = cfg["gradient_accumulation_states"]
    total_updates = math.ceil(len(groups) / accum) * cfg["epochs"]
    interval = max(1, total_updates // 4)
    rng = np.random.default_rng(cfg["seed"])
    started = time.perf_counter()
    update = 0
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(groups))
        for start in range(0, len(order), accum):
            chunk = [groups[i] for i in order[start : start + accum]]
            denominator = sum(map(len, chunk))
            optimizer.zero_grad(set_to_none=True)
            model.set_training(True)
            loss_total = 0.0
            for rows in chunk:
                scores, _ = model.score(rows[0].state, [r.question for r in rows], cached=True)
                losses = []
                for r, s in zip(rows, scores, strict=True):
                    target = torch.tensor(r.target, device=model.device, dtype=torch.float32)
                    losses.append(-(target * s.float().log_softmax(-1)).sum())
                loss = torch.stack(losses).sum() / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                loss.backward()
                loss_total += float(loss.detach().cpu())
            norm = torch.nn.utils.clip_grad_norm_(
                adapter_parameters + list(model.head.parameters()), 1.0
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            update += 1
            if update % 8 == 0:
                print(
                    json.dumps(
                        {
                            "update": update,
                            "total_updates": total_updates,
                            "train_nll": loss_total,
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if update % interval == 0 or update == total_updates:
                scores, _ = branch_logits(model, dev)
                value = nll(scores, targets)
                if value < best:
                    best, selected = value, update
                model.save(run / f"checkpoint-{update}", selected_update=update)
                row = {
                    "update": update,
                    "dev_nll": value,
                    "best_update": selected,
                    "seconds": time.perf_counter() - started,
                }
                history.append(row)
                write_json(run / "history.json", history)
                print("DEV", json.dumps(row), flush=True)
    path = run / f"checkpoint-{selected}"
    summary = {
        "config": cfg,
        "run": str(run.relative_to(root)),
        "train_records": len(train),
        "train_state_groups": len(groups),
        "dev_records": len(dev),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter_parameters),
        "trainable_head_parameters": sum(p.numel() for p in model.head.parameters()),
        "selected_checkpoint": str(path.relative_to(root)),
        "selected_update": selected,
        "initial_dev_nll": history[0]["dev_nll"],
        "best_dev_nll": best,
        "training_seconds": time.perf_counter() - started,
        "adapter_sha256": sha256(path / "adapter/adapter_model.safetensors"),
        "head_sha256": sha256(path / "head.safetensors"),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "selection": "development NLL only, including untrained baseline",
        "hardware": hardware(),
    }
    write_json(run / "run.json", summary)
    write_json(root / "reports/v03/TRAINING_SELECTION.json", summary)
    del model, optimizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary
