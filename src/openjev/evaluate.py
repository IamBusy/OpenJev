import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .io import hardware, sha256, write_json
from .metrics import summarize
from .model import cache_features, new_head, predict_logits

TEST_SPLITS = ["test_id", "test_unseen", "test_transfer", "test_simulator"]


def freeze_selection(root: Path, run_dirs, selection_file=None):
    summaries = {
        str(p.relative_to(root)): json.loads((p / "run.json").read_text()) for p in run_dirs
    }
    selected = min(summaries, key=lambda p: summaries[p]["best_dev_nll"])
    selection = {
        "selection_rule": "lowest development NLL, before any final test evaluation",
        "selected_run": selected,
        "selected_checkpoint": selected + "/checkpoint",
        "runs": {
            p: {
                "dev_nll": r["best_dev_nll"],
                "epoch": r["best_epoch"],
                "checkpoint_sha256": r["checkpoint_sha256"],
            }
            for p, r in summaries.items()
        },
        "test_files": {s: sha256(root / f"data/processed/{s}.jsonl") for s in TEST_SPLITS},
    }
    path = root / "reports" / (selection_file or "FINAL_SELECTION.json")
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("A different selection is already frozen; create a new experiment version")
    write_json(path, selection)
    return selection


def evaluate(root: Path, checkpoint: Path, report_path: Path, device="auto", selection_file=None):
    torch.set_num_threads(4)
    selection_path = root / "reports" / (selection_file or "FINAL_SELECTION.json")
    if not selection_path.exists():
        raise ValueError("Freeze model selection before opening final tests")
    selection = json.loads(selection_path.read_text())
    if sha256(checkpoint / "head.safetensors") not in {
        x["checkpoint_sha256"] for x in selection["runs"].values()
    }:
        raise ValueError("Checkpoint was not included in the frozen selection")
    config = json.loads((checkpoint / "config.json").read_text())
    calibration = json.loads((checkpoint / "calibration.json").read_text())
    if calibration["checkpoint_sha256"] != sha256(checkpoint / "head.safetensors"):
        raise ValueError("Calibration belongs to a different checkpoint")
    head = new_head(config, config["embedding_dimension"])
    head.load_state_dict(load_file(str(checkpoint / "head.safetensors")))
    report = {
        "checkpoint": str(checkpoint.relative_to(root)),
        "config": config,
        "checkpoint_sha256": sha256(checkpoint / "head.safetensors"),
        "calibration_sha256": sha256(checkpoint / "calibration.json"),
        "hardware": hardware(),
        "groups": {},
    }
    artifact_dir = checkpoint.parent / "evaluation"
    artifact_dir.mkdir(exist_ok=True)
    for split in TEST_SPLITS:
        data_path = root / f"data/processed/{split}.jsonl"
        if sha256(data_path) != selection["test_files"][split]:
            raise ValueError("Test data changed after selection was frozen")
        arrays, records, _ = cache_features(
            [data_path], config, root / "artifacts/features", device
        )
        baseline = predict_logits(head, arrays, baseline=True)
        trained = predict_logits(head, arrays)
        uniform_logits = np.where(arrays["mask"], 0.0, -1e9)
        variants = {
            "uniform": (uniform_logits, 1.0),
            "semantic_raw": (baseline, 1.0),
            "trained_raw": (trained, 1.0),
        }
        for name, logits in [("semantic", baseline), ("trained", trained)]:
            key = "baseline" if name == "semantic" else "trained"
            t = np.array([calibration[key][r.question.type]["temperature"] for r in records])[
                :, None
            ]
            variants[name + "_calibrated"] = (logits, t)
        grouped = defaultdict(list)
        for i, r in enumerate(records):
            grouped[f"{split}/{r.source}/{r.task}"].append(i)
        for group, indices in grouped.items():
            rows = [records[i] for i in indices]
            result = {}
            for variant, (logits, temperature) in variants.items():
                t = temperature[indices] if isinstance(temperature, np.ndarray) else temperature
                result[variant] = summarize(
                    rows, logits[indices], arrays["targets"][indices], arrays["mask"][indices], t
                )
            report["groups"][group] = result
            print(
                json.dumps(
                    {
                        "group": group,
                        "n": len(rows),
                        "semantic": {
                            k: v
                            for k, v in result["semantic_calibrated"].items()
                            if k in ["accuracy", "nll", "ece", "distribution_squared_error"]
                        },
                        "trained": {
                            k: v
                            for k, v in result["trained_calibrated"].items()
                            if k in ["accuracy", "nll", "ece", "distribution_squared_error"]
                        },
                    }
                ),
                flush=True,
            )
        np.savez_compressed(
            artifact_dir / f"{split}.npz",
            targets=arrays["targets"],
            mask=arrays["mask"],
            semantic_logits=baseline,
            trained_logits=trained,
            record_ids=np.array([r.id for r in records]),
        )
        write_json(report_path, report)
    return report


def compare_runs(root: Path, run_a: Path, run_b: Path, output_name="synthetic_ablation.json"):
    from .io import read_records

    result = {
        "a": str(run_a.relative_to(root)),
        "b": str(run_b.relative_to(root)),
        "metric": "paired accuracy difference B minus A; 1000 bootstrap samples within each task",
        "groups": {},
    }
    rng = np.random.default_rng(20260919)
    for split in TEST_SPLITS:
        if split == "test_simulator":
            continue
        a = np.load(run_a / f"evaluation/{split}.npz", allow_pickle=False)
        b = np.load(run_b / f"evaluation/{split}.npz", allow_pickle=False)
        if not np.array_equal(a["record_ids"], b["record_ids"]):
            raise ValueError("Comparison records do not align")
        gold = a["targets"].argmax(axis=-1)
        correct_a = a["trained_logits"].argmax(axis=-1) == gold
        correct_b = b["trained_logits"].argmax(axis=-1) == gold
        rows = read_records(root / f"data/processed/{split}.jsonl")
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[f"{split}/{r.source}/{r.task}"].append(i)
        for group, indices in groups.items():
            delta = correct_b[indices].astype(float) - correct_a[indices].astype(float)
            bootstrap = np.array(
                [rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)]
            )
            result["groups"][group] = {
                "n": len(delta),
                "accuracy_difference": float(delta.mean()),
                "bootstrap_95": list(map(float, np.quantile(bootstrap, [0.025, 0.975]))),
                "a_correct_b_wrong": int((delta == -1).sum()),
                "a_wrong_b_correct": int((delta == 1).sum()),
            }
    write_json(root / "reports" / output_name, result)
    return result
