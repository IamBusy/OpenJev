"""Paired Qwen/MiniLM/reference evaluation on the frozen v0.2 pack."""

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .io import read_records, sha256, write_json
from .metrics import nll, summarize, wilson
from .model import cache_features, new_head, predict_logits
from .qwen import QwenDecision
from .qwen_train import target_array


def fit_temperatures(scores, records, targets):
    result = {}
    for kind in ["choice", "noul", "score"]:
        indices = [i for i, r in enumerate(records) if r.question.type == kind]
        x, y = scores[indices], targets[indices]
        fit = minimize_scalar(
            lambda t: nll(x, y, np.exp(t)), bounds=(np.log(0.05), np.log(20.0)), method="bounded"
        )
        result[kind] = {
            "temperature": float(np.exp(fit.x)),
            "n": len(indices),
            "calibration_nll_before": nll(x, y),
            "calibration_nll_after": float(fit.fun),
        }
    return result


def group_metrics(scores, records, temperatures):
    targets, mask = target_array(records, scores.shape[1])
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r.task].append(i)
    output = {}
    for task, indices in groups.items():
        rows = [records[i] for i in indices]
        t = np.array([temperatures[r.question.type]["temperature"] for r in rows])[:, None]
        output[task] = {
            "raw": summarize(rows, scores[indices], targets[indices], mask[indices]),
            "calibrated": summarize(rows, scores[indices], targets[indices], mask[indices], t),
        }
    return output


def latency_qwen(runner, record, repeats=12):
    runner.model.eval()
    with torch.inference_mode():
        for _ in range(3):
            runner.logits([record]).cpu().numpy()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            scores = runner.logits([record])
            scores.softmax(-1).cpu().numpy()
            times.append(time.perf_counter() - start)
    return {
        "p50_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.quantile(times, 0.95) * 1000),
        "raw_seconds": times,
        "repeats": repeats,
        "warmup": 3,
        "candidates": len(record.target),
        "record_id": record.id,
        "precision": "BF16",
        "boundary": "prompt construction, tokenization, forward, candidate softmax and CPU readback",
    }


def run_evaluation(root: Path):
    data = root / "data/v02/processed"
    manifest = json.loads((data / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if sha256(data / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("Frozen split changed")
    selection = json.loads((root / "reports/v02/TRAINING_SELECTION.json").read_text())
    config = selection["config"]
    adapter = root / selection["selected_adapter"]
    if sha256(adapter / "adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError("Selected adapter changed")
    calibration = read_records(data / "calibration.jsonl")
    test = read_records(data / "test.jsonl")
    reference = read_records(data / "reference.jsonl")
    cal_y, _ = target_array(calibration)
    run = adapter.parent
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    runner = QwenDecision(root, config, adapter=adapter)
    models = {}
    logits = {}
    timed = next(r for r in test if r.task == "banking_seen_choice")
    for name, disabled in [("qwen3_0.6b_native", True), ("qwen3_0.6b_lora", False)]:
        import contextlib

        context = runner.model.disable_adapter() if disabled else contextlib.nullcontext()
        with context:
            cal_scores, _ = runner.predict_records(calibration)
            temps = fit_temperatures(cal_scores, calibration, cal_y)
            test_scores, batch_times = runner.predict_records(test)
            logits[name] = test_scores
            models[name] = {
                "model": config["model_name"],
                "model_revision": config["model_revision"],
                "adapter": None if disabled else str(adapter.relative_to(root)),
                "groups": group_metrics(test_scores, test, temps),
                "temperatures": temps,
                "latency": latency_qwen(runner, timed),
                "evaluation_batch_times": batch_times,
            }
            np.savez_compressed(
                out / f"{name}.npz",
                logits=test_scores,
                calibration_logits=cal_scores,
                record_ids=np.array([r.id for r in test]),
            )
            print("MODEL_EVALUATED", name, flush=True)
    # Compare option reordering by probability and decision consistency; a causal
    # LM is not assumed permutation invariant by construction.
    perm_cases = [r for r in test if r.question.type == "choice"][:16]
    originals, _ = runner.predict_records(perm_cases)
    reordered = []
    for r in perm_cases:
        r = r.model_copy(deep=True)
        r.question.criteria = dict(reversed(list(r.question.criteria.items())))
        r.target = list(reversed(r.target))
        reordered.append(r)
    reversed_logits, _ = runner.predict_records(reordered)
    consistency = []
    for i, r in enumerate(perm_cases):
        k = len(r.target)
        consistency.append(
            int(originals[i, :k].argmax()) == k - 1 - int(reversed_logits[i, :k].argmax())
        )
    permutation = {
        "n": len(consistency),
        "decision_consistency": float(np.mean(consistency)),
        "scope": "exploratory option-order diagnostic; not used for selection",
    }
    del runner
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    legacy = root / "artifacts/openjev-minilm-v0.1"
    cfg = json.loads((legacy / "config.json").read_text())
    cal_arrays, cal_records, _ = cache_features(
        [data / "calibration.jsonl"], cfg, root / "artifacts/features"
    )
    test_arrays, test_records, _ = cache_features(
        [data / "test.jsonl"], cfg, root / "artifacts/features"
    )
    head = new_head(cfg, cfg["embedding_dimension"])
    head.load_state_dict(load_file(str(legacy / "head.safetensors")))
    cal_scores = predict_logits(head, cal_arrays)
    test_scores = predict_logits(head, test_arrays)
    temps = fit_temperatures(cal_scores, cal_records, cal_arrays["targets"])
    models["openjev_minilm_v01"] = {
        "model": "OpenJev-MiniLM-v0.1",
        "groups": group_metrics(test_scores, test_records, temps),
        "temperatures": temps,
        "note": "Released v0.1 weights, recalibrated on the same 96-row v0.2 calibration set",
    }
    logits["openjev_minilm_v01"] = test_scores
    np.savez_compressed(
        out / "openjev_minilm_v01.npz",
        logits=test_scores,
        calibration_logits=cal_scores,
        record_ids=np.array([r.id for r in test]),
    )

    train = read_records(data / "train.jsonl")
    old = read_records(root / "data/processed/train.jsonl")
    for name, rows in [
        ("tfidf_matching_choice_data", [r for r in train if r.task == "banking_choice"]),
        (
            "tfidf_full_seen_training",
            [r for r in old if r.task == "intent_choice" and r.source == "banking77"],
        ),
    ]:
        classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(C=4.0, max_iter=1000),
        )
        classifier.fit([r.state for r in rows], [r.provenance["original_label"] for r in rows])
        classes = {str(label): i for i, label in enumerate(classifier[-1].classes_)}

        def score(items):
            probability = classifier.predict_proba([r.state for r in items])
            values = np.array(
                [
                    [probability[i, classes[key]] for key in r.question.options()[0]]
                    for i, r in enumerate(items)
                ]
            )
            return np.log(np.maximum(values, 1e-20))

        cal_rows = [r for r in calibration if r.task == "banking_choice"]
        test_rows = [r for r in test if r.task == "banking_seen_choice"]
        cal_scores = score(cal_rows)
        y, _ = target_array(cal_rows)
        fit = minimize_scalar(
            lambda t: nll(cal_scores, y, np.exp(t)),
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
        )
        temperature = float(np.exp(fit.x))
        scores = score(test_rows)
        y, mask = target_array(test_rows)
        models[name] = {
            "training_records": len(rows),
            "task_support": "seen banking taxonomy only",
            "groups": {
                "banking_seen_choice": {
                    "raw": summarize(test_rows, scores, y, mask),
                    "calibrated": summarize(test_rows, scores, y, mask, temperature),
                }
            },
            "temperature": temperature,
        }
        print("MODEL_EVALUATED", name, flush=True)
    paired = {}
    test_index = {r.id: i for i, r in enumerate(test)}
    reference_by_task = defaultdict(list)
    for r in reference:
        reference_by_task[r.task].append(r)
    for name, scores in logits.items():
        paired[name] = {}
        for task, rows in reference_by_task.items():
            correct = sum(
                int(scores[test_index[r.id], : len(r.target)].argmax()) == int(np.argmax(r.target))
                for r in rows
            )
            paired[name][task] = {
                "n": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "wilson_95": wilson(correct, len(rows)),
            }
    remote_file = root / "reports/v02/deepseek_reference.json"
    if remote_file.exists():
        remote = json.loads(remote_file.read_text())
        paired["deepseek_v4_pro"] = {}
        for task in reference_by_task:
            rows = [r for r in remote["records"] if r["task"] == task]
            correct = sum(r["correct"] for r in rows)
            paired["deepseek_v4_pro"][task] = {
                "n": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "valid": sum(r["valid"] for r in rows),
            }
    result = {
        "pack_sha256": sha256(data / "test.jsonl"),
        "n": len(test),
        "models": models,
        "paired_reference_64": paired,
        "option_permutation": permutation,
        "selection": selection,
        "limitations": [
            "One Qwen training seed; small local subsets.",
            "Classical classifier support is limited to the seen banking taxonomy.",
            "DeepSeek uses only the frozen 64-record subset; remote latency is separate.",
            "Calibrators see only the same 96 in-domain records for all local general models.",
            "Qwen outputs probabilities conditional on the allowed label tokens.",
        ],
    }
    write_json(root / "reports/v02/comparison.json", result)
    return result
