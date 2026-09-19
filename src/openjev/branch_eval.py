"""Fresh/generalization evaluation and structural diagnostics for shared branches."""

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .branch_model import BranchDecision
from .branch_train import branch_logits, state_groups
from .eval_v02 import fit_temperatures, group_metrics
from .io import read_records, sha256, write_json
from .qwen import QwenDecision
from .qwen_train import target_array


def release_memory(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def structural_diagnostics(model, fresh, regression):
    old_cases = [r for r in regression if r.question.type == "choice"][:16]
    new_cases = [r for r in fresh if r.question.type == "choice"][:32]
    report = {}
    for name, records in [("v02_matched_16", old_cases), ("fresh_32", new_cases)]:
        differences = []
        consistent = []
        for r in records:
            a = model.predict(r.state, {"q": r.question})["answers"]["q"]
            q = r.question.model_copy(deep=True)
            q.criteria = dict(reversed(list(q.criteria.items())))
            b = model.predict(r.state, {"q": q})["answers"]["q"]
            differences.append(
                max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
            )
            consistent.append(a["choice"] == b["choice"])
        report[name] = {
            "n": len(records),
            "decision_consistency": float(np.mean(consistent)),
            "max_probability_error": float(max(differences)),
            "record_ids": [r.id for r in records],
        }
    groups = [g for g in state_groups(fresh) if len(g) > 1][:8]
    isolation = []
    cache_error = []
    choices_equal = []
    model.set_training(False)
    for group in groups:
        questions = {str(i): r.question for i, r in enumerate(group)}
        shared = model.predict(group[0].state, questions)
        repeated = model.predict(group[0].state, questions, cached=False)
        for i, r in enumerate(group):
            alone = model.predict(r.state, {"q": r.question})["answers"]["q"]
            together = shared["answers"][str(i)]
            other = repeated["answers"][str(i)]
            if r.question.type == "noul":
                isolation.append(abs(alone["noul"] - together["noul"]))
                cache_error.append(abs(together["noul"] - other["noul"]))
            else:
                isolation.extend(
                    abs(alone["probabilities"][k] - together["probabilities"][k])
                    for k in alone["probabilities"]
                )
                cache_error.extend(
                    abs(together["probabilities"][k] - other["probabilities"][k])
                    for k in together["probabilities"]
                )
                if r.question.type == "choice":
                    choices_equal.append(together["choice"] == other["choice"])
    report["question_isolation"] = {
        "state_groups": len(groups),
        "max_probability_error": float(max(isolation)),
    }
    report["cache_equivalence"] = {
        "state_groups": len(groups),
        "max_probability_error": float(max(cache_error)),
        "choice_decisions_equal": all(choices_equal),
    }
    options = {f"alternative_{i}": f"An unrelated entry numbered {i}." for i in range(254)}
    options["match"] = "The recorded color is red."
    large = model.predict(
        "The recorded color is red.",
        {
            "q": {
                "type": "choice",
                "instructions": "Which candidate agrees with the recorded color?",
                "criteria": options,
            }
        },
    )
    p = large["answers"]["q"]["probabilities"]
    report["large_candidate_contract"] = {
        "n": len(p),
        "probability_sum": sum(p.values()),
        "all_finite": all(np.isfinite(v) for v in p.values()),
        "choice_in_input": large["answers"]["q"]["choice"] in options,
        "simple_example_correct": large["answers"]["q"]["choice"] == "match",
        "usage": large["usage"],
        "scope": "Shape and execution test, not a calibrated 255-way benchmark",
    }
    return report


def cache_benchmark(model):
    workloads = []
    for long_state in [False, True]:
        state = {
            "ticket": "The customer requests a refund for a duplicate payment.",
            "archive": "Unrelated log entry. " * 80 if long_state else "No extra records.",
        }
        for count in [1, 8]:
            questions = {
                str(i): {
                    "type": "choice",
                    "instructions": f"Which description best fits the customer request? Assessment {i + 1}.",
                    "criteria": {
                        "refund": "Request a refund.",
                        "delivery": "Ask about parcel delivery.",
                        "password": "Reset a password.",
                        "address": "Change an address.",
                    },
                }
                for i in range(count)
            }
            item = {
                "state_length": "long" if long_state else "short",
                "questions": count,
                "candidates_per_question": 4,
            }
            for mode in [True, False]:
                for _ in range(2):
                    model.predict(state, questions, cached=mode)
                times = []
                for _ in range(8):
                    started = time.perf_counter()
                    r = model.predict(state, questions, cached=mode)
                    times.append(time.perf_counter() - started)
                item["cached" if mode else "recomputed"] = {
                    "p50_ms": float(np.median(times) * 1000),
                    "p95_ms": float(np.quantile(times, 0.95) * 1000),
                    "raw_seconds": times,
                    "usage": r["usage"],
                }
            item["p50_speedup"] = item["recomputed"]["p50_ms"] / item["cached"]["p50_ms"]
            workloads.append(item)
            print(
                "CACHE_BENCH",
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k in ["state_length", "questions", "p50_speedup"]
                    }
                ),
                flush=True,
            )
    return {
        "workloads": workloads,
        "warmups": 2,
        "repeats": 8,
        "precision": "BF16 backbone, FP32 scalar head",
        "comparison": "Same model, inputs and branch batch size; only prefix reuse changes",
        "scope": "Warm in-process time including encoding and output construction; no cross-request cache",
        "not_a_claim": "No speed comparison to Jev's remote API",
    }


def evaluate_branch(root: Path):
    folder = root / "data/v03/processed"
    manifest = json.loads((folder / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if sha256(folder / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("Frozen data changed")
    selection = json.loads((root / "reports/v03/TRAINING_SELECTION.json").read_text())
    checkpoint = root / selection["selected_checkpoint"]
    if sha256(checkpoint / "head.safetensors") != selection["head_sha256"]:
        raise ValueError("Head changed")
    if sha256(checkpoint / "adapter/adapter_model.safetensors") != selection["adapter_sha256"]:
        raise ValueError("Adapter changed")
    calibration = read_records(folder / "calibration.jsonl")
    datasets = {
        "fresh": read_records(folder / "test_fresh.jsonl"),
        "regression": read_records(root / "data/v02/processed/test.jsonl"),
    }
    cal_targets, _ = target_array(calibration)
    output = checkpoint.parent / "evaluation"
    output.mkdir(exist_ok=True)
    all_scores = {}
    models = {}
    for name, path in [
        ("branch_initial", checkpoint.parent / "checkpoint-0"),
        ("branch_trained", checkpoint),
    ]:
        model = BranchDecision(root, checkpoint=path)
        cal_scores, _ = branch_logits(model, calibration)
        temperatures = fit_temperatures(cal_scores, calibration, cal_targets)
        models[name] = {"temperatures": temperatures, "groups": {}}
        all_scores[name] = {}
        for split, records in datasets.items():
            scores, usage = branch_logits(model, records)
            models[name]["groups"][split] = group_metrics(scores, records, temperatures)
            models[name][split + "_prefix_evaluations"] = sum(
                u["prefix_evaluations"] for u in usage
            )
            all_scores[name][split] = scores
            np.savez_compressed(
                output / f"{name}-{split}.npz",
                logits=scores,
                record_ids=np.array([r.id for r in records]),
            )
            print("EVALUATED", name, split, len(records), flush=True)
        write_json(path / "calibration-v03.json", temperatures)
        if name == "branch_trained":
            diagnostics = structural_diagnostics(model, datasets["fresh"], datasets["regression"])
            write_json(root / "reports/v03/structural_checks.json", diagnostics)
            benchmark = cache_benchmark(model)
            write_json(root / "reports/v03/cache_benchmark.json", benchmark)
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    old_selection = json.loads((root / "reports/v02/TRAINING_SELECTION.json").read_text())
    old = QwenDecision(
        root, old_selection["config"], adapter=root / old_selection["selected_adapter"]
    )
    cal_scores, _ = old.predict_records(calibration)
    temperatures = fit_temperatures(cal_scores, calibration, cal_targets)
    models["qwen_v02"] = {
        "temperatures": temperatures,
        "groups": {},
        "scope": "Unchanged v0.2 adapter, recalibrated on the same v0.3 calibration set",
    }
    all_scores["qwen_v02"] = {}
    for split, records in datasets.items():
        scores, _ = old.predict_records(records)
        models["qwen_v02"]["groups"][split] = group_metrics(scores, records, temperatures)
        all_scores["qwen_v02"][split] = scores
        np.savez_compressed(
            output / f"qwen_v02-{split}.npz",
            logits=scores,
            record_ids=np.array([r.id for r in records]),
        )
        print("EVALUATED qwen_v02", split, len(records), flush=True)
    del old
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    refs = read_records(folder / "reference.jsonl")
    index = {r.id: i for i, r in enumerate(datasets["fresh"])}
    paired = {}
    for name, values in all_scores.items():
        groups = defaultdict(list)
        for r in refs:
            correct = int(values["fresh"][index[r.id], : len(r.target)].argmax()) == int(
                np.argmax(r.target)
            )
            groups[r.task].append(correct)
        paired[name] = {
            task: {"n": len(v), "correct": sum(v), "accuracy": float(np.mean(v))}
            for task, v in groups.items()
        }
    ref_file = root / "reports/v03/deepseek_reference.json"
    if ref_file.exists():
        remote = json.loads(ref_file.read_text())
        groups = defaultdict(list)
        for r in remote["records"]:
            groups[r["task"]].append(r["correct"])
        paired["deepseek_v4_pro"] = {
            task: {"n": len(v), "correct": sum(v), "accuracy": float(np.mean(v))}
            for task, v in groups.items()
        }
    result = {
        "selection": selection,
        "data_manifest_sha256": sha256(folder / "manifest.json"),
        "models": models,
        "paired_reference": paired,
        "diagnostics": diagnostics,
        "limitations": [
            "One training seed and bounded data.",
            "v0.3 changes both architecture and training data; accuracy differences are not an architecture-only ablation.",
            "Rule simulations and public data do not establish production calibration.",
            "Relative-to-set and none-of-the-above questions need additional validation.",
        ],
    }
    write_json(root / "reports/v03/comparison.json", result)
    return result
