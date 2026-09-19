import json
import time
from pathlib import Path

import numpy as np
import torch

from .io import hardware, write_json
from .model import OpenJev


def benchmark(root: Path, checkpoint: Path, repeats=20, device="auto"):
    if repeats < 10:
        raise ValueError("At least 10 timing repeats are required")
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = OpenJev(checkpoint, device)
    load_seconds = time.perf_counter() - started
    state = "I used my debit card yesterday, but the merchant charged the same payment twice. Please help me recover the duplicate payment."
    categories = json.loads((root / "data/processed/manifest.json").read_text())["banking77"][
        "seen_labels"
    ]
    descriptions = ["The customer is asking about " + x.replace("_", " ") + "." for x in categories]
    variants = [(1, 4), (1, 16), (1, 62), (8, 4), (32, 4)]
    result = {
        "checkpoint": str(checkpoint.relative_to(root)),
        "hardware": hardware(),
        "encoder_device": str(model.encoder.device),
        "scorer_device": "cpu",
        "precision": "float32",
        "model_load_seconds": load_seconds,
        "timing_boundary": "Request validation, tokenization, encoder, scorer, temperature, output construction; model load excluded",
        "cache_policy": "Model stays loaded. No embeddings cached across requests. Exact duplicate texts deduplicated within one request.",
        "warmups_per_shape": 3,
        "repeats_per_shape": repeats,
        "workloads": [],
    }
    for q_count, k in variants:
        questions = {
            f"q{i}": {
                "type": "choice",
                "instructions": f"Choose the best matching customer service category for assessment {i + 1}.",
                "criteria": {f"c{j}": descriptions[(j + i) % len(descriptions)] for j in range(k)},
            }
            for i in range(q_count)
        }
        for _ in range(3):
            model.predict(state, questions)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            response = model.predict(state, questions)
            timings.append(time.perf_counter() - start)
        row = {
            "questions": q_count,
            "candidates_per_question": k,
            "p50_ms": float(np.median(timings) * 1000),
            "p95_ms": float(np.quantile(timings, 0.95) * 1000),
            "mean_ms": float(np.mean(timings) * 1000),
            "raw_seconds": timings,
            "input_tokens": response["usage"]["input_tokens"],
            "unique_encoded_texts": response["usage"]["unique_encoded_texts"],
            "truncated_inputs": response["usage"]["truncated_inputs"],
            "note": "Synthetic throughput workload: distinct instruction strings over a shared short state; not a semantic benchmark.",
        }
        result["workloads"].append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ["raw_seconds", "note"]}),
            flush=True,
        )
    criteria = dict(zip(categories[:8], descriptions[:8], strict=True))
    q = {
        "type": "choice",
        "instructions": "Which service best matches the request?",
        "criteria": criteria,
    }
    a = model.predict(state, {"original_id": q})["answers"]["original_id"]
    reversed_q = {**q, "criteria": dict(reversed(list(criteria.items())))}
    b = model.predict(state, {"renamed_id": reversed_q})["answers"]["renamed_id"]
    c = model.predict(
        state,
        {
            "original_id": q,
            "unrelated": {
                "type": "noul",
                "instructions": "The customer is asking for a new credit card.",
            },
        },
    )["answers"]["original_id"]
    perm_error = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in criteria)
    isolation_error = max(abs(a["probabilities"][k] - c["probabilities"][k]) for k in criteria)
    if max(perm_error, isolation_error) > 1e-4:
        raise AssertionError("Empirical permutation/isolation invariant failed")
    result["invariants"] = {
        "candidate_permutation_max_probability_error": perm_error,
        "unrelated_question_max_probability_error": isolation_error,
        "tolerance": 1e-4,
        "passed": True,
    }
    write_json(root / "reports/latency.json", result)
    return result
