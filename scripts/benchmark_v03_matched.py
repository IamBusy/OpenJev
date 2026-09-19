"""Interleave all runtime paths on identical requests to reduce timing-order bias."""

import json
import time
from pathlib import Path

import numpy as np

from openjev.branch_model import BranchDecision
from openjev.io import write_json
from openjev.qwen import QwenDecision

ROOT = Path(__file__).resolve().parents[1]


def main():
    selection = json.loads((ROOT / "reports/v03/TRAINING_SELECTION.json").read_text())
    branch = BranchDecision(ROOT, checkpoint=ROOT / selection["selected_checkpoint"])
    old_selection = json.loads((ROOT / "reports/v02/TRAINING_SELECTION.json").read_text())
    old = QwenDecision(
        ROOT, old_selection["config"], adapter=ROOT / old_selection["selected_adapter"]
    )
    workloads = []
    modes = ["shared", "recomputed", "v02"]
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

            def run(mode):
                return (
                    old.predict(state, questions)
                    if mode == "v02"
                    else branch.predict(state, questions, cached=mode == "shared")
                )

            for _ in range(2):
                for mode in modes:
                    run(mode)
            times = {m: [] for m in modes}
            responses = {}
            for i in range(8):
                order = modes[i % 3 :] + modes[: i % 3]
                for mode in order:
                    started = time.perf_counter()
                    responses[mode] = run(mode)
                    times[mode].append(time.perf_counter() - started)
            errors = []
            for key in questions:
                a = responses["shared"]["answers"][key]["probabilities"]
                b = responses["recomputed"]["answers"][key]["probabilities"]
                errors.extend(abs(a[k] - b[k]) for k in a)
            item = {
                "state_length": "long" if long_state else "short",
                "questions": count,
                "candidates_per_question": 4,
                "shared_vs_recomputed_max_probability_error": max(errors),
                "paths": {
                    m: {
                        "p50_ms": float(np.median(t) * 1000),
                        "p95_ms": float(np.quantile(t, 0.95) * 1000),
                        "raw_seconds": t,
                        "usage": responses[m]["usage"],
                    }
                    for m, t in times.items()
                },
            }
            item["shared_speedup_over_recomputed"] = (
                item["paths"]["recomputed"]["p50_ms"] / item["paths"]["shared"]["p50_ms"]
            )
            item["v03_shared_over_v02_latency_ratio"] = (
                item["paths"]["shared"]["p50_ms"] / item["paths"]["v02"]["p50_ms"]
            )
            workloads.append(item)
            print(
                json.dumps(
                    {
                        "state": item["state_length"],
                        "questions": count,
                        "milliseconds": {k: v["p50_ms"] for k, v in item["paths"].items()},
                    }
                ),
                flush=True,
            )
    result = {
        "warmups": 2,
        "repeats": 8,
        "mode_order": "cyclically interleaved",
        "hardware": "Apple M3 Pro, 36 GB, MPS, 4 CPU threads",
        "precision": "BF16 backbones; FP32 branch head",
        "boundary": "public predict call, warm model, no network or model load",
        "checkpoint": selection["selected_checkpoint"],
        "workloads": workloads,
        "important": "Sharing speedup compares identical v0.3 models; the v0.2 column is the actual previous architecture.",
    }
    write_json(ROOT / "reports/v03/matched_latency.json", result)


if __name__ == "__main__":
    main()
