import math

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score


def probabilities(logits, temperatures=1.0):
    x = np.asarray(logits, dtype=np.float64) / temperatures
    return np.exp(x - logsumexp(x, axis=-1, keepdims=True))


def nll(logits, target, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64) / temperature
    logp = x - logsumexp(x, axis=-1, keepdims=True)
    return float(-(target * logp).sum(axis=-1).mean())


def reliability(confidence, correct, bins=15):
    result = []
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for i in range(bins):
        selected = assignments == i
        count = int(selected.sum())
        row = {"lower": i / bins, "upper": (i + 1) / bins, "count": count}
        if count:
            row.update(
                confidence=float(confidence[selected].mean()),
                accuracy=float(correct[selected].mean()),
            )
            ece += count / len(correct) * abs(row["confidence"] - row["accuracy"])
        result.append(row)
    return float(ece), result


def wilson(successes, n):
    z = 1.95996398454
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def summarize(records, logits, target, mask, temperatures=1.0):
    p = probabilities(logits, temperatures)
    predicted = p.argmax(axis=-1)
    gold = target.argmax(axis=-1)
    confidence = p.max(axis=-1)
    soft = not np.allclose(target.max(axis=-1), 1.0)
    correct = (
        target[np.arange(len(target)), predicted] if soft else (predicted == gold).astype(float)
    )
    ece, bins = reliability(confidence, correct)
    result = {
        "n": len(records),
        "unique_source_groups": len({r.group_id for r in records}),
        "target_kind": "known_distribution" if soft else "human_or_teacher_label",
        "nll": nll(logits, target, temperatures),
        "mean_confidence": float(confidence.mean()),
        "candidate_count_min": int(mask.sum(axis=-1).min()),
        "candidate_count_max": int(mask.sum(axis=-1).max()),
        "reliability_bins": bins,
    }
    if soft:
        result["distribution_squared_error"] = float(((p - target) ** 2).sum(axis=-1).mean())
        result["expected_event_brier"] = float(
            (1 + (p * p).sum(axis=-1) - 2 * (p * target).sum(axis=-1)).mean()
        )
        result["expected_accuracy"] = float(correct.mean())
        result["expected_ece"] = ece
        entropy = -(target * np.log(np.maximum(target, 1e-30))).sum(axis=-1).mean()
        result["excess_nll_over_true_distribution"] = float(result["nll"] - entropy)
        result["mode_agreement"] = float((predicted == gold).mean())
    else:
        result["accuracy"] = float(correct.mean())
        result["accuracy_wilson_95"] = wilson(int(correct.sum()), len(records))
        result["brier"] = float(((p - target) ** 2).sum(axis=-1).mean())
        result["ece"] = ece
        pred_names = [
            r.question.options()[0][int(i)] for r, i in zip(records, predicted, strict=True)
        ]
        gold_names = [r.question.options()[0][int(i)] for r, i in zip(records, gold, strict=True)]
        result["macro_f1"] = float(
            f1_score(gold_names, pred_names, average="macro", zero_division=0)
        )
    coverage = []
    for fraction in [0.5, 0.8, 0.9, 1.0]:
        cutoff = np.sort(confidence)[::-1][max(0, math.ceil(len(records) * fraction) - 1)]
        accepted = confidence >= cutoff
        coverage.append(
            {
                "requested_coverage": fraction,
                "actual_coverage": float(accepted.mean()),
                "n_accepted": int(accepted.sum()),
                "threshold": float(cutoff),
                "accuracy" if not soft else "expected_accuracy": float(correct[accepted].mean()),
                "risk": float(1 - correct[accepted].mean()),
                "ties": "include all examples at the threshold",
            }
        )
    result["risk_coverage"] = coverage
    if records[0].question.type == "score":
        levels = np.arange(p.shape[1])
        result["ordinal_mae"] = float(np.abs(p @ levels - target @ levels).mean())
        cdf_error = (p.cumsum(axis=-1) - target.cumsum(axis=-1)) ** 2
        result["ranked_probability_score"] = float(
            (cdf_error[:, :-1].sum(axis=-1) / (mask.sum(axis=-1) - 1)).mean()
        )
    return result
