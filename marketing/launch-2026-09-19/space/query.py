# Vendored from OpenJev-Vision commit 4fa973e5212881d704def6818b0def190b094f12.
# Only local import paths are changed. Licensed Apache-2.0.
"""Safe compositional event queries over a fixed, declared scene ontology."""

import ast
import re

import numpy as np
from world import SLOTS, WORLDS

PREDICATES = {"blue": (0, 0), "red": (0, 1), "circle": (1, 0), "square": (1, 1)}


def compile_event(expression):
    if not isinstance(expression, str) or not 1 <= len(expression) <= 2048:
        raise ValueError("An event must be a nonempty expression of at most 2048 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid event expression") from exc
    if len(list(ast.walk(tree))) > 128:
        raise ValueError("Event is too complex")

    def slot(node):
        if not isinstance(node, ast.Name) or node.id not in SLOTS:
            raise ValueError("Object must be left, center, or right")
        return SLOTS.index(node.id)

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return np.full(64, node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~visit(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [visit(item) for item in node.values]
            operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            return operation.reduce(values)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in PREDICATES and len(node.args) == 1:
                attr, value = PREDICATES[name]
                return WORLDS[:, 2 * slot(node.args[0]) + attr] == value
            if name in {"same_color", "same_shape"} and len(node.args) == 2:
                attr = int(name == "same_shape")
                return (
                    WORLDS[:, 2 * slot(node.args[0]) + attr]
                    == WORLDS[:, 2 * slot(node.args[1]) + attr]
                )
        raise ValueError("Unsupported event syntax; arbitrary Python is never executed")

    return visit(tree.body)


def question_from_text(text):
    """An explicit controlled-English parser, not a pretrained language model."""
    if not isinstance(text, str):
        raise ValueError("Question must be text")
    normalized = re.sub(r"\s+", " ", text.strip().lower().rstrip("?."))
    match = re.fullmatch(
        r"is the (left|center|right) object (?:a )?(red|blue|circle|square)", normalized
    )
    if match:
        name, predicate = match.groups()
        return {"type": "noul", "event": f"{predicate}({name})"}
    match = re.fullmatch(r"is (any|every) object (red|blue|a circle|a square)", normalized)
    if match:
        quantifier, predicate = match.groups()
        predicate = predicate.removeprefix("a ")
        join = " or " if quantifier == "any" else " and "
        return {"type": "noul", "event": join.join(f"{predicate}({s})" for s in SLOTS)}
    match = re.fullmatch(r"how many objects are (red|blue|circles|squares)", normalized)
    if match:
        predicate = match[1].removesuffix("s")
        return {"type": "score", "events": [f"{predicate}({s})" for s in SLOTS]}
    match = re.fullmatch(
        r"do the (left|center|right) and (left|center|right) objects have the same (color|shape)",
        normalized,
    )
    if match:
        a, b, attribute = match.groups()
        return {"type": "noul", "event": f"same_{attribute}({a}, {b})"}
    raise ValueError(
        "Question is outside the controlled-English grammar; provide an explicit event expression"
    )


def answer_questions(posterior, questions):
    p = np.asarray(posterior, dtype=np.float64)
    if p.shape != (64,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid posterior")
    p = p / p.sum()
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 256:
        raise ValueError("Provide 1–256 named questions")
    answers = {}
    for name, spec in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Question IDs must be nonempty strings")
        if isinstance(spec, str):
            spec = question_from_text(spec)
        kind = spec.get("type")
        if kind == "noul":
            answers[name] = {"type": kind, "noul": float(p @ compile_event(spec["event"]))}
        elif kind == "choice":
            criteria = spec["criteria"]
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 64:
                raise ValueError("Choice needs 2–64 event candidates")
            masks = np.stack([compile_event(expr) for expr in criteria.values()])
            if not np.all(masks.sum(0) == 1):
                raise ValueError(
                    "Choice events must form an exhaustive, mutually exclusive partition"
                )
            probs = masks @ p
            keys = list(criteria)
            answers[name] = {
                "type": kind,
                "choice": keys[int(probs.argmax())],
                "probabilities": dict(zip(keys, map(float, probs), strict=True)),
            }
        elif kind == "score":
            events = spec["events"]
            if not isinstance(events, list) or not 1 <= len(events) <= 32:
                raise ValueError("Score needs 1–32 counted events")
            count = np.stack([compile_event(event) for event in events]).sum(0)
            distribution = np.bincount(count, weights=p, minlength=len(events) + 1)
            answers[name] = {
                "type": kind,
                "score": float(p @ count),
                "probabilities": {str(i): float(v) for i, v in enumerate(distribution)},
            }
        else:
            raise ValueError("Question type must be noul, choice, or score")
    return answers


def evaluation_queries():
    specs = []
    for predicate in PREDICATES:
        for slot_name in SLOTS:
            specs.append(("unary", f"{predicate}({slot_name})"))
        expressions = [f"{predicate}({s})" for s in SLOTS]
        specs.extend(
            [
                ("compound", " and ".join(expressions)),
                ("compound", " or ".join(expressions)),
                ("compound", f"{expressions[0]} and not {expressions[2]}"),
            ]
        )
    for a, b in (("left", "center"), ("center", "right"), ("left", "right")):
        specs.extend(
            [
                ("compound", f"same_color({a}, {b})"),
                ("compound", f"same_shape({a}, {b})"),
                ("compound", f"not red({a}) or square({b})"),
                ("compound", f"red({a}) and square({b})"),
            ]
        )
    return specs, np.stack([compile_event(expression) for _, expression in specs]).astype(float)
