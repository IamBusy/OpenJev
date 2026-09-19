"""Versioned Qwen training data and a frozen, matched-input evaluation pack."""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from .data import make_record, normalize, onehot, rank
from .io import read_records, sha256, write_json, write_records
from .qwen import load_tokenizer, prompt
from .schema import query_text

SEED = 29
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def clean(text):
    return not (PHONE.search(text) or EMAIL.search(text) or "\\" in text or '"""' in text)


def acquire(root):
    raw = root / "data/v02/raw"
    raw.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "boolq_validation.parquet",
            "https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/validation-00000-of-00001.parquet",
            "CC-BY-SA-3.0",
        ),
        (
            "arc_easy_test.parquet",
            "https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Easy/test-00000-of-00001.parquet",
            "CC-BY-SA-4.0",
        ),
    ]
    lock_file = root / "data/v02/SOURCE_LOCK.json"
    previous = json.loads(lock_file.read_text()) if lock_file.exists() else []
    expected = {x["file"]: x["sha256"] for x in previous}
    metadata = []
    for name, url, license_name in specs:
        path = raw / name
        if not path.exists():
            response = httpx.get(url, follow_redirects=True, timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
        if name in expected and sha256(path) != expected[name]:
            raise ValueError("Downloaded source differs from its lock")
        metadata.append(
            {
                "file": name,
                "url": url,
                "license": license_name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    write_json(lock_file, metadata)
    return raw


def balanced(rows, count, key=lambda r: r.label):
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: rank(SEED, r.id)):
        groups[key(r)].append(r)
    selected = []
    labels = sorted(groups, key=lambda label: rank(SEED, str(label)))
    while len(selected) < count and any(groups.values()):
        for label in labels:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
    if len(selected) != count:
        raise ValueError(f"Not enough eligible records: needed {count}, found {len(selected)}")
    return selected


def description(label):
    text = label.replace("_", " ").replace("?", "")
    fixes = {
        "card_arrival": "delivery or arrival of a bank card",
        "get_physical_card": "ordering a physical bank card",
        "transaction_charged_twice": "being charged twice for one transaction",
        "declined_cash_withdrawal": "a declined attempt to withdraw cash",
        "passcode_forgotten": "a forgotten account passcode",
        "card_not_working": "a bank card that does not work",
        "top_up_failed": "an unsuccessful account top-up",
        "topping_up_by_card": "adding money to the account using a bank card",
    }
    return "The request concerns " + fixes.get(label, text) + "."


def neighbor_table(labels):
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([description(x) for x in labels])
    sim = (matrix @ matrix.T).toarray()
    return {
        label: [labels[j] for j in np.argsort(-sim[i], kind="stable") if j != i]
        for i, label in enumerate(labels)
    }


def banking_record(r, pool, neighbors, task, split, verification=False):
    label = r.provenance.get("original_label", r.label)
    rng = random.Random(rank(SEED, r.id + task + split))
    if verification:
        positive = rng.random() < 0.5
        target = label if positive else rng.choice(neighbors[label][:8])
        q = {
            "type": "noul",
            "instructions": f"Using a single best-intent label, is '{target.replace('_', ' ')}' the best label for this customer request?",
            "criteria": {
                "false": "No. A different intent is the best label.",
                "true": "Yes. This is the best intent label.",
            },
        }
        result = make_record(
            "banking77",
            r.source_id,
            split,
            task,
            r.state,
            q,
            onehot(2, int(positive)),
            str(positive),
            original_label=label,
            proposition_label=target,
            negative_sampling="lexically related intent; single-best-label semantics",
        )
    else:
        nearby = neighbors[label][:12]
        negatives = rng.sample(nearby, min(4, len(nearby)))
        remaining = [x for x in pool if x != label and x not in negatives]
        negatives += rng.sample(remaining, 7 - len(negatives))
        labels = [label] + negatives
        rng.shuffle(labels)
        result = make_record(
            "banking77",
            r.source_id,
            split,
            task,
            r.state,
            {
                "type": "choice",
                "instructions": "Select the single best intent category for the customer's request.",
                "criteria": {x: description(x) for x in labels},
            },
            onehot(8, labels.index(label)),
            label,
            original_label=label,
            candidate_protocol="8 candidates: gold, 4 related negatives, 3 random negatives",
        )
    result.id = "v02:" + result.id
    return result


def prepare_v02(root: Path):
    out = root / "data/v02/processed"
    if (out / "manifest.json").exists():
        raise FileExistsError("V0.2 pack already frozen")
    raw = acquire(root)
    qtok = load_tokenizer(root)
    minilm_local = root / "artifacts/openjev-minilm-v0.1/backbone"
    if minilm_local.exists():
        mtok = AutoTokenizer.from_pretrained(minilm_local)
    else:
        minilm_config = json.loads((root / "configs/local.json").read_text())
        mtok = AutoTokenizer.from_pretrained(
            minilm_config["model_name"], revision=minilm_config["model_revision"]
        )
    config = json.loads((root / "configs/qwen-v02.json").read_text())
    rejected = Counter()

    def eligible(r):
        if not clean(r.state):
            rejected["contact_or_escape_pattern"] += 1
            return False
        qn = len(qtok(prompt(qtok, r.state, r.question), add_special_tokens=False)["input_ids"])
        mn = len(mtok(query_text(r.state, r.question), add_special_tokens=True)["input_ids"])
        option_lengths = [len(mtok(x)["input_ids"]) for x in r.question.options()[1]]
        if qn > config["max_length"] or mn > 192 or max(option_lengths) > 192:
            rejected["exceeds_shared_context"] += 1
            return False
        r.provenance.update(qwen_input_tokens=qn, minilm_query_tokens=mn)
        return True

    old = {}
    for name in [
        "train",
        "dev",
        "calibration",
        "test_id",
        "test_unseen",
        "test_transfer",
        "test_simulator",
    ]:
        old[name] = read_records(root / f"data/processed/{name}.jsonl")
    seen_labels = json.loads((root / "reports/data_manifest.json").read_text())["banking77"][
        "seen_labels"
    ]
    held_labels = json.loads((root / "reports/data_manifest.json").read_text())["banking77"][
        "unseen_labels"
    ]
    seen_neighbors = neighbor_table(seen_labels)
    held_neighbors = neighbor_table(held_labels)
    outputs = {}
    for split, n in [("train", 256), ("dev", 32), ("calibration", 32)]:
        banks = [r for r in old[split] if r.task == "intent_choice"]
        choices = [
            banking_record(r, seen_labels, seen_neighbors, "banking_choice", split) for r in banks
        ]
        nouls = [
            banking_record(r, seen_labels, seen_neighbors, "banking_verify", split, True)
            for r in banks
        ]
        scores = [r.model_copy(deep=True) for r in old[split] if r.task == "sentiment_score"]
        for r in scores:
            r.id = "v02:" + r.id
        outputs[split] = (
            balanced([r for r in choices if eligible(r)], n)
            + balanced([r for r in nouls if eligible(r)], n)
            + balanced([r for r in scores if eligible(r)], n)
        )
    tests = []
    bank_test = [r for r in old["test_id"] if r.task == "intent_choice"]
    for task, is_verify in [("banking_seen_choice", False), ("banking_seen_verify", True)]:
        rows = [
            banking_record(r, seen_labels, seen_neighbors, task, "test", is_verify)
            for r in bank_test
        ]
        for r in rows:
            r.provenance["evaluation_scope"] = "previously analyzed regression corpus"
        tests += balanced([r for r in rows if eligible(r)], 64)
    sentiment = [r.model_copy(deep=True) for r in old["test_id"] if r.task == "sentiment_score"]
    for r in sentiment:
        r.id = "v02:" + r.id
        r.split = "test"
        r.provenance["evaluation_scope"] = "previously analyzed regression corpus"
    tests += balanced([r for r in sentiment if eligible(r)], 64)
    used_old = {normalize(r.state) for values in old.values() for r in values}
    fresh_banks = []
    with (root / "data/raw/banking77/train.csv").open() as f:
        for i, item in enumerate(csv.DictReader(f)):
            if item["category"] not in held_labels or normalize(item["text"]) in used_old:
                continue
            r = make_record(
                "banking77",
                f"train/{i}",
                "test",
                "placeholder",
                item["text"],
                {"type": "noul", "instructions": "placeholder"},
                [1.0, 0.0],
                item["category"],
                original_label=item["category"],
            )
            r = banking_record(r, held_labels, held_neighbors, "banking_unseen_choice", "test")
            r.provenance["evaluation_scope"] = (
                "new source records from intents excluded from all project training"
            )
            fresh_banks.append(r)
    tests += balanced([r for r in fresh_banks if eligible(r)], 64)
    arc = []
    for item in pq.read_table(raw / "arc_easy_test.parquet").to_pylist():
        c = item["choices"]
        if isinstance(c, dict):
            texts, keys = c["text"], c["label"]
        else:
            texts, keys = [x["text"] for x in c], [x["label"] for x in c]
        if len(texts) != 4 or len(set(texts)) != 4:
            continue
        order = list(range(4))
        random.Random(rank(SEED, item["id"])).shuffle(order)
        gold = keys.index(item["answerKey"])
        r = make_record(
            "arc_easy",
            item["id"],
            "test",
            "arc_easy",
            item["question"],
            {
                "type": "choice",
                "instructions": "Choose the correct answer to the question.",
                "criteria": {keys[i]: texts[i] for i in order},
            },
            onehot(4, order.index(gold)),
            keys[gold],
            evaluation_scope="new task family, public test subset",
        )
        arc.append(r)
    tests += balanced([r for r in arc if eligible(r)], 64)
    boolq = []
    for i, item in enumerate(pq.read_table(raw / "boolq_validation.parquet").to_pylist()):
        r = make_record(
            "boolq",
            f"validation/{i}",
            "test",
            "boolq",
            item["passage"],
            {
                "type": "noul",
                "instructions": "Answer using only the passage as written: "
                + item["question"]
                + "?",
                "criteria": {"false": "No.", "true": "Yes."},
            },
            onehot(2, int(item["answer"])),
            str(item["answer"]),
            evaluation_scope="new task family; public validation used as held-out project test",
        )
        boolq.append(r)
    tests += balanced([r for r in boolq if eligible(r)], 64)
    rng = random.Random(SEED)
    for language in ["en", "zh"]:
        for i in range(32):
            label = i % 3
            affected = 0 if label == 0 else rng.randint(1, 300)
            workaround = label != 2
            if language == "en":
                state = f"Service report {i + 1}: {affected} users cannot finish their work. A workaround is {'available' if workaround else 'not available'}. Other functions are normal."
                instructions = "Assign the incident level using exactly the supplied rubric."
                levels = [
                    "No users are blocked.",
                    "Some users are blocked, but a workaround exists.",
                    "Some users are blocked and no workaround exists.",
                ]
            else:
                state = f"服务记录{i + 1}：有{affected}名用户无法完成工作。{'存在临时替代方案' if workaround else '没有临时替代方案'}。其他功能正常。"
                instructions = "严格按照给定标准判断事件等级。"
                levels = [
                    "没有用户受阻。",
                    "有用户受阻，但存在临时替代方案。",
                    "有用户受阻，且没有临时替代方案。",
                ]
            r = make_record(
                "rubric_generator",
                f"{language}/{i}",
                "test",
                "rubric_" + language,
                state,
                {"type": "score", "instructions": instructions, "criteria": levels},
                onehot(3, label),
                str(label),
                language=language,
                evaluation_scope="new exact-rule diagnostic, not natural human judgments",
                generator_version="incident-v1",
            )
            if not eligible(r):
                raise ValueError("Rule diagnostic exceeded shared context budget")
            tests.append(r)
    outputs["test"] = tests
    groups = defaultdict(set)
    for split, rows in outputs.items():
        for r in rows:
            groups[r.group_id].add(split)
    collisions = [g for g, splits in groups.items() if len(splits) > 1]
    if collisions:
        raise AssertionError("V0.2 source groups leak across splits")
    summaries = {}
    for split, rows in outputs.items():
        rows.sort(key=lambda r: rank(SEED, r.id))
        write_records(out / f"{split}.jsonl", rows)
        summaries[split] = {
            "records": len(rows),
            "source_groups": len({r.group_id for r in rows}),
            "tasks": dict(Counter(r.task for r in rows)),
            "sha256": sha256(out / f"{split}.jsonl"),
        }
    # Select the paired remote reference before any model predictions.
    remote = []
    for task in sorted({r.task for r in tests}):
        remote += sorted([r for r in tests if r.task == task], key=lambda r: rank(SEED + 1, r.id))[
            :8
        ]
    write_records(out / "reference.jsonl", remote)
    summaries["reference"] = {
        "records": len(remote),
        "sha256": sha256(out / "reference.jsonl"),
        "relation": "paired subset of test, intentionally overlapping",
    }
    manifest = {
        "version": "openjev-eval-v0.2",
        "seed": SEED,
        "splits": summaries,
        "frozen_before_model_selection": True,
        "primary_train_records": 768,
        "all_v01_synthetic_excluded": True,
        "training_sources": ["BANKING77", "TweetEval sentiment"],
        "new_tasks_not_used_for_training": ["ARC-Easy", "BoolQ", "rubric_en", "rubric_zh"],
        "shared_input_rule": "No input truncation: Qwen prompt <=768 tokens and MiniLM query/candidates <=192 tokens",
        "banking_candidate_count": 8,
        "context_rejection_counts": dict(rejected),
        "reference_budget": "8 preselected records per task; no more than 64 DeepSeek calls",
        "limitations": [
            "Small local subsets, not official leaderboard scores.",
            "Old banking/sentiment test slices are regression checks, not newly blinded tests.",
            "Public benchmark contamination of pretrained models cannot be excluded.",
            "English/Chinese rubric records are exact programmatic diagnostics.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    write_json(root / "reports/v02/manifest.json", manifest)
    return manifest
