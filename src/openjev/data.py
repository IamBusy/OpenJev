"""Pinned public sources, group-disjoint splits, and a known-probability simulator."""

import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .io import digest, sha256, write_json, write_records
from .schema import Question, Record

BANK_REV = "57ec275d8078af65b7731c2a98be812d844a6d6b"
CLINC_REV = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
TWEET_REV = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def source_specs():
    rows = []

    def add(source, repo, revision, name, path, license_name):
        rows.append(
            {
                "source": source,
                "revision": revision,
                "file": name,
                "url": f"https://raw.githubusercontent.com/{repo}/{revision}/{path}",
                "license": license_name,
            }
        )

    for path in ["train.csv", "test.csv", "categories.json"]:
        add(
            "banking77",
            "PolyAI-LDN/task-specific-datasets",
            BANK_REV,
            f"banking77/{path}",
            f"banking_data/{path}",
            "CC-BY-4.0",
        )
    add(
        "banking77",
        "PolyAI-LDN/task-specific-datasets",
        BANK_REV,
        "banking77/LICENSE",
        "LICENSE",
        "CC-BY-4.0",
    )
    add(
        "clinc150",
        "clinc/oos-eval",
        CLINC_REV,
        "clinc150/data_full.json",
        "data/data_full.json",
        "CC-BY-3.0",
    )
    add("clinc150", "clinc/oos-eval", CLINC_REV, "clinc150/LICENSE", "LICENSE", "CC-BY-3.0")
    for split in ["train", "val", "test"]:
        for kind in ["text", "labels"]:
            name = f"{split}_{kind}.txt"
            add(
                "tweeteval_sentiment",
                "cardiffnlp/tweeteval",
                TWEET_REV,
                f"tweeteval_sentiment/{name}",
                f"datasets/sentiment/{name}",
                "CC-BY-3.0",
            )
    add(
        "tweeteval_sentiment",
        "cardiffnlp/tweeteval",
        TWEET_REV,
        "tweeteval_sentiment/README.md",
        "README.md",
        "see task-specific CC-BY-3.0 notice",
    )
    return rows


def fetch_sources(root: Path):
    raw = root / "data/raw"
    lock_path = root / "data/SOURCE_LOCK.json"
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else []
    expected = {x["file"]: x["sha256"] for x in previous}

    def fetch(spec):
        path = raw / spec["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(spec["url"])
                response.raise_for_status()
                path.write_bytes(response.content)
        file_hash = sha256(path)
        if spec["file"] in expected and expected[spec["file"]] != file_hash:
            raise ValueError(f"Source checksum changed: {spec['file']}")
        return {**spec, "sha256": file_hash, "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=6) as pool:
        lock = list(pool.map(fetch, source_specs()))
    write_json(lock_path, lock)
    return raw


def onehot(n, idx):
    return [float(i == idx) for i in range(n)]


def rank(seed, text):
    return digest(f"{seed}:{text}")


def make_record(source, source_id, split, task, state, question, target, label, **provenance):
    group = digest(normalize(state))
    return Record(
        id=f"{source}:{source_id}:{task}",
        group_id=group,
        source=source,
        source_id=str(source_id),
        split=split,
        task=task,
        state=state,
        question=Question.model_validate(question),
        target=target,
        label=label,
        provenance=provenance,
    )


def intent_description(label):
    return "The customer is asking about " + label.replace("_", " ") + "."


def intent_choice(text, label, pool, split, seed, include_other=True):
    rng = random.Random(rank(seed, normalize(text) + split))
    options = list(pool)
    # Training uses changing subsets. Held-out evaluation uses the complete pool.
    if split == "train":
        n = rng.choice([4, 8, 16])
        options = rng.sample([x for x in pool if x != label], min(n - 1, len(pool) - 1))
        options.append(label)
        if rng.random() < 0.1:
            options.remove(label)
            label = "other"
    if include_other:
        options.append("other")
    rng.shuffle(options)
    criteria = {x: intent_description(x) for x in options}
    if include_other:
        criteria["other"] = "None of the offered services matches the customer's request."
    return (
        {
            "type": "choice",
            "instructions": "Which service best matches the customer's request?",
            "criteria": criteria,
        },
        onehot(len(options), options.index(label)),
        label,
    )


def build_banking(raw: Path, seed):
    categories = json.loads((raw / "banking77/categories.json").read_text())
    heldout = sorted(categories, key=lambda x: rank(seed, x))[:15]
    seen = [x for x in categories if x not in heldout]
    records = []
    label_rows = defaultdict(list)
    with (raw / "banking77/train.csv").open() as f:
        for i, row in enumerate(csv.DictReader(f)):
            label_rows[row["category"]].append((str(i), row["text"]))
    all_rows = []
    for label in seen:
        rows = sorted(label_rows[label], key=lambda x: rank(seed, normalize(x[1])))
        n = len(rows)
        for i, (idx, text) in enumerate(rows):
            split = "train" if i < int(n * 0.75) else "dev" if i < int(n * 0.875) else "calibration"
            all_rows.append((f"train/{idx}", text, label, split, seen))
    with (raw / "banking77/test.csv").open() as f:
        for i, row in enumerate(csv.DictReader(f)):
            label = row["category"]
            split = "test_unseen" if label in heldout else "test_id"
            pool = heldout if label in heldout else seen
            all_rows.append((f"test/{i}", row["text"], label, split, pool))
    for idx, text, label, split, pool in all_rows:
        question, target, target_label = intent_choice(text, label, pool, split, seed)
        records.append(
            make_record(
                "banking77",
                idx,
                split,
                "intent_choice",
                text,
                question,
                target,
                target_label,
                original_label=label,
                candidate_protocol="subset_train_full_pool_eval",
            )
        )
        rng = random.Random(rank(seed + 1, idx))
        proposition_label = (
            label if rng.random() < 0.5 else rng.choice([x for x in pool if x != label])
        )
        phrase = proposition_label.replace("_", " ")
        question = {
            "type": "noul",
            "instructions": f"The customer is asking about {phrase}.",
            "criteria": {
                "false": f"The customer is not asking about {phrase}.",
                "true": f"The customer is asking about {phrase}.",
            },
        }
        records.append(
            make_record(
                "banking77",
                idx,
                split,
                "intent_verification",
                text,
                question,
                onehot(2, int(proposition_label == label)),
                str(proposition_label == label),
                original_label=label,
                proposition_label=proposition_label,
            )
        )
    return records, {"seen_labels": seen, "unseen_labels": heldout}


def build_sentiment(raw: Path, seed):
    records = []
    options = [
        "The text expresses negative sentiment.",
        "The text expresses neutral sentiment.",
        "The text expresses positive sentiment.",
    ]
    for official in ["train", "val", "test"]:
        texts = (raw / f"tweeteval_sentiment/{official}_text.txt").read_text().splitlines()
        labels = (raw / f"tweeteval_sentiment/{official}_labels.txt").read_text().splitlines()
        if len(texts) != len(labels):
            raise ValueError("TweetEval text/label row mismatch")
        groups = defaultdict(list)
        for i, (text, label) in enumerate(zip(texts, labels, strict=True)):
            groups[int(label)].append((i, text))
        for label, rows in groups.items():
            rows.sort(key=lambda x: rank(seed, normalize(x[1])))
            if official == "train":
                rows = rows[:1800]  # Predeclared balanced local-compute subset.
            for j, (i, text) in enumerate(rows):
                split = {"train": "train", "test": "test_id"}.get(official)
                if official == "val":
                    split = "dev" if j < len(rows) // 2 else "calibration"
                records.append(
                    make_record(
                        "tweeteval_sentiment",
                        f"{official}/{i}",
                        split,
                        "sentiment_score",
                        text,
                        {
                            "type": "score",
                            "instructions": "Rate the sentiment from negative to positive.",
                            "criteria": options,
                        },
                        onehot(3, label),
                        str(label),
                        original_split=official,
                        label_source="human annotation",
                    )
                )
    return records


def build_transfer(raw: Path, seed):
    data = json.loads((raw / "clinc150/data_full.json").read_text())
    categories = sorted({label for _, label in data["test"]})
    rows = defaultdict(list)
    for i, (text, label) in enumerate(data["test"]):
        rows[label].append((f"test/{i}", text))
    selected = []
    for label, values in rows.items():
        selected.extend(
            (idx, text, label)
            for idx, text in sorted(values, key=lambda x: rank(seed, normalize(x[1])))[:10]
        )
    selected.extend(
        (f"oos_test/{i}", text, "other")
        for i, (text, _) in sorted(
            enumerate(data["oos_test"]), key=lambda x: rank(seed, normalize(x[1][0]))
        )[:300]
    )
    records = []
    for idx, text, label in selected:
        question, target, target_label = intent_choice(
            text, label, categories, "test_transfer", seed
        )
        records.append(
            make_record(
                "clinc150",
                idx,
                "test_transfer",
                "intent_choice",
                text,
                question,
                target,
                target_label,
                candidate_protocol="full_150_intents_plus_other",
                note="Cross-dataset transfer; some intent meanings overlap BANKING77.",
            )
        )
    return records


def build_probability_simulator(seed):
    records = []
    colors = ["red", "blue", "green", "yellow", "white", "black"]
    for split, n in [("train", 1800), ("dev", 300), ("calibration", 300), ("test_simulator", 600)]:
        rng = random.Random(rank(seed, split))
        for i in range(n):
            k = rng.choice([2, 3, 4, 5, 6])
            chosen = rng.sample(colors, k)
            counts = [rng.randint(1, 60 if split == "test_simulator" else 30) for _ in chosen]
            inventory = ", ".join(f"{v} {c}" for c, v in zip(chosen, counts, strict=True))
            state = f"A bag contains {inventory} balls. One ball is drawn uniformly at random."
            probabilities = [v / sum(counts) for v in counts]
            question = {
                "type": "choice",
                "instructions": "What color will the randomly drawn ball be?",
                "criteria": {c: f"The drawn ball is {c}." for c in chosen},
            }
            records.append(
                make_record(
                    "urn_simulator",
                    f"{split}/{i}",
                    split,
                    "known_probability",
                    state,
                    question,
                    probabilities,
                    chosen[max(range(k), key=lambda x: counts[x])],
                    generator="urn-v1",
                    seed=seed,
                    target_source="exact counts / total",
                )
            )
    return records


def prepare(root: Path, seed=17):
    out = root / "data/processed"
    if (out / "manifest.json").exists():
        raise FileExistsError(
            "Prepared data exists; select a new project output or remove it deliberately."
        )
    raw = fetch_sources(root)
    bank, labels = build_banking(raw, seed)
    records = (
        bank
        + build_sentiment(raw, seed)
        + build_transfer(raw, seed)
        + build_probability_simulator(seed)
    )
    # Prioritize untouched tests. Remove conflicting source groups from earlier
    # stages, using text hashes only. Views within the same source row stay together.
    priority = {
        "test_transfer": 0,
        "test_unseen": 1,
        "test_id": 2,
        "test_simulator": 3,
        "calibration": 4,
        "dev": 5,
        "train": 6,
    }
    owners = {}
    kept, dropped = [], Counter()
    for r in sorted(records, key=lambda x: (priority[x.split], x.source, x.source_id, x.id)):
        owner = (r.split, r.source, r.source_id)
        if r.group_id not in owners:
            owners[r.group_id] = owner
        if owners[r.group_id] != owner:
            dropped[r.split] += 1
            continue
        kept.append(r)
    grouped = defaultdict(list)
    for r in kept:
        grouped[r.split].append(r)
    summaries = {}
    for split, rows in grouped.items():
        rows.sort(key=lambda x: rank(seed, x.id))
        path = out / f"{split}.jsonl"
        write_records(path, rows)
        summaries[split] = {
            "records": len(rows),
            "source_groups": len({r.group_id for r in rows}),
            "by_source": dict(Counter(r.source for r in rows)),
            "by_type": dict(Counter(r.question.type for r in rows)),
            "sha256": sha256(path),
        }
    manifest = {
        "version": "openjev-data-v0.1",
        "seed": seed,
        "protocol": "group-disjoint; development and calibration only from official non-test splits",
        "splits": summaries,
        "removed_duplicate_records": dict(dropped),
        "banking77": labels,
        "source_lock_sha256": sha256(root / "data/SOURCE_LOCK.json"),
        "local_only": True,
        "paid_teacher_calls": 0,
        "synthetic_source": "local exact-probability urn simulator; not LLM-generated",
        "limitations": [
            "English public-data pilot; general instruction following is not established.",
            "Unseen means not used in this project's post-training; pretraining contamination is unknown.",
            "CLINC uses 10 examples per intent plus 300 OOS; not the official full-test benchmark.",
            "TweetEval training is capped at 1800 examples per class; full official test used after deduplication.",
            "Soft-target simulator is reported separately from human-labeled semantic tasks.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    write_json(root / "reports/data_manifest.json", manifest)
    return manifest
