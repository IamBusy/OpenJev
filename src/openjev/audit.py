"""Independent checks of split identity, lineage and frozen source hashes."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import normalize
from .io import digest, read_records, sha256, write_json


def audit(root: Path):
    processed = root / "data/processed"
    manifest = json.loads((processed / "manifest.json").read_text())
    failures = []
    for name, expected in manifest["splits"].items():
        if sha256(processed / f"{name}.jsonl") != expected["sha256"]:
            failures.append(f"Modified prepared split: {name}")
    groups, surfaces = defaultdict(set), defaultdict(set)
    ids, counts, records_by_id = set(), Counter(), {}
    all_records = []
    for path in sorted(processed.glob("*.jsonl")):
        all_records.extend(read_records(path))
    for r in all_records:
        if r.id in ids:
            failures.append(f"Duplicate record ID: {r.id}")
        ids.add(r.id)
        records_by_id[r.id] = r
        groups[r.group_id].add(r.split)
        surfaces[digest(normalize(r.state))].add(r.split)
        counts[r.split] += 1
    for group, splits in groups.items():
        if len(splits) > 1:
            failures.append(f"Cross-split source lineage: {group}")
    for group, splits in surfaces.items():
        if len(splits) > 1:
            failures.append(f"Cross-split normalized text: {group}")
    unseen = set(manifest["banking77"]["unseen_labels"])
    for r in all_records:
        if r.source == "deepseek_synthetic":
            parent = records_by_id.get(r.provenance.get("source_record"))
            if (
                parent is None
                or parent.split != "train"
                or r.split != "train"
                or parent.group_id != r.group_id
            ):
                failures.append(f"Invalid synthetic lineage: {r.id}")
            if r.question != parent.question or r.target != parent.target:
                failures.append(f"Synthetic label/schema changed: {r.id}")
        if r.source == "banking77" and r.split in {"train", "dev", "calibration"}:
            if r.provenance.get("original_label") in unseen:
                failures.append(f"Unseen label used before evaluation: {r.id}")
            if r.question.type == "choice" and set(r.question.criteria) & unseen:
                failures.append(f"Unseen candidate used before evaluation: {r.id}")
    for source in json.loads((root / "data/SOURCE_LOCK.json").read_text()):
        path = root / "data/raw" / source["file"]
        if sha256(path) != source["sha256"]:
            failures.append(f"Raw source checksum mismatch: {source['file']}")
    report = {
        "passed": not failures,
        "records": len(all_records),
        "groups": len(groups),
        "records_by_effective_split": dict(counts),
        "failures": failures,
        "checks": [
            "source checksums",
            "prepared split checksums",
            "unique record IDs",
            "normalized text split isolation",
            "source lineage split isolation",
            "synthetic training-only parentage",
            "held-out label isolation",
        ],
    }
    write_json(root / "reports/data_audit.json", report)
    if failures:
        raise AssertionError(f"Data audit failed: {failures[:5]}")
    return report
