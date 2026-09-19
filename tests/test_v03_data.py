import json
from collections import defaultdict
from pathlib import Path

import pytest

from openjev.data_v03 import world_records
from openjev.io import read_records, sha256


def choice(rows, task):
    record = next(r for r in rows if r.task == task)
    return record.question.options()[0][record.target.index(1.0)]


@pytest.mark.parametrize(
    "days,opened,damaged,expected",
    [
        (3, False, False, "yes"),
        (40, None, False, "no"),
        (3, None, False, "unknown"),
        (3, False, True, "no"),
        (3, True, False, "no"),
    ],
)
def test_refund_oracle_does_not_equate_missing_evidence_with_half_probability(
    days, opened, damaged, expected
):
    world = {
        "id": "test",
        "group_id": "test-group",
        "split": "train",
        "family": "refund",
        "language": "en",
        "text": "Test observations",
        "render_provenance": {"method": "unit fixture"},
        "facts": {
            "days_since_purchase": days,
            "return_window_days": 14,
            "opened": opened,
            "damaged": damaged,
            "refund_requested": True,
        },
    }
    records = world_records(world)
    assert choice(records, "refund_eligibility") == expected
    assert len({r.state for r in records}) == 1
    assert all(r.group_id == "test-group" for r in records)
    assert "refund_requested" not in json.loads(records[0].state)["facts"]


def test_shipping_rule_precedence():
    world = {
        "id": "test",
        "group_id": "g",
        "split": "test_fresh",
        "family": "shipping",
        "language": "zh",
        "text": "观察",
        "render_provenance": {},
        "facts": {"shipped": False, "damaged": True, "days_late": 12, "trace_threshold": 3},
    }
    assert choice(world_records(world), "shipping_action") == "replace"


def test_v03_fresh_pack_and_group_isolation():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/v03/processed"
    if not (folder / "manifest.json").exists():
        pytest.skip("Local v0.3 data missing")
    manifest = json.loads((folder / "manifest.json").read_text())
    groups = defaultdict(set)
    surfaces = defaultdict(set)
    for split in ["train", "dev", "calibration", "test_fresh"]:
        assert sha256(folder / f"{split}.jsonl") == manifest["splits"][split]["sha256"]
        for record in read_records(folder / f"{split}.jsonl"):
            groups[record.group_id].add(split)
            from openjev.data import normalize

            surfaces[normalize(record.state)].add(split)
            if split == "train":
                assert record.provenance.get("family") not in {"access", "shipping"}
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(len(splits) == 1 for splits in surfaces.values())
    previous = {r.group_id for r in read_records(root / "data/v02/processed/test.jsonl")}
    fresh = read_records(folder / "test_fresh.jsonl")
    assert not previous & {r.group_id for r in fresh}
    ids = {r.id for r in fresh}
    assert {r.id for r in read_records(folder / "reference.jsonl")} <= ids
    labels = defaultdict(set)
    for r in fresh:
        labels[r.task].add(r.label)
    assert labels["refund_eligibility"] == {"yes", "no", "unknown"}
    assert labels["incident_severity"] == {"0", "1", "2"}
    assert labels["shipping_action"] == {"replace", "warehouse", "trace", "wait"}
    assert labels["access_action"] == {"allow", "deny", "approval"}
