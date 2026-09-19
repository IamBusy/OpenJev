from pathlib import Path

import pytest

from openjev.io import sha256
from openjev.synthesize import synthesize


def test_teacher_snapshots_replay_without_key_or_paid_calls(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    target = root / "data/processed/synthetic_train.jsonl"
    snapshots = root / "artifacts/synthesis/openjev-paraphrase-v1-deepseek-v4-pro-17"
    if not target.exists() or not snapshots.exists():
        pytest.skip("Recorded teacher snapshots are not part of a source-only checkout")

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted a provider call")

    monkeypatch.setattr("openjev.synthesize.call_teacher", forbidden)
    monkeypatch.setattr("openjev.synthesize.credentials", forbidden)
    before = sha256(target)
    summary = synthesize(root, offline=True)
    assert summary["accepted_records"] == 176
    assert sha256(target) == before
