import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openjev.server import create_app


def test_exported_model_serves_validated_decisions_offline(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts/openjev-minilm-v0.1"
    if not checkpoint.exists():
        pytest.skip("Exported checkpoint is not in a source-only checkout")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    client = TestClient(create_app(checkpoint))
    assert client.get("/health").status_code == 200
    request = json.loads((root / "examples/refund.json").read_text())
    response = client.post("/v1/decide", json=request)
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["department"]["choice"] == "duplicate_payment"
    assert abs(sum(answers["department"]["probabilities"].values()) - 1) < 1e-6
    assert 0 <= answers["duplicate_charge"]["noul"] <= 1
    assert 0 <= answers["sentiment"]["score"] <= 2
    invalid = {
        "state": "hello",
        "questions": {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "one"}}},
    }
    assert client.post("/v1/decide", json=invalid).status_code == 422
