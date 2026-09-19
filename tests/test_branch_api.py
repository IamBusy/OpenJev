import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_branch_api_reuses_state_and_validates_output_offline(monkeypatch):
    pytest.importorskip("peft")
    from openjev.branch_cli import create_branch_app

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts/openjev-branch-v0.3"
    if not checkpoint.exists():
        pytest.skip("Exported branch model is a local artifact")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    client = TestClient(create_branch_app(root, checkpoint))
    request = json.loads((root / "examples/refund.json").read_text())
    response = client.post("/v1/decide", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["usage"]["prefix_evaluations"] == 1
    assert result["usage"]["questions"] == 3
    assert result["usage"]["candidate_branches"] == 9
    for name, answer in result["answers"].items():
        if "probabilities" in answer:
            assert abs(sum(answer["probabilities"].values()) - 1) < 1e-8
        if "choice" in answer:
            assert answer["choice"] in request["questions"][name]["criteria"]
    invalid = {
        "state": "test",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {str(i): f"Option {i}" for i in range(256)},
            }
        },
    }
    assert client.post("/v1/decide", json=invalid).status_code == 422
