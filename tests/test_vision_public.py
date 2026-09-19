import io
import json
import zipfile
from itertools import product

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from openjev.vision.export_data import write_parquet
from openjev.vision.public_data import RangeReader
from openjev.vision.public_inference import answer_public, compile_public_event
from openjev.vision.public_model import AttributeHead, classification_metrics


@pytest.mark.parametrize("variant", ["joint", "independent", "binding"])
def test_attribute_distribution_and_gradients(variant):
    torch.manual_seed(9)
    model = AttributeHead([3, 4], variant, feature_dim=8, rank=2)
    features = torch.randn(5, 8)
    logp = model(features)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(5))
    (-logp[:, 2].mean()).backward()
    assert model.output.weight.grad.abs().sum() > 0
    if variant == "independent":
        p = logp.exp().reshape(5, 3, 4)
        torch.testing.assert_close(p, p.sum(2)[:, :, None] * p.sum(1)[:, None, :])
    if variant == "binding":
        assert model.context.weight.grad.abs().sum() > 0


def test_public_event_algebra():
    worlds = [{"color": c, "shape": s} for c, s in product(("red", "blue"), ("circle", "square"))]
    p = np.array([0.4, 0.1, 0.1, 0.4])
    qs = {
        "red": {"type": "noul", "event": {"color": "red"}},
        "not_red": {"type": "noul", "event": {"not": {"color": "red"}}},
        "both": {"type": "noul", "event": {"color": "red", "shape": "square"}},
        "color": {
            "type": "choice",
            "criteria": {"red": {"color": "red"}, "blue": {"color": "blue"}},
        },
    }
    r = answer_public(p, qs, worlds)
    assert r["red"]["noul"] + r["not_red"]["noul"] == pytest.approx(1)
    assert r["both"]["noul"] == pytest.approx(0.1)
    assert r["color"]["probabilities"] == {"red": 0.5, "blue": 0.5}
    assert answer_public(p, dict(reversed(list(qs.items()))), worlds) == r
    with pytest.raises(ValueError):
        compile_public_event({"color": "green"}, worlds)
    with pytest.raises(ValueError):
        answer_public(
            p,
            {
                "x": {
                    "type": "choice",
                    "criteria": {
                        "red": {"color": "red"},
                        "square": {"shape": "square"},
                    },
                }
            },
            worlds,
        )


def test_derived_species_is_same_breed_distribution():
    worlds = [
        {"breed": "persian", "species": "cat"},
        {"breed": "siamese", "species": "cat"},
        {"breed": "beagle", "species": "dog"},
    ]
    out = answer_public(
        [0.2, 0.3, 0.5],
        {
            "cat": {"type": "noul", "event": {"species": "cat"}},
            "breeds": {
                "type": "choice",
                "criteria": {
                    "persian": {"breed": "persian"},
                    "siamese": {"breed": "siamese"},
                    "beagle": {"breed": "beagle"},
                },
            },
        },
        worlds,
    )
    assert out["cat"]["noul"] == pytest.approx(
        out["breeds"]["probabilities"]["persian"] + out["breeds"]["probabilities"]["siamese"]
    )


def test_parquet_preserves_image_and_metadata(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(buffer, format="PNG")
    raw = buffer.getvalue()
    row = {
        "id": "source:1",
        "image": {"bytes": raw, "path": None},
        "questions_json": json.dumps([{"target": 0.5}]),
    }
    file = tmp_path / "test.parquet"
    result = write_parquet(file, [row])
    assert result["rows"] == 1
    table = pq.read_table(file)
    assert table.to_pylist()[0] == row
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    assert metadata["info"]["features"]["image"]["_type"] == "Image"


def test_range_zip_member_and_crc():
    raw = b"original pixel data" * 500
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("image.png", raw)
    archive_bytes = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        info = archive.getinfo("image.png")

    class MemoryRange:
        def range(self, offset, length):
            return archive_bytes[offset : offset + length]

    assert RangeReader.member(MemoryRange(), info) == raw
    info.CRC ^= 1
    with pytest.raises(ValueError, match="integrity"):
        RangeReader.member(MemoryRange(), info)


def test_metrics_handle_perfect_predictions_without_nan():
    r = classification_metrics(np.eye(3), np.arange(3))
    assert r["accuracy"] == 1
    assert r["brier"] < 1e-20
    assert np.isfinite(r["nll"])
    assert sum(b["n"] for b in r["reliability"]) == 3
