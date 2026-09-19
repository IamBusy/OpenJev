import json

import numpy as np
import torch

from openjev.io import digest, sha256
from openjev.schema import Question, Record
from openjev.train import train


def test_cache_hit_and_miss_have_identical_training_rng(tmp_path, monkeypatch):
    """Simulate encoder loading consuming RNG on a miss; saved heads must match."""
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("test fixture lock")
    for split in ["train", "dev"]:
        (tmp_path / f"data/processed/{split}.jsonl").write_text("test fixture")
    config = {
        "seed": 17,
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "hidden_dim": 8,
        "dropout": 0.1,
        "base_scale": 20.0,
        "patience": 3,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    rng = np.random.default_rng(44)
    vectors = rng.normal(size=(12, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    arrays = {
        "embeddings": vectors,
        "query_index": np.arange(8),
        "option_index": np.tile([10, 11], (8, 1)),
        "mask": np.ones((8, 2), dtype=bool),
        "targets": np.tile([[1.0, 0.0], [0.0, 1.0]], (4, 1)).astype(np.float32),
        "types": np.zeros(8, dtype=np.int64),
    }
    simulate_miss = [True]

    def fake_cache(paths, config, cache_dir, device):
        if simulate_miss[0]:
            torch.randn(5000)  # Model constructor work before pretrained weights load.
        split = paths[0].stem
        rows = [
            Record(
                id=f"{split}/{i}",
                group_id=digest(f"{split}/{i}"),
                source="test",
                source_id=str(i),
                split=split,
                task="choice",
                state=f"{split} {i}",
                question=Question(
                    type="choice", instructions="Choose", criteria={"a": "one", "b": "two"}
                ),
                target=[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0],
                label="a" if i % 2 == 0 else "b",
            )
            for i in range(8)
        ]
        return arrays, rows, tmp_path / "fixture.npz"

    monkeypatch.setattr("openjev.train.cache_features", fake_cache)
    train(tmp_path, config_path, tmp_path / "miss")
    simulate_miss[0] = False
    train(tmp_path, config_path, tmp_path / "hit")
    assert sha256(tmp_path / "miss/checkpoint/head.safetensors") == sha256(
        tmp_path / "hit/checkpoint/head.safetensors"
    )
