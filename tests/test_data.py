from pathlib import Path

import pytest

from openjev.audit import audit


def test_real_data_has_no_split_or_lineage_leakage():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/processed/manifest.json").exists():
        pytest.skip("Run data preparation to enable the real-data audit")
    assert audit(root)["passed"]
