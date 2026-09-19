import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from .schema import Record


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_records(path: str | Path) -> list[Record]:
    return [Record.model_validate_json(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_records(path: str | Path, records: list[Record]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware() -> dict:
    import torch

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
