import json
import shutil
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .model import OpenJev


def export(root: Path, checkpoint: Path, output: Path, device="auto"):
    if output.exists():
        raise FileExistsError("Export destination already exists")
    output.mkdir(parents=True)
    model = OpenJev(checkpoint, device)
    for name in ["head.safetensors", "config.json", "calibration.json"]:
        shutil.copy2(checkpoint / name, output / name)
    model.encoder.save_pretrained(str(output / "backbone"))
    config = json.loads((output / "config.json").read_text())
    config["release_name"] = "OpenJev-MiniLM-v0.1"
    write_json(output / "config.json", config)
    for name in ["LICENSE", "THIRD_PARTY.md"]:
        if (root / name).exists():
            shutil.copy2(root / name, output / name)
    shutil.copy2(checkpoint.parent / "run.json", output / "training_run.json")
    request = json.loads((root / "examples/refund.json").read_text())
    before = model.predict(**request)
    restored = OpenJev(output, device)
    after = restored.predict(**request)
    errors = []
    for name, answer in before["answers"].items():
        if "noul" in answer:
            errors.append(abs(answer["noul"] - after["answers"][name]["noul"]))
        else:
            errors.extend(
                abs(value - after["answers"][name]["probabilities"][key])
                for key, value in answer["probabilities"].items()
            )
    if max(errors) > 1e-5:
        raise AssertionError("Export changed text-inference predictions")
    backbone_parameters = sum(p.numel() for p in restored.encoder.parameters())
    head_parameters = sum(p.numel() for p in restored.head.parameters())
    result = {
        "release_name": config["release_name"],
        "source_checkpoint": str(checkpoint.relative_to(root)),
        "backbone_parameters": backbone_parameters,
        "trainable_head_parameters": head_parameters,
        "total_parameters": backbone_parameters + head_parameters,
        "export_reload_max_probability_error": float(np.max(errors)),
        "self_contained_weights": True,
        "files": {
            str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "EXPORT_MANIFEST.json", result)
    write_json(root / "reports/export.json", result)
    write_json(root / "reports/example_prediction.json", after)
    return result
