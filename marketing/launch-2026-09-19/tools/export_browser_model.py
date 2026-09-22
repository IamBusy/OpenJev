"""Export the published small model for live browser inference and compare outputs."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
LAUNCH = ROOT / "marketing/launch-2026-09-19"
sys.path.insert(0, str(LAUNCH / "space"))
from model import SceneModel, image_tensor  # noqa: E402 - use the standalone Space modules

OUT = LAUNCH / "space-static"
OUT.mkdir(exist_ok=True)
model = SceneModel(ROOT / "artifacts/openjev-vision-v0.1/synthetic-joint", device="cpu")


class Posterior(torch.nn.Module):
    def __init__(self, scene_model):
        super().__init__()
        self.net = scene_model.model
        self.temperature = scene_model.temperature

    def forward(self, pixels, prior):
        return self.net.log_posterior(self.net(pixels, prior), prior, self.temperature).exp()


wrapper = Posterior(model).eval()
pixels = torch.zeros(1, 3, 64, 192)
prior = torch.full((1, 64), 1 / 64)
torch.onnx.export(
    wrapper,
    (pixels, prior),
    str(OUT / "model.onnx"),
    input_names=["pixels", "prior"],
    output_names=["posterior"],
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(OUT / "model.onnx"))
session = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
examples = []
checks = []
for index in (0, 1):
    image = Image.open(ROOT / f"examples/vision/scene-{index}.png").convert("RGB")
    prior = np.array(
        json.loads((ROOT / f"examples/vision/scene-{index}-prior.json").read_text()),
        dtype=np.float64,
    )
    prior = (prior / prior.sum()).astype(np.float32)
    got = session.run(
        None, {"pixels": image_tensor(np.asarray(image)).numpy(), "prior": prior[None]}
    )[0][0]
    expected = model.posterior(image, prior)
    delta = float(np.max(np.abs(got - expected)))
    assert delta < 1e-5, delta
    examples.append({"image": f"scene-{index}.png", "prior": prior.tolist()})
    checks.append({"view": index, "max_posterior_difference": delta})
    shutil.copyfile(ROOT / f"examples/vision/scene-{index}.png", OUT / f"scene-{index}.png")
(OUT / "demo-data.json").write_text(
    json.dumps({"examples": examples}, separators=(",", ":")) + "\n"
)
provenance = {
    "source_commit": "4fa973e5212881d704def6818b0def190b094f12",
    "model_repository": "IamBusy/OpenJev-Vision",
    "model_revision": "8cf6cbd39a7dc72840c9b52810a3f672786566b0",
    "checkpoint": "synthetic-joint seed17, published calibration included",
    "format": "ONNX opset17, float32, no quantization",
    "onnx_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(),
    "equivalence_checks": checks,
}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
shutil.copyfile(ROOT / "LICENSE", OUT / "LICENSE")
for name in (
    "cover-en.png",
    "cover-zh.png",
    "demo-en.gif",
    "demo-zh.gif",
    "research-result.png",
    "openjev-vision-42s.mp4",
):
    shutil.copyfile(LAUNCH / "assets" / name, OUT / name)
print(json.dumps(provenance, indent=2))
