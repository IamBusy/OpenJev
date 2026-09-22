# Vendored from OpenJev-Vision commit 4fa973e5212881d704def6818b0def190b094f12.
# Only local import paths are changed. Licensed Apache-2.0.
"""Small learned image models; all inference inputs are pixels and a public prior."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from world import WORLDS, normalize_prior

from data import sha256, write_json


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def image_tensor(images, device="cpu"):
    x = torch.as_tensor(np.asarray(images).copy(), device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or tuple(x.shape[1:]) != (64, 192, 3):
        raise ValueError("Synthetic-scene images must have shape (64,192,3)")
    return x.permute(0, 3, 1, 2).float() / 255.0


class VisualEncoder(nn.Module):
    def __init__(self, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(192, hidden_dim),
            nn.GELU(),
        )

    def forward(self, images):
        b = images.shape[0]
        patches = images.reshape(b, 3, 64, 3, 64).permute(0, 3, 1, 2, 4)
        return self.net(patches.reshape(b * 3, 3, 64, 64)).reshape(b, 3, -1)


class VisualPosterior(nn.Module):
    def __init__(self, variant="joint", hidden_dim=96, sensor_error=0.08):
        super().__init__()
        if variant not in {"joint", "independent", "evidence"}:
            raise ValueError("Unknown model variant")
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.sensor_error = sensor_error
        self.encoder = VisualEncoder(hidden_dim)
        self.register_buffer("worlds", torch.tensor(WORLDS, dtype=torch.float32))
        if variant == "evidence":
            self.head = nn.Linear(hidden_dim, 6)
        else:
            self.prior_encoder = nn.Sequential(nn.Linear(64, 64), nn.GELU())
            self.head = nn.Sequential(
                nn.Linear(3 * hidden_dim + 64, 128),
                nn.GELU(),
                nn.Linear(128, 64 if variant == "joint" else 6),
            )

    def forward(self, images, priors):
        z = self.encoder(images)
        if self.variant == "evidence":
            return self.head(z).reshape(-1, 6, 3)
        context = self.prior_encoder(priors.clamp_min(1e-12).log() / 10)
        return self.head(torch.cat((z.flatten(1), context), dim=-1))

    def log_posterior(self, outputs, priors, temperature=1.0):
        if self.variant == "joint":
            return F.log_softmax(outputs / temperature, dim=-1)
        if self.variant == "independent":
            logits = outputs / temperature
            return (
                F.logsigmoid(logits)[:, None, :] * self.worlds[None]
                + F.logsigmoid(-logits)[:, None, :] * (1 - self.worlds[None])
            ).sum(-1)
        observation = outputs.softmax(-1)
        # Observation categories: 0, 1, hidden. Hidden contributes likelihood one.
        likelihood0 = (
            observation[..., 2]
            + (1 - self.sensor_error) * observation[..., 0]
            + self.sensor_error * observation[..., 1]
        )
        likelihood1 = (
            observation[..., 2]
            + (1 - self.sensor_error) * observation[..., 1]
            + self.sensor_error * observation[..., 0]
        )
        ll = (
            likelihood1.clamp_min(1e-12).log()[:, None, :] * self.worlds[None]
            + likelihood0.clamp_min(1e-12).log()[:, None, :] * (1 - self.worlds[None])
        ).sum(-1)
        return F.log_softmax(priors.clamp_min(1e-30).log() + ll, dim=-1)


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    write_json(
        path / "config.json",
        {
            "variant": model.variant,
            "hidden_dim": model.hidden_dim,
            "sensor_error": model.sensor_error,
            "format_version": 1,
            **metadata,
            "weights_sha256": sha256(path / "model.safetensors"),
        },
    )


class SceneModel:
    def __init__(self, checkpoint, device="auto"):
        self.path = Path(checkpoint)
        self.config = json.loads((self.path / "config.json").read_text())
        weights = self.path / "model.safetensors"
        if sha256(weights) != self.config["weights_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        self.device = device_for(device)
        self.model = VisualPosterior(
            **{key: self.config[key] for key in ("variant", "hidden_dim", "sensor_error")}
        )
        self.model.load_state_dict(load_file(str(weights)))
        self.model.to(self.device).eval()
        calibration = self.path / "calibration.json"
        self.temperature = (
            json.loads(calibration.read_text())["temperature"] if calibration.exists() else 1.0
        )

    @torch.inference_mode()
    def posterior(self, image, prior=None, calibrated=True):
        p = normalize_prior(np.ones(64) if prior is None else prior)
        tensor = image_tensor(np.asarray(image), self.device)
        prior_tensor = torch.tensor(p[None], device=self.device, dtype=torch.float32)
        raw = self.model(tensor, prior_tensor)
        logp = self.model.log_posterior(raw, prior_tensor, self.temperature if calibrated else 1.0)
        return logp.exp().cpu().numpy()[0].astype(np.float64)
