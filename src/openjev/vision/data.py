"""Deterministic case-grouped synthetic data with exact posterior targets."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .render import render_scene
from .world import WORLDS, posterior, sample_prior

SPLIT_NAMES = ("train", "dev", "calibration", "test_id", "test_topology", "test_appearance")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def build_split(episodes, seed, split, sensor_error):
    rng = np.random.default_rng(seed)
    result = {
        key: []
        for key in ("images", "priors", "targets", "observed", "truth", "groups", "visibility")
    }
    families, render_seeds = [], []
    for episode in range(episodes):
        family = "cross" if split == "test_topology" else ("local", "chain")[episode % 2]
        prior = sample_prior(rng, family)
        truth = int(rng.choice(len(WORLDS), p=prior))
        observation = WORLDS[truth] ^ (rng.random(6) < sensor_error)
        visible = rng.random(3) < 0.5
        if visible.all():
            visible[int(rng.integers(3))] = False
        appearance_seed = int(rng.integers(0, 2**31))
        for mask in (visible, np.ones(3, dtype=bool)):
            observed = np.where(np.repeat(mask, 2), observation, -1)
            image = render_scene(
                observed,
                appearance_seed,
                "shift" if split == "test_appearance" else "default",
            )
            for key, value in {
                "images": np.asarray(image),
                "priors": prior.astype(np.float32),
                "targets": posterior(prior, observed, sensor_error).astype(np.float32),
                "observed": observed.astype(np.int8),
                "truth": truth,
                "groups": f"{split}:{episode}",
                "visibility": int(mask.sum()),
            }.items():
                result[key].append(value)
            families.append(family)
            render_seeds.append(appearance_seed)
    arrays = {key: np.asarray(value) for key, value in result.items()}
    arrays["families"] = np.asarray(families)
    arrays["render_seeds"] = np.asarray(render_seeds, dtype=np.int64)
    return arrays


def generate(output, config, *, overwrite=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Data directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    manifest = {
        "version": "scenebelief-data-v0.1",
        "config": config,
        "worlds": len(WORLDS),
        "source_sha256": {
            name: sha256(source / name) for name in ("world.py", "render.py", "data.py")
        },
        "splits": {},
    }
    seen_hashes = {}
    for i, split in enumerate(SPLIT_NAMES):
        count = config.get(f"{split}_episodes", config["test_episodes"])
        seed = config["data_seed"] + (i + 1) * 100_003
        arrays = build_split(count, seed, split, config["sensor_error"])
        for pixels in arrays["images"]:
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError("Exact image duplicate across splits")
            seen_hashes[digest] = split
        path = output / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        manifest["splits"][split] = {
            "episodes": count,
            "images": len(arrays["images"]),
            "seed": seed,
            "sha256": sha256(path),
            "families": sorted(set(arrays["families"].tolist())),
        }
        print(json.dumps({"generated": split, "images": len(arrays["images"])}), flush=True)
    manifest["audit"] = {"cross_split_image_duplicates": 0, "group_disjoint": True}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory, split, verify=True):
    if split not in SPLIT_NAMES:
        raise ValueError("Unknown split")
    directory = Path(directory)
    path = directory / f"{split}.npz"
    if verify:
        manifest = json.loads((directory / "manifest.json").read_text())
        if sha256(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Data hash mismatch for {split}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
