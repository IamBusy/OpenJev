"""Download the pinned public release; never execute archive contents."""

import json
import shutil
import stat
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath

import httpx

from .io import sha256


def verify_bundle(folder):
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe manifest path")
        if sha256(folder / name) != expected:
            raise ValueError(f"Model file checksum mismatch: {name}")


def install_archive(archive, destination, expected_sha256):
    """Validate hash, bound extraction, and publish only a complete bundle."""
    if sha256(archive) != expected_sha256:
        raise ValueError("Release archive checksum mismatch")
    if destination.exists():
        raise FileExistsError("Destination already exists; no files were replaced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        folder = Path(temp) / "bundle"
        folder.mkdir()
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if sum(info.file_size for info in members) > 32 * 1024 * 1024:
                raise ValueError("Release archive exceeds the size limit")
            names = set()
            for info in members:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.filename in names
                ):
                    raise ValueError("Unsafe release archive path")
                names.add(info.filename)
                target = folder / path
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as stream, target.open("wb") as output:
                        shutil.copyfileobj(stream, output)
        verify_bundle(folder)
        folder.rename(destination)


def download_branch(root, base_only=False):
    from huggingface_hub import snapshot_download

    metadata = json.loads(files("openjev").joinpath("release_assets.json").read_text())
    base = snapshot_download(
        metadata["base_model"],
        revision=metadata["base_revision"],
        local_dir=root / "artifacts/base/qwen3-0.6b",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
    )
    result = {"base": str(base), "base_revision": metadata["base_revision"]}
    if base_only:
        return result
    destination = root / "artifacts/openjev-branch-v0.3"
    if destination.exists():
        verify_bundle(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            archive = Path(temp) / "release.zip"
            with httpx.stream(
                "GET", metadata["url"], follow_redirects=True, timeout=120
            ) as response:
                response.raise_for_status()
                size = 0
                with archive.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 32 * 1024 * 1024:
                            raise ValueError("Download exceeds the release size limit")
                        output.write(chunk)
            install_archive(archive, destination, metadata["sha256"])
    return {**result, "checkpoint": str(destination), "verified": True}
