import json
import zipfile

import pytest

from openjev.download import install_archive
from openjev.io import digest, sha256


def archive(tmp_path, entries):
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as output:
        for name, value in entries.items():
            output.writestr(name, value)
    return path


def test_release_install_checks_hash_and_does_not_replace_existing_files(tmp_path):
    path = archive(
        tmp_path,
        {
            "head.safetensors": "fixture",
            "MANIFEST.json": json.dumps({"files": {"head.safetensors": digest("fixture")}}),
        },
    )
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="checksum"):
        install_archive(path, target, "wrong")
    assert not target.exists()
    install_archive(path, target, sha256(path))
    assert (target / "head.safetensors").read_text() == "fixture"
    with pytest.raises(FileExistsError):
        install_archive(path, target, sha256(path))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../../escape"])
def test_release_install_rejects_archive_traversal(tmp_path, name):
    path = archive(tmp_path, {name: "bad"})
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="Unsafe"):
        install_archive(path, target, sha256(path))
    assert not target.exists()
