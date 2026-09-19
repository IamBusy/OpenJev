"""Public model identity, separate from implementation and experiment names."""

from pathlib import Path

MODEL_ID = "OpenJev-0.6B"
MODEL_VERSION = "0.3.0"
HUB_MODEL_ID = "IamBusy/OpenJev-0.6B"
CHECKPOINT_DIRECTORY = Path("artifacts/openjev-0.6b")
LEGACY_CHECKPOINT_DIRECTORY = Path("artifacts/openjev-branch-v0.3")


def default_checkpoint(root: Path) -> Path:
    canonical = root / CHECKPOINT_DIRECTORY
    legacy = root / LEGACY_CHECKPOINT_DIRECTORY
    if (
        not (canonical / "openjev_config.json").is_file()
        and (legacy / "openjev_config.json").is_file()
    ):
        return legacy
    return canonical
