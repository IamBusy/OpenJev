"""Check the source inventory before packaging or publication."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".env", ".venv", "artifacts", "runs", "dist", "__pycache__"}
PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}"),
    re.compile(rb"\bsk-[A-Za-z0-9]{24,}\b"),
]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).split(b"\0")
    checked = 0
    for raw in names:
        if not raw:
            continue
        relative = Path(raw.decode())
        path = ROOT / relative
        if not path.is_file():
            continue
        if any(part in FORBIDDEN for part in relative.parts):
            raise SystemExit(f"Private/generated path in source inventory: {relative}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise SystemExit(f"Unexpected large source file: {relative}")
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in PATTERNS):
            raise SystemExit(f"Potential credential in {relative}; value intentionally suppressed")
        if path.suffix in {".md", ".json", ".py", ".yml", ".toml"}:
            if re.search(rb"/(?:Users|home)/[^/\s]+/", data):
                raise SystemExit(f"Machine-specific absolute path: {relative}")
        checked += 1
    print(f"Source hygiene passed: {checked} files")


if __name__ == "__main__":
    main()
