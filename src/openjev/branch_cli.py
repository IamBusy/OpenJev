import argparse
import json
import shutil
from pathlib import Path
from threading import Lock

from .io import sha256, write_json


def export_branch(root, output):
    selection = json.loads((root / "reports/v03/TRAINING_SELECTION.json").read_text())
    source = root / selection["selected_checkpoint"]
    if output.exists():
        raise FileExistsError("Export directory already exists")
    shutil.copytree(source, output)
    adapter_config = json.loads((output / "adapter/adapter_config.json").read_text())
    adapter_config["base_model_name_or_path"] = selection["config"]["model_name"]
    adapter_config["revision"] = selection["config"]["model_revision"]
    write_json(output / "adapter/adapter_config.json", adapter_config)
    cfg = json.loads((output / "openjev_config.json").read_text())
    cfg["architecture"] = "SharedStateCandidateBranches"
    write_json(output / "openjev_config.json", cfg)
    manifest = {
        "base_model": selection["config"]["model_name"],
        "base_revision": selection["config"]["model_revision"],
        "base_weights_included": False,
        "source_checkpoint": selection["selected_checkpoint"],
        "files": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
    }
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def create_branch_app(root, checkpoint):
    from fastapi import FastAPI, HTTPException

    from .branch_model import BranchDecision
    from .schema import Request

    model = BranchDecision(root, checkpoint=checkpoint)
    calibration = checkpoint / "calibration-v03.json"
    temperatures = json.loads(calibration.read_text()) if calibration.exists() else None
    lock = Lock()
    app = FastAPI(title="OpenJev shared-state scorer", version="0.3.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "OpenJev-Branch-v0.3"}

    @app.post("/v1/decide")
    def decide(request: Request):
        try:
            with lock:
                return model.predict(request.state, request.questions, temperatures)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser(description="OpenJev shared-state candidate scorer")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download pinned base and release weights")
    download.add_argument("--base-only", action="store_true")
    prepare = commands.add_parser("prepare")
    rendering = prepare.add_mutually_exclusive_group()
    rendering.add_argument("--replay-rendering", action="store_true")
    rendering.add_argument("--live-rendering", action="store_true", help="Call DeepSeek (paid)")
    train = commands.add_parser("train")
    train.add_argument("--run", default="branch-v03-seed43")
    commands.add_parser("evaluate")
    commands.add_parser("reference")
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    for name in ["predict", "serve"]:
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", type=Path)
        if name == "predict":
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--recompute-state", action="store_true")
        else:
            p.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    root = args.root.resolve()

    def full(path):
        return path if path.is_absolute() else root / path

    if args.command == "download":
        from .download import download_branch

        result = download_branch(root, base_only=args.base_only)
    elif args.command == "prepare":
        from .data_v03 import prepare_v03

        result = prepare_v03(root, replay=args.replay_rendering, live_rendering=args.live_rendering)
    elif args.command == "train":
        from .branch_train import train_branch

        result = train_branch(root, args.run)
    elif args.command == "evaluate":
        from .branch_eval import evaluate_branch

        result = evaluate_branch(root)
    elif args.command == "reference":
        from .reference_v02 import run_reference

        result = run_reference(
            root,
            data_dir=root / "data/v03/processed",
            report_dir=root / "reports/v03",
            cache_dir=root / "artifacts/reference-v03",
        )
    elif args.command == "export":
        result = export_branch(root, full(args.output))
    else:
        if args.checkpoint:
            checkpoint = full(args.checkpoint)
        else:
            checkpoint = root / "artifacts/openjev-branch-v0.3"
        if not (checkpoint / "openjev_config.json").is_file():
            parser.error("Checkpoint missing. Run openjev-branch download or pass --checkpoint.")
        if args.command == "serve":
            import uvicorn

            uvicorn.run(create_branch_app(root, checkpoint), host="127.0.0.1", port=args.port)
            return
        from .branch_model import BranchDecision

        model = BranchDecision(root, checkpoint=checkpoint)
        path = checkpoint / "calibration-v03.json"
        temperatures = json.loads(path.read_text()) if path.exists() else None
        request = json.loads(full(args.input).read_text())
        result = model.predict(
            **request, temperatures=temperatures, cached=not args.recompute_state
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
