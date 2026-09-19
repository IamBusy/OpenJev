import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="OpenJev reproducible decision-model pilot")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("audit")
    synthesis = sub.add_parser("synthesize")
    synthesis.add_argument("--model", default="deepseek-v4-pro")
    synthesis.add_argument("--batches", type=int, default=24)
    synthesis.add_argument("--workers", type=int, default=3)
    synthesis.add_argument(
        "--offline", action="store_true", help="Replay saved teacher responses without API calls"
    )
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, default=Path("configs/local.json"))
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--augmented", action="store_true")
    training.add_argument("--device", default="auto")
    selection = sub.add_parser("freeze-selection")
    selection.add_argument("--runs", type=Path, nargs="+", required=True)
    selection.add_argument("--selection-file")
    for name in ["calibrate", "evaluate", "benchmark", "predict", "serve", "export"]:
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--device", default="auto")
        if name == "evaluate":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--selection-file")
        if name == "benchmark":
            p.add_argument("--repeats", type=int, default=20)
        if name == "predict":
            p.add_argument("--input", type=Path, required=True)
        if name == "serve":
            p.add_argument("--port", type=int, default=8080)
        if name == "export":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def full(path):
        return path if path.is_absolute() else root / path

    if args.command == "prepare":
        from .data import prepare

        result = prepare(root)
    elif args.command == "audit":
        from .audit import audit

        result = audit(root)
    elif args.command == "synthesize":
        from .synthesize import synthesize

        result = synthesize(
            root, model=args.model, batches=args.batches, workers=args.workers, offline=args.offline
        )
    elif args.command == "train":
        from .train import train

        result = train(root, full(args.config), full(args.run), args.augmented, args.device)
    elif args.command == "freeze-selection":
        from .evaluate import freeze_selection

        result = freeze_selection(root, [full(p) for p in args.runs], args.selection_file)
    elif args.command == "calibrate":
        from .train import calibrate

        result = calibrate(root, full(args.checkpoint), args.device)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            root, full(args.checkpoint), full(args.output), args.device, args.selection_file
        )
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(root, full(args.checkpoint), args.repeats, args.device)
    elif args.command == "predict":
        from .model import OpenJev

        request = json.loads(full(args.input).read_text())
        result = OpenJev(full(args.checkpoint), args.device).predict(**request)
    elif args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(
            create_app(full(args.checkpoint), args.device), host="127.0.0.1", port=args.port
        )
        return
    elif args.command == "export":
        from .release import export

        result = export(root, full(args.checkpoint), full(args.output), args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
