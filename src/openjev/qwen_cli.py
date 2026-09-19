import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Qwen-based OpenJev v0.2")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("prepare")
    train = sub.add_parser("train")
    train.add_argument("--run", default="qwen-v02-seed29")
    sub.add_parser("evaluate")
    sub.add_parser("reference")
    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--native", action="store_true")
    predict.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        config = json.loads((root / "configs/qwen-v02.json").read_text())
        location = snapshot_download(
            config["model_name"],
            revision=config["model_revision"],
            local_dir=root / "artifacts/base/qwen3-0.6b",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
        )
        result = {"downloaded_to": str(location), "revision": config["model_revision"]}
    elif args.command == "prepare":
        from .data_v02 import prepare_v02

        result = prepare_v02(root)
    elif args.command == "train":
        from .qwen_train import train_qwen

        result = train_qwen(root, args.run)
    elif args.command == "evaluate":
        from .eval_v02 import run_evaluation

        result = run_evaluation(root)
    elif args.command == "reference":
        from .reference_v02 import run_reference

        result = run_reference(root)
    else:
        from .qwen import QwenDecision

        config = json.loads((root / "configs/qwen-v02.json").read_text())
        if args.native:
            adapter = None
        elif args.adapter:
            adapter = args.adapter if args.adapter.is_absolute() else root / args.adapter
        else:
            selection = json.loads((root / "reports/v02/TRAINING_SELECTION.json").read_text())
            adapter = root / selection["selected_adapter"]
        runner = QwenDecision(root, config, adapter=adapter)
        comparison = root / "reports/v02/comparison.json"
        temps = None
        from .io import sha256

        selection_file = root / "reports/v02/TRAINING_SELECTION.json"
        matched_adapter = args.native or (
            selection_file.exists()
            and sha256(adapter / "adapter_model.safetensors")
            == json.loads(selection_file.read_text())["adapter_sha256"]
        )
        if comparison.exists() and matched_adapter:
            name = "qwen3_0.6b_native" if args.native else "qwen3_0.6b_lora"
            temps = json.loads(comparison.read_text())["models"][name]["temperatures"]
        input_file = args.input if args.input.is_absolute() else root / args.input
        request = json.loads(input_file.read_text())
        result = runner.predict(**request, temperatures=temps)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
