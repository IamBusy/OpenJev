"""Command-line entry point for OpenJev's visual research track."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="OpenJev visual posterior research")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate original synthetic images and exact targets"
    )
    generate.add_argument("--config", default="configs/vision-v01.json")
    generate.add_argument("--output", default="data/vision/synthetic")
    train = commands.add_parser("train", help="Train all preregistered variants/seeds")
    train.add_argument("--config", default="configs/vision-v01.json")
    train.add_argument("--data", default="data/vision/synthetic")
    train.add_argument("--output", default="runs/vision-v01")
    train.add_argument("--device", default="auto")
    evaluate = commands.add_parser("evaluate", help="Evaluate untouched tests")
    evaluate.add_argument("--config", default="configs/vision-v01.json")
    evaluate.add_argument("--data", default="data/vision/synthetic")
    evaluate.add_argument("--runs", default="runs/vision-v01")
    evaluate.add_argument("--output", default="reports/vision-v01/evaluation")
    evaluate.add_argument("--device", default="auto")
    predict = commands.add_parser(
        "predict", help="Answer questions from one shared visual posterior"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--questions", required=True)
    predict.add_argument("--prior", help="JSON file containing the 64 supplied prior weights")
    predict.add_argument("--device", default="auto")
    public = commands.add_parser(
        "prepare-public", help="Reconstruct licensed, pinned public-image subsets"
    )
    public.add_argument("--output", default="data/vision/public")
    public_train = commands.add_parser(
        "train-public", help="Extract frozen features and fit public-image readouts"
    )
    public_train.add_argument("--dataset", required=True, choices=["pets", "clevr4"])
    public_train.add_argument("--config", default="configs/vision-public-v01.json")
    public_train.add_argument("--data", default="data/vision/public/processed")
    public_train.add_argument("--backbone-cache", default="artifacts/vision/dinov2-small")
    public_train.add_argument("--features")
    public_train.add_argument("--output")
    public_train.add_argument("--device", default="auto")
    public_predict = commands.add_parser(
        "public-predict", help="Run a public-image head and shared event queries"
    )
    public_predict.add_argument("--checkpoint", required=True)
    public_predict.add_argument("--backbone", default="artifacts/vision/dinov2-small")
    public_predict.add_argument("--ontology", required=True)
    public_predict.add_argument("--image", required=True)
    public_predict.add_argument("--questions", required=True)
    public_predict.add_argument("--device", default="auto")
    export = commands.add_parser("export-data", help="Build image-Parquet data for Hugging Face")
    export.add_argument("--synthetic", default="data/vision/synthetic")
    export.add_argument("--public", default="data/vision/public/processed")
    export.add_argument("--output", default="artifacts/vision/hf-dataset")
    download = commands.add_parser(
        "download", help="Download and verify experimental visual weights"
    )
    download.add_argument("--output", default="artifacts/openjev-vision-v0.1")
    download.add_argument("--repo", default="IamBusy/OpenJev-Vision-v0.1")
    download.add_argument("--revision")
    download.add_argument("--backbone", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        from .data import generate

        result = generate(args.output, json.loads(Path(args.config).read_text()))
    elif args.command == "train":
        from .train import fit

        cfg = json.loads(Path(args.config).read_text())
        result = []
        for seed in cfg["training_seeds"]:
            for variant in cfg["variants"]:
                result.append(
                    fit(
                        args.data,
                        Path(args.output) / f"{variant}-{seed}",
                        cfg,
                        variant,
                        seed,
                        args.device,
                    )
                )
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            args.data,
            args.runs,
            args.output,
            json.loads(Path(args.config).read_text()),
            args.device,
        )
    elif args.command == "prepare-public":
        from .public_data import prepare_public

        result = prepare_public(args.output)
    elif args.command == "train-public":
        from huggingface_hub import snapshot_download

        from .public_model import cache_features, train_public

        cfg = json.loads(Path(args.config).read_text())
        backbone = snapshot_download(
            cfg["backbone"],
            revision=cfg["revision"],
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "README.md",
            ],
            local_dir=args.backbone_cache,
            token=False,
        )
        features = args.features or f"artifacts/vision/{args.dataset}_features.npz"
        output = args.output or f"runs/vision-v01/public-{args.dataset}"
        cache_features(args.data, args.dataset, backbone, features, cfg, args.device)
        result = train_public(args.data, args.dataset, features, output, cfg)
    elif args.command == "export-data":
        from .dataset_card import write_card
        from .export_data import export_dataset

        result = export_dataset(args.synthetic, args.public, args.output)
        write_card(args.output, "LICENSE")
    elif args.command == "download":
        from .hub import download_models

        result = download_models(args.output, args.repo, args.revision, backbone=args.backbone)
    elif args.command == "public-predict":
        from PIL import Image

        from .public_inference import PublicVisionModel

        model = PublicVisionModel(args.checkpoint, args.backbone, args.ontology, args.device)
        result = model.predict(
            Image.open(args.image),
            json.loads(Path(args.questions).read_text()),
        )
    else:
        from PIL import Image

        from .model import SceneModel
        from .query import answer_questions
        from .world import describe_world

        model = SceneModel(args.checkpoint, args.device)
        image = Image.open(args.image).convert("RGB")
        prior = json.loads(Path(args.prior).read_text()) if args.prior else None
        p = model.posterior(image, prior)
        questions = json.loads(Path(args.questions).read_text())
        result = {
            "model": "OpenJev-Vision-v0.1/" + model.config["variant"],
            "scope": "Controlled three-slot scenes and declared event semantics.",
            "answers": answer_questions(p, questions),
            "top_world": describe_world(p.argmax()),
            "posterior": p.tolist(),
            "visual_encodings": 1,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
