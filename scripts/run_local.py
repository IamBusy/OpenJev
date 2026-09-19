"""Run a new named experiment without overwriting prior results."""

import argparse
import json
from pathlib import Path

from openjev.audit import audit
from openjev.data import prepare
from openjev.evaluate import compare_runs, evaluate, freeze_selection
from openjev.io import write_json
from openjev.train import calibrate, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--with-synthetic", action="store_true")
    args = parser.parse_args()
    if not args.experiment.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use a simple experiment name")
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/processed/manifest.json").exists():
        prepare(root)
    audit(root)
    if args.with_synthetic and not (root / "data/processed/synthetic_train.jsonl").exists():
        raise FileNotFoundError("Run openjev synthesize (or synthesize --offline) first")
    runs = []
    for seed in args.seeds:
        config = json.loads((root / "configs/local.json").read_text())
        config["seed"] = seed
        config_path = root / f"artifacts/configs/{args.experiment}-seed{seed}.json"
        write_json(config_path, config)
        arms = [("public", False)] + ([("augmented", True)] if args.with_synthetic else [])
        for arm, augmented in arms:
            run = root / f"runs/{args.experiment}-{arm}-seed{seed}"
            train(root, config_path, run, augmented)
            calibrate(root, run / "checkpoint")
            runs.append(run)
    selection_file = args.experiment + "-selection.json"
    freeze_selection(root, runs, selection_file)
    for run in runs:
        evaluate(
            root,
            run / "checkpoint",
            root / "reports" / (run.name + ".json"),
            selection_file=selection_file,
        )
    if args.with_synthetic:
        for seed in args.seeds:
            compare_runs(
                root,
                root / f"runs/{args.experiment}-public-seed{seed}",
                root / f"runs/{args.experiment}-augmented-seed{seed}",
                f"{args.experiment}-ablation-seed{seed}.json",
            )
    print("Completed new experiment:", args.experiment)


if __name__ == "__main__":
    main()
