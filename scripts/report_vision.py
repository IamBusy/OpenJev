"""Build the visual research report from actual frozen experiment outputs."""

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openjev.vision.data import load_split, write_json
from openjev.vision.model import SceneModel
from openjev.vision.query import answer_questions


def mean_sd(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    root = Path.cwd()
    report_dir = root / "reports/vision-v01"
    report_dir.mkdir(parents=True, exist_ok=True)
    synthetic = json.loads((report_dir / "evaluation/metrics.json").read_text())
    public = {}
    for dataset in ("pets", "clevr4"):
        source = root / f"runs/vision-v01/public-{dataset}/results.json"
        public[dataset] = json.loads(source.read_text())
        shutil.copyfile(source, report_dir / f"public-{dataset}.json")
    seeds = synthetic["seeds"]
    table = {}
    for split in ("test_id", "test_topology", "test_appearance"):
        table[split] = {}
        for label, key in (
            ("Joint", "joint-{seed}/calibrated"),
            ("Same joint, factorized", "joint-{seed}/factorized"),
            ("Independent heads", "independent-{seed}/calibrated"),
            ("Visual evidence + known Bayes", "evidence-{seed}/calibrated"),
        ):
            values = [
                synthetic["groups"][f"{split}/{key.format(seed=seed)}"]["all"][
                    "compound_probability_mse"
                ]
                for seed in seeds
            ]
            table[split][label] = mean_sd(values)
    public_table = {}
    for variant in ("joint", "independent", "binding"):
        public_table[variant] = {
            split: mean_sd(
                [
                    public["clevr4"]["results"][f"{variant}-{seed}"][split]["accuracy"]
                    for seed in seeds
                ]
            )
            for split in ("calibrated", "seen", "unseen")
        }
    pets = mean_sd([v["calibrated"]["accuracy"] for v in public["pets"]["results"].values()])
    summary = {
        "synthetic_compound_probability_mse": table,
        "clevr4_accuracy": public_table,
        "pets_breed_accuracy": pets,
        "seeds": seeds,
        "dispersion": "sample standard deviation across training seeds",
    }
    write_json(report_dir / "SUMMARY.json", summary)
    lines = [
        "# OpenJev visual research v0.1 results",
        "",
        "Experimental baselines, three seeds (17/23/42). No claim of a new state-of-the-art model.",
        "Means ± sample standard deviation across training seeds. Synthetic and public-image tasks are separate.",
        "",
        "## Exact-posterior synthetic scenes",
        "",
        "Compound-event mean squared probability error (lower is better). Each test has 1,024 images from 512 episodes.",
        "",
        "| Model | ID | New dependency topology | New appearance |",
        "|---|---:|---:|---:|",
    ]
    for label in table["test_id"]:
        values = [table[split][label] for split in table]
        lines.append(
            "| " + label + " | " + " | ".join(f"{m:.6f} ± {s:.6f}" for m, s in values) + " |"
        )
    lines += [
        "",
        "The matched factorization control retains the learned joint model's unary marginals.",
        "Joint probabilities help in distribution, but the direct joint predictor degrades on new",
        "dependency topologies. Exact fusion with the known prior/sensor is much stronger; its",
        "visual classifier receives privileged observation labels during training.",
        "This is evidence for a research problem, not proof of a general solution.",
        "",
        "Episode-bootstrap intervals for every seed are in evaluation/metrics.json.",
        "The factorized model is itself mathematically coherent; its limitation is missing dependencies.",
        "",
        "## Public images",
        "",
        f"**Oxford-IIIT Pet:** {pets[0] * 100:.2f}% ± {pets[1] * 100:.2f} percentage points breed accuracy",
        "on a balanced 740-image official-test subset. Frozen DINOv2-small plus a trained linear head.",
        "This is not the full official benchmark, and pretrained-backbone overlap cannot be excluded.",
        "",
        "**CLEVR-4:** 400 official-validation images used as test: 320 seen-composition and 80",
        "unseen-composition images. Twenty color/shape combinations are absent from train/dev/calibration.",
        "",
        "| Readout | All | Seen pairs | Unseen pairs |",
        "|---|---:|---:|---:|",
    ]
    for name, values in public_table.items():
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{values[s][0] * 100:.2f}% ± {values[s][1] * 100:.2f} pp"
                for s in ("calibrated", "seen", "unseen")
            )
            + " |"
        )
    lines += [
        "",
        "**Negative result:** the 100-way joint classifier fails on unseen pair labels.",
        "The factorized attribute baseline generalizes better. The low-rank binding head does",
        "not improve on it. This expected failure of a flat joint label space must not be",
        "misrepresented as evidence against all joint probabilistic architectures.",
        "",
        "All variants share the exact same frozen features. Small heads are trained, not DINOv2.",
        "The binding mechanism is a known low-rank interaction model, included as a baseline.",
        "",
        "## Reproduction and interpretation",
        "",
        "See ../../docs/VISION.md and the two preregistered protocols. Checkpoints are selected by",
        "development NLL; temperature is fitted on calibration only. All three seeds are retained.",
        "The query executor uses declared event semantics, not learned free-form language understanding.",
        "Probability identities do not guarantee correct perception or real-world calibration.",
        "",
        "![Measured comparison](comparison.png)",
        "",
    ]
    latency_path = report_dir / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text())
        lines += [
            "## Matched timing",
            "",
            "| Questions | Shared p50 | Re-encode per question p50 |",
            "|---:|---:|---:|",
        ]
        for row in latency["results"]:
            lines.append(
                f"| {row['questions']} | {row['timings']['shared']['p50_ms']:.2f} ms | "
                f"{row['timings']['reencoded']['p50_ms']:.2f} ms |"
            )
        lines += [
            "",
            "Synthetic 64×192 images, small CNN, float32 MPS. Model load excluded; pixel",
            "preprocessing, inference and query construction included. Three warmups, 20 repeats.",
            "These are in-process local timings, not DINOv2 timings or comparisons to Jev.",
            "",
        ]
    (report_dir / "RESULTS.md").write_text("\n".join(lines))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    palette = ["#3454d1", "#7d8597", "#23a19b", "#e3a33b"]
    labels = list(table["test_id"])
    x = np.arange(3)
    for i, label in enumerate(labels):
        means = [table[split][label][0] for split in table]
        axes[0].bar(x + (i - 1.5) * 0.19, means, width=0.18, color=palette[i], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, ["ID", "New topology", "New appearance"])
    axes[0].set_ylabel("Compound probability MSE (lower is better)")
    axes[0].set_title("Retaining dependencies helps; learning them can fail")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8, loc="lower left", bbox_to_anchor=(0.065, 0.01), ncol=2)
    for i, (variant, values) in enumerate(public_table.items()):
        means = [values[split][0] * 100 for split in ("seen", "unseen")]
        axes[1].bar(
            np.arange(2) + (i - 1) * 0.24, means, width=0.23, label=variant, color=palette[i]
        )
    axes[1].set_xticks([0, 1], ["Seen color-shape pairs", "Unseen pairs"])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Joint classification accuracy (%)")
    axes[1].set_title("CLEVR-4: independent attributes generalize better")
    axes[1].legend(fontsize=9)
    axes[1].text(0.76, 1.5, "0%", ha="center", color=palette[0])
    fig.suptitle("OpenJev · Vision research v0.1 · measured baselines", fontsize=15, x=0.51)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(report_dir / "comparison.png", dpi=160)
    plt.close(fig)

    data = load_split(root / "data/vision/synthetic", "dev")
    model = SceneModel(root / "runs/vision-v01/joint-17/checkpoint")
    questions = {
        "left_red": "Is the left object red?",
        "right_red": "Is the right object red?",
        "same_color": {"type": "noul", "event": "same_color(left, right)"},
        "both_red": {"type": "noul", "event": "red(left) and red(right)"},
    }
    images, predictions = [], []
    example_dir = root / "examples/vision"
    example_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for i in (0, 1):
        image = data["images"][i]
        p = model.posterior(image, data["priors"][i])
        answers = answer_questions(p, questions)
        exact = answer_questions(data["targets"][i], questions)
        images.append(image)
        predictions.append((answers, exact))
        Image.fromarray(image).save(example_dir / f"scene-{i}.png")
        write_json(example_dir / f"scene-{i}-prior.json", data["priors"][i].tolist())
    write_json(example_dir / "questions.json", questions)
    write_json(
        example_dir / "demo-output.json",
        {
            "selection": "Fixed development episode 0, views 0 and 1",
            "checkpoint": "joint seed17, selected by development NLL",
            "views": [{"model": a, "oracle": b} for a, b in predictions],
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (image, (answers, exact)) in enumerate(zip(images, predictions, strict=True)):
        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Occluded observation" if i == 0 else "More visual evidence")
        names = list(questions)
        y = np.arange(len(names))
        axes[i, 1].barh(
            y - 0.16,
            [answers[n]["noul"] for n in names],
            height=0.3,
            color="#3454d1",
            label="Learned posterior",
        )
        axes[i, 1].barh(
            y + 0.16,
            [exact[n]["noul"] for n in names],
            height=0.3,
            color="#23a19b",
            label="Exact simulator posterior",
        )
        axes[i, 1].set_yticks(y, names)
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_xlabel("Event probability")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("One visual posterior, several compositional decisions", fontsize=15)
    fig.tight_layout()
    fig.savefig(report_dir / "demo.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
