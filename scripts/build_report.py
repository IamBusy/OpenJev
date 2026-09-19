"""Build the report and research figures exclusively from recorded results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openjev.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    reports = {
        (arm, seed): load(f"reports/v01-{arm}-seed{seed}.json")
        for seed in [17, 23, 42]
        for arm in ["public", "augmented"]
    }
    public = reports["public", 17]
    augmented = reports["augmented", 17]
    labels = {
        "test_id/banking77/intent_choice": "BANKING77 已见类别选择",
        "test_id/banking77/intent_verification": "BANKING77 已见类别真假判断",
        "test_id/tweeteval_sentiment/sentiment_score": "TweetEval 情感等级",
        "test_unseen/banking77/intent_choice": "BANKING77 未见类别选择",
        "test_unseen/banking77/intent_verification": "BANKING77 未见类别真假判断",
        "test_transfer/clinc150/intent_choice": "CLINC 跨数据集迁移",
    }
    english = [
        "Seen intent",
        "Seen verification",
        "Sentiment",
        "Unseen intent",
        "Unseen verification",
        "Transfer",
    ]
    synthesis = load("reports/synthesis.json")
    data = load("reports/data_manifest.json")
    export = load("reports/export.json")
    latency = load("reports/latency.json")
    rows = [
        "# OpenJev v0.1 实验报告",
        "",
        "2026-09-19。全部训练、特征提取和本地推理在 Apple M3 Pro / 36GB 内存上完成；"
        "训练数据改写使用获授权的 DeepSeek API。代码、数据准备、训练、校准、独立评测、"
        "模型导出及离线重载均已执行。",
        "",
        "## 结论",
        "",
        "这个首版验证了小模型可以直接输出动态候选项的结构化概率分布。"
        "已见类别的任务性能明显优于冻结编码器的语义相似度基线；"
        "跨数据集能力仍有不足。少量合成数据的收益随任务和随机种子变化，"
        "当前证据不足以宣称合成数据稳定改善泛化。",
        "",
        "这是一项最小可复现实验：冻结约 22.7M 参数的 MiniLM 编码器，"
        f"训练 {export['trainable_head_parameters']:,} 参数的共享残差评分头；"
        "没有微调整个编码器，也没有实现 TypeSafe 未公开的 RLCD。"
        "输出合法性、回答正确性和概率校准是不同性质。",
        "",
        "## 数据及合成",
        "",
        "| 数据分区 | 记录数 | 独立来源组数 |",
        "| --- | ---: | ---: |",
    ]
    for split, info in data["splits"].items():
        rows.append(f"| {split} | {info['records']:,} | {info['source_groups']:,} |")
    rows += [
        "",
        f"另有 DeepSeek 合成训练记录 {synthesis['accepted_records']} 条；它们继承原训练样本的来源组，"
        "不当作新的独立标注样本。生成 192 条，独立教师复核剔除 16 条。"
        f"共记录 {synthesis['completed_calls']} 次调用，输入 {synthesis['prompt_tokens']:,} tokens，"
        f"输出 {synthesis['completion_tokens']:,} tokens。没有新增调用用于随机种子复核。",
        "",
        "公共来源是 BANKING77、TweetEval sentiment 和 CLINC150。另有本地 urn 模拟器，"
        "其标签为球数占总数的精确概率。处理保留官方测试边界，并按规范化原文做跨分区去重；"
        "同一原文的 Choice/Noul 视图、教师改写共享来源组。"
        "未见类别的 15 个 BANKING77 标签不用于训练、开发或校准。",
        "",
        "BANKING77 已见类别选择使用 62 个已见类别加 other（63 候选），"
        "未见类别测试使用 15 个未见类别加 other（16 候选），所以二者准确率不能直接比较难度。"
        "CLINC 测试为预先固定的每类 10 条及 300 条 OOS，再去重，使用完整 151 候选。"
        "它是项目子集评测，不是官方全量排行榜结果。TweetEval 训练每类最多 1800 条，"
        "测试使用官方测试集经原文去重后的记录。",
        "",
        "## 冻结模型选择后的测试结果",
        "",
        "主发布模型始终为 seed=17 的两数据版本中开发集 NLL 更低者，即加入合成数据的版本。"
        "校准只使用 calibration 分区。下面的测试结果未用于更换主发布模型。",
        "",
        "| 任务 | N | 语义基线准确率 | 仅公开数据 | 加入合成数据 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    replication = {}
    for group, title in labels.items():
        base = augmented["groups"][group]["semantic_calibrated"]
        a = public["groups"][group]["trained_calibrated"]
        b = augmented["groups"][group]["trained_calibrated"]
        rows.append(
            f"| {title} | {b['n']:,} | {base['accuracy']:.2%} | {a['accuracy']:.2%} | {b['accuracy']:.2%} |"
        )
        values = {
            arm: [
                reports[arm, seed]["groups"][group]["trained_calibrated"]["accuracy"]
                for seed in [17, 23, 42]
            ]
            for arm in ["public", "augmented"]
        }
        deltas = np.array(values["augmented"]) - np.array(values["public"])
        replication[group] = {
            "accuracy_by_seed": values,
            "seeds": [17, 23, 42],
            "mean_augmented_minus_public": float(deltas.mean()),
            "min_difference": float(deltas.min()),
            "max_difference": float(deltas.max()),
            "seed_count": 3,
        }
    rows += [
        "",
        "温度缩放不改变分类 argmax，所以上表校准前后准确率相同。"
        "宏平均 F1、Wilson 区间、逐桶计数、Brier、NLL、等级 MAE、RPS 和风险/覆盖率曲线"
        "保存在对应的结果 JSON 中。",
        "",
        "## 概率校准",
        "",
        "| 任务 | 原始 NLL | 校准 NLL | 原始 ECE | 校准 ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group, title in labels.items():
        raw = augmented["groups"][group]["trained_raw"]
        cal = augmented["groups"][group]["trained_calibrated"]
        rows.append(
            f"| {title} | {raw['nll']:.4f} | {cal['nll']:.4f} | {raw['ece']:.4f} | {cal['ece']:.4f} |"
        )
    rows += [
        "",
        "ECE 使用 15 个固定等宽区间。最大候选概率作为 confidence，"
        "校准域内拟合的温度不能保证新领域或新候选规模上的校准。"
        "风险/覆盖率计算保留阈值上的所有并列样本，报告实际覆盖率。",
        "",
        "![Calibration reliability](figures/reliability.png)",
        "",
        "## 合成数据效应的随机种子复核",
        "",
        "种子为 17、23、42；划分、超参数和数据不变。额外种子为诊断复核，"
        "不用于重选主模型。下面是加入合成数据减去仅公开数据的准确率差，单位为百分点。",
        "",
        "| 任务 | 三种子均值 | 最小值 | 最大值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, title in labels.items():
        r = replication[group]
        rows.append(
            f"| {title} | {100 * r['mean_augmented_minus_public']:+.2f} | {100 * r['min_difference']:+.2f} | {100 * r['max_difference']:+.2f} |"
        )
    rows += [
        "",
        "三个种子不足以提供精确的训练分布置信区间。单种子报告中的成对 bootstrap"
        "区间只反映测试样本不确定性，不能替代训练种子方差。"
        "特别是主种子的情感任务下降，不能被解释成所有种子上的稳定下降。",
        "",
        "![Augmentation across seeds](figures/augmentation.png)",
        "",
        "## 已知概率模拟器",
        "",
        "这是额外的数值概率诊断，不是自然语言业务能力证明。"
        "软目标不按单个人工标签报告准确率；使用与真实分布的平方距离和期望 NLL。",
        "",
        "| 方法 | 与真实分布的平方距离 | 期望 NLL |",
        "| --- | ---: | ---: |",
    ]
    sim = augmented["groups"]["test_simulator/urn_simulator/known_probability"]
    for key, name in [
        ("uniform", "均匀分布"),
        ("semantic_calibrated", "语义基线"),
        ("trained_calibrated", "训练并校准"),
    ]:
        m = sim[key]
        rows.append(f"| {name} | {m['distribution_squared_error']:.5f} | {m['nll']:.5f} |")
    rows += [
        "",
        "模型在这个诊断中尚未胜过简单均匀分布；它没有学会可靠读取球数并计算真实概率。"
        "因此不能宣称首版已经普遍具备可信概率判断能力。",
        "",
        "## 本机延迟与计算成本",
        "",
        "M3 Pro，编码器 MPS、评分头 CPU，FP32。每种形状 3 次预热、20 次重复。"
        "计时包含验证、分词、编码、评分、校准和响应构造，不含模型加载或 HTTP 传输。"
        "不跨请求缓存 embedding；同一请求内完全相同的文本去重。"
        "下面是合成吞吐负载，不是业务正确率评测。",
        "",
        "| 问题数 | 每题候选数 | p50 / ms | p95 / ms | 编码文本数 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for w in latency["workloads"]:
        rows.append(
            f"| {w['questions']} | {w['candidates_per_question']} | {w['p50_ms']:.2f} | {w['p95_ms']:.2f} | {w['unique_encoded_texts']} |"
        )
    training_times = [
        load(f"runs/v01-{arm}-seed{seed}/run.json")["training_seconds"]
        for seed in [17, 23, 42]
        for arm in ["public", "augmented"]
    ]
    rows += [
        "",
        f"六次评分头训练合计约 {sum(training_times):.1f} 秒，"
        "不含下载、首次编码与评测；这不是完整基础模型训练成本。"
        f"本次模型加载约 {latency['model_load_seconds']:.2f} 秒。"
        "不能用这些本地数字和 Jev 的远端 API 延迟计算公平速度倍数。",
        "",
        "## 可复现性与限制",
        "",
        "- 原始文件版本、SHA-256、数据分区、来源组、模型配置、权重和校准参数均有记录。",
        "- 合成数据可离线重放；教师标签来自同一模型族的复核，存在相关偏差。",
        "- 权重导出后实际文本推理结果与原检查点一致，最大概率误差为 0。",
        "- 增加了缓存命中/未命中不影响训练随机数的回归测试；主种子重新训练的两个权重均与初始结果逐字节一致。",
        "- 候选置换及加入无关问题的实测最大概率差均在记录的容差内。",
        "- 暂不保证中文、长文本、任意评分规则、复杂推理或生产级拒判。超过 192 tokens 的编码输入会披露截断。",
        "- CLINC 总体迁移结果弱于语义基线，反映任务适配的负迁移风险。",
        "- 不存在 Jev API 的同条件对比，因此不声称复现其性能或 RLCD。",
        "",
        "## 下一阶段最值得做的实验",
        "",
        "优先引入更多独立任务和高质量对比候选，做合成数据量与质量消融，"
        "再尝试微调编码器。每一轮保留新的评测任务，避免把目前已经分析过的测试集"
        "继续当作完全未见的最终验收集。先改进跨任务泛化和概率诊断，再扩大模型或引入强化学习。",
        "",
        "来源、许可及引用见 [THIRD_PARTY.md](../THIRD_PARTY.md)。",
    ]
    (ROOT / "reports/RESULTS.md").write_text("\n".join(rows) + "\n")
    write_json(ROOT / "reports/replication_summary.json", replication)

    figures = ROOT / "reports/figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    groups = list(labels)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, group, title in zip(axes, groups, english[:3], strict=True):
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1)
        for variant, name, color in [
            ("trained_raw", "Before calibration", "#9c5b26"),
            ("trained_calibrated", "After calibration", "#2563a6"),
        ]:
            bins = [
                b for b in augmented["groups"][group][variant]["reliability_bins"] if b["count"]
            ]
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                marker="o",
                markersize=3,
                label=name,
                color=color,
            )
        ax.set(
            title=title,
            xlabel="Mean confidence",
            ylabel="Observed accuracy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("OpenJev v0.1: held-out reliability, primary seed 17")
    fig.savefig(figures / "reliability.png", dpi=180)
    fig.savefig(figures / "reliability.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for index, seed in enumerate([17, 23, 42]):
        diffs = [
            100
            * (
                reports["augmented", seed]["groups"][g]["trained_calibrated"]["accuracy"]
                - reports["public", seed]["groups"][g]["trained_calibrated"]["accuracy"]
            )
            for g in labels
        ]
        ax.scatter(np.arange(len(labels)) + (index - 1) * 0.1, diffs, label=f"Seed {seed}", s=45)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)), [x.replace(" ", "\n") for x in english])
    ax.set(
        ylabel="Accuracy change (percentage points)",
        title="Adding 176 teacher-reviewed paraphrases: mixed effects across seeds",
    )
    ax.grid(axis="y", alpha=0.15)
    ax.legend(ncol=3)
    fig.savefig(figures / "augmentation.png", dpi=180)
    fig.savefig(figures / "augmentation.svg")
    plt.close(fig)

    model_dir = ROOT / "artifacts/openjev-minilm-v0.1"
    card = [
        "---",
        "license: apache-2.0",
        "language: en",
        "base_model: sentence-transformers/all-MiniLM-L6-v2",
        "pipeline_tag: zero-shot-classification",
        "library_name: pytorch",
        "tags:",
        "- openjev",
        "- decision-model",
        "- experimental",
        "---",
        "# OpenJev-MiniLM-v0.1",
        "",
        "Independent Jev-inspired research pilot: a frozen MiniLM encoder plus a trained shared candidate scorer.",
        f"Total parameters: {export['total_parameters']:,}; trainable head: {export['trainable_head_parameters']:,}.",
        "The backbone was not fine-tuned. No affiliation with TypeSafe; no reproduction of its unpublished RLCD.",
        "",
        "## Loading",
        "",
        "Install the OpenJev repository/package and use openjev.model.OpenJev with this directory.",
        "This is a custom model bundle, not a Transformers AutoModel checkpoint.",
        "Weights, tokenizer, pooling module, scorer and calibration are self-contained.",
        "",
        "## Training and evaluation",
        "",
        "Seed 17, 12 epochs, AdamW, learning rate 0.001, batch size 128, development-NLL selection.",
        "19,351 prepared records plus 176 DeepSeek-reviewed paraphrases. Data uses BANKING77,",
        "TweetEval sentiment and an exact-probability urn generator; CLINC is held out for transfer.",
        "Development and calibration data are disjoint. Three seeds were measured in two data arms.",
        "",
        "| Task | Accuracy | NLL | ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, title in zip(labels, english, strict=True):
        m = augmented["groups"][group]["trained_calibrated"]
        card.append(f"| {title} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['ece']:.4f} |")
    card += [
        "",
        "## Limitations",
        "",
        "English-only pilot with 192-token truncation per encoded text. Not a general reasoning model.",
        "Seen and unseen intent tests have different candidate counts (63 versus 16). CLINC uses 151.",
        "Cross-dataset performance is below the semantic baseline. Synthetic-data gains are not stable across seeds.",
        "The urn probability diagnostic does not beat a uniform predictor. Calibration is distribution-specific.",
        "Typed output is not proof of factual accuracy or suitability for autonomous decisions.",
        "Pretrained-encoder data contamination cannot be excluded.",
        "See RESULTS.md and THIRD_PARTY.md for full protocols, licenses, figures and negative results.",
    ]
    (model_dir / "README.md").write_text("\n".join(card) + "\n")
    for name in [
        "RESULTS.md",
        "data_manifest.json",
        "data_audit.json",
        "synthesis.json",
        "replication_summary.json",
    ]:
        import shutil

        shutil.copy2(ROOT / "reports" / name, model_dir / name)
    import shutil

    shutil.copytree(figures, model_dir / "figures", dirs_exist_ok=True)
    for path in [
        ROOT / "reports/figures/reliability.png",
        ROOT / "reports/figures/augmentation.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
