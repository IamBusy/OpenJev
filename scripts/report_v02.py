"""Build a concise report and a portable LoRA adapter bundle from measured results."""

import json
import shutil
from pathlib import Path

from openjev.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "reports/v02/comparison.json").read_text())
    selection = report["selection"]
    names = {
        "banking_seen_choice": "已见银行意图，8 候选",
        "banking_unseen_choice": "未见银行意图，8 候选",
        "banking_seen_verify": "最佳意图真假判断",
        "sentiment_score": "情感等级",
        "arc_easy": "ARC-Easy",
        "boolq": "BoolQ",
        "rubric_en": "英文规则评分",
        "rubric_zh": "中文规则评分",
    }
    rows = [
        "# OpenJev v0.2：Qwen 基座与横向评测",
        "",
        "本轮使用 Qwen3-0.6B，在本机训练约 115 万个 LoRA 参数，更新注意力的 q_proj/v_proj。"
        "模型读取合法候选字母的 logits，直接返回有限候选分布；不生成解释或答案字符串。"
        "这不是 TypeSafe 未公开架构或 RLCD 的复现。",
        "",
        "## 本轮数据与选择方式",
        "",
        "训练 768 条、开发 96 条、校准 96 条；测试集 448 条；另提前固定其中 64 条"
        "作为 DeepSeek 配对参考。全部旧版合成数据暂时排除，保留原文件用于审计。"
        "新训练数据使用公开人工标签，采用词面近邻候选采样，并把真假问题限定为最佳单一意图。"
        "近邻仍是粗糙启发式，可能包含简单负例，不能视作人工审核过的困难负例。",
        "",
        "固定测试覆盖旧任务回归、从未参与本项目训练的新意图原文、ARC-Easy、BoolQ，"
        "以及中英文精确规则评分。旧银行和情感测试已在 v0.1 被分析过，明确作为回归检查，"
        "不能再声称完全盲测。预训练语料是否包含这些公开任务无法排除。",
        "",
        "同一题的状态、问题、候选集合及顺序对所有模型一致。入选输入同时满足 MiniLM"
        "与 Qwen 的长度上限，没有静默截断。开发集仅用于选择检查点，校准集用于温度拟合。",
        "",
        f"本轮 LoRA 训练用时约 {selection['training_seconds']:.1f} 秒，"
        f"共 96 次更新，选择第 {selection['selected_update']} 次更新的检查点。"
        "这是单种子、小样本试验，不是完整规模训练。",
        "",
        "## 相同 448 题上的准确率",
        "",
        "| 任务 | N | 旧版 MiniLM | 未训练 Qwen | Qwen＋LoRA |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, title in names.items():
        values = [
            report["models"][model]["groups"][task]["calibrated"]
            for model in ["openjev_minilm_v01", "qwen3_0.6b_native", "qwen3_0.6b_lora"]
        ]
        rows.append(
            f"| {title} | {values[0]['n']} | {values[0]['accuracy']:.2%} | {values[1]['accuracy']:.2%} | {values[2]['accuracy']:.2%} |"
        )
    rows += [
        "",
        "这里是项目筛选的子集与候选协议，不能直接对照公开排行榜。"
        "中文规则题只有一个模板、32 个状态变化，不能据此推断广泛中文能力。"
        "每项样本较少，尤其 1–2 道题的差异不应被解释为稳定优势。",
        "",
        "## 专用传统分类器",
        "",
        "| 方法 | 意图训练样本 | 已见银行意图准确率（相同 64 题） |",
        "| --- | ---: | ---: |",
    ]
    for model, title in [
        ("tfidf_matching_choice_data", "TF-IDF＋逻辑回归，匹配 Choice 数据"),
        ("tfidf_full_seen_training", "TF-IDF＋逻辑回归，完整已见意图训练数据"),
    ]:
        m = report["models"][model]
        rows.append(
            f"| {title} | {m['training_records']} | {m['groups']['banking_seen_choice']['calibrated']['accuracy']:.2%} |"
        )
    rows += [
        "",
        "完整数据分类器使用更多银行标签监督，因此训练预算不相同；它展示了固定业务"
        "分类的强专用参照。它不支持未训练的新类别或任意自然语言问题，所以其它任务标为"
        "不适用，不伪装成通用决策模型。",
        "",
        "## DeepSeek 配对参考（相同 64 题）",
        "",
        "| 方法 | 正确 / 总数 | 准确率 |",
        "| --- | ---: | ---: |",
    ]
    pretty = {
        "openjev_minilm_v01": "旧版 MiniLM",
        "qwen3_0.6b_native": "未训练 Qwen",
        "qwen3_0.6b_lora": "Qwen＋LoRA",
        "deepseek_v4_pro": "DeepSeek V4 Pro",
    }
    for model, title in pretty.items():
        values = report["paired_reference_64"][model].values()
        correct = sum(x["correct"] for x in values)
        count = sum(x["n"] for x in values)
        rows.append(f"| {title} | {correct}/{count} | {correct / count:.2%} |")
    rows += [
        "",
        "每个任务只有 8 道配对题，这是额外参考而非稳定排行榜。DeepSeek 返回离散答案，"
        "不把它的预测当作真值，也不伪造概率质量指标。实际标签仍来自公开标注或规则生成器。",
        "",
        "## 概率质量与顺序敏感性",
        "",
        "| 任务 | Qwen＋LoRA NLL | ECE |",
        "| --- | ---: | ---: |",
    ]
    for task, title in names.items():
        m = report["models"]["qwen3_0.6b_lora"]["groups"][task]["calibrated"]
        rows.append(f"| {title} | {m['nll']:.4f} | {m['ece']:.4f} |")
    p = report["option_permutation"]
    rows += [
        "",
        "以上为按类型拟合温度后的指标；每类校准只有 32 条，跨领域的校准仍不充分。"
        "分布是合法标签 token 上的条件概率，不能直接等同于真实语义正确率。",
        "",
        f"另对 {p['n']} 道选择题反转候选顺序，决策一致率为 {p['decision_consistency']:.2%}。"
        "这暴露了明显的位置敏感性，后续需用候选换序训练或更合适的候选评分结构改善。"
        "该诊断没有用于重选检查点。",
        "",
        "## 本机热启动延迟",
        "",
        "相同的一道银行意图题、8 个候选；3 次预热、12 次计时，排除模型加载。"
        "Qwen 使用 BF16，MiniLM 使用 FP32。各自原生输入处理和评分开销计入；"
        "不含 HTTP 网络时间。",
        "",
        "| 方法 | p50 / ms | p95 / ms |",
        "| --- | ---: | ---: |",
    ]
    for model, title in [
        ("openjev_minilm_v01", "旧版 MiniLM"),
        ("qwen3_0.6b_native", "未训练 Qwen"),
        ("qwen3_0.6b_lora", "Qwen＋LoRA"),
    ]:
        if "latency" not in report["models"][model]:
            continue
        timing = report["models"][model]["latency"]
        rows.append(f"| {title} | {timing['p50_ms']:.2f} | {timing['p95_ms']:.2f} |")
    rows += [
        "",
        "不能把远程 DeepSeek/Jev API 时间与这些本地数值相除后宣传模型加速倍数。",
        "",
        "## 使用与复现",
        "",
        "数据格式和真实样例见 [DATA_WALKTHROUGH.md](../../docs/DATA_WALKTHROUGH.md)。"
        "原始 JSON 指标、数据锁及 LoRA 选择记录与本报告放在一起。"
        "项目主线选择 Qwen；旧 MiniLM 保留为对照。",
        "",
        "~~~bash",
        "uv sync --frozen --extra dev --extra qwen",
        "uv run --no-sync openjev-qwen download",
        "uv run --no-sync openjev-qwen predict --input examples/refund.json",
        "~~~",
        "",
        "本机已有所有数据和检查点。新副本先按 v0.1 流程准备公开数据及其历史检查点"
        "（用于 MiniLM 对照），再运行 openjev-qwen prepare、train、reference 和 evaluate。"
        "reference 使用获授权的 DeepSeek 账户；已有结果自动读取缓存。",
        "",
        "当前 Qwen 直接读出后端最多支持 26 个候选。超过输入长度上限会拒绝，而非截断。"
        "本轮没有测试 Gemma，也没有 Jev API 的同条件对照。",
        "",
        "数据来源新增 [BoolQ](https://huggingface.co/datasets/google/boolq)"
        "（CC BY-SA 3.0）和 [ARC](https://huggingface.co/datasets/allenai/ai2_arc)"
        "（CC BY-SA 4.0），均仅用于评测。原数据许可见 [THIRD_PARTY.md](../../THIRD_PARTY.md)。",
    ]
    (ROOT / "reports/v02/RESULTS.md").write_text("\n".join(rows) + "\n")
    output = ROOT / "artifacts/openjev-qwen3-0.6b-v0.2"
    output.mkdir(exist_ok=True)
    source = ROOT / selection["selected_adapter"]
    for filename in ["adapter_model.safetensors", "adapter_config.json"]:
        shutil.copy2(source / filename, output / filename)
    config = json.loads((output / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = selection["model_name"]
    config["revision"] = selection["model_revision"]
    write_json(output / "adapter_config.json", config)
    write_json(output / "openjev_config.json", selection["config"])
    write_json(output / "calibration.json", report["models"]["qwen3_0.6b_lora"]["temperatures"])
    shutil.copy2(ROOT / "reports/v02/RESULTS.md", output / "RESULTS.md")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    card = [
        "---",
        "license: apache-2.0",
        "base_model: Qwen/Qwen3-0.6B",
        "library_name: peft",
        "tags:",
        "- openjev",
        "- lora",
        "- experimental",
        "---",
        "# OpenJev-Qwen3-0.6B-v0.2",
        "",
        "LoRA adapter for direct typed candidate decisions. Base model revision: "
        + selection["model_revision"]
        + ".",
        "Trainable parameters: 1,146,880; training records: 768; selected update: 96.",
        "The base checkpoint is separate. Use the OpenJev Qwen loader and recorded chat template,",
        "not ordinary free-text generation, for the reported decision behavior.",
        "Maximum 26 candidates and 768 prompt tokens in this implementation.",
        "The same 64-question reference slice scored 49/64 for this adapter, 44/64 for the",
        "native base, 35/64 for the MiniLM pilot, and 58/64 for DeepSeek V4 Pro.",
        "These are small project subsets, not official benchmark or frontier-model claims.",
        "Only 11/16 sampled decisions were unchanged after candidate reversal.",
        "Probability calibration is limited; see RESULTS.md and the full project report.",
    ]
    (output / "README.md").write_text("\n".join(card) + "\n")
    write_json(
        output / "MANIFEST.json",
        {
            "source_adapter": selection["selected_adapter"],
            "files": {
                p.name: sha256(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "MANIFEST.json"
            },
        },
    )
    print(ROOT / "reports/v02/RESULTS.md")


if __name__ == "__main__":
    main()
