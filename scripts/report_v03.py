"""Publish measured v0.3 results, including regressions and runtime tradeoffs."""

import json
import shutil
from pathlib import Path

from openjev.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    comparison = json.loads((ROOT / "reports/v03/comparison.json").read_text())
    selection = comparison["selection"]
    checks = json.loads((ROOT / "reports/v03/structural_checks.json").read_text())
    timing = json.loads((ROOT / "reports/v03/matched_latency.json").read_text())
    data = json.loads((ROOT / "reports/v03/data_manifest.json").read_text())
    rendering = json.loads((ROOT / "reports/v03/rendering.json").read_text())
    rows = [
        "# OpenJev v0.3：共享状态与候选直接评分",
        "",
        "本轮保留 Qwen3-0.6B，替换 A/B/C 字母概率读出为一个独立训练的标量评分头。"
        "同一请求的状态前缀只编码一次，各问题及候选分支互相隔离；"
        "候选在本题内部归一化为概率。仍使用监督交叉熵和独立校准，没有声称复现 RLCD。",
        "",
        "## 计算结构",
        "",
        "~~~mermaid",
        "flowchart LR",
        " S[共同状态] --> P[Qwen 前缀编码一次]",
        " P --> A[问题 1 的候选分支]",
        " P --> B[问题 2 的候选分支]",
        " P --> C[其他独立问题]",
        " A --> H[共享标量评分头]",
        " B --> H",
        " C --> H",
        " H --> D[每题独立归一化]",
        " D --> R[Choice / Noul / Score]",
        "~~~",
        "",
        "评分函数只读取本题的状态、问题和当前候选，不包含其它候选的顺序或内容。"
        "评分头以预训练 Yes/No 的语义方向初始化，随后单独学习；"
        "预测时不执行语言模型的词表输出投影，也不生成答案 token。",
        "",
        "缓存只在一次请求或一个训练状态组内存在，不跨权重更新复用。"
        "单元测试验证了共享和重复计算的输出、梯度一致；"
        "运行时还去除了冗余 KV 复制和逐题 GPU 往返。右侧填充的未来 token 被因果注意力隔离，"
        "并通过无填充参考验证。",
        "",
        "## 数据和训练",
        "",
        "| 分区 | 判断记录 | 独立状态组 |",
        "| --- | ---: | ---: |",
    ]
    for split in ["train", "dev", "calibration", "test_fresh"]:
        item = data["splits"][split]
        rows.append(f"| {split} | {item['records']} | {item['state_groups']} |")
    rows += [
        "",
        "公开训练部分包括 BoolQ、ARC、银行意图和情感数据。新构造的退款、故障场景"
        "包含同一状态下的多种判断；权限和物流场景整体留作未训练的新任务族。"
        "标签来自公开标注或可执行规则。信息不足使用明确的 unknown 候选，不伪造 0.5 概率。",
        "",
        f"DeepSeek 对 48 个世界做叙述改写与事实复核，{rendering['accepted']} 个通过严格核对，"
        f"{len(rendering['rejected'])} 个未通过，改用确定性模板。"
        f"实际记录 {rendering['recorded_calls']} 次调用；输入 {rendering['prompt_tokens']:,}、"
        f"输出 {rendering['completion_tokens']:,} tokens。"
        "复核失败可能包含类型或枚举值不一致，不等于全部都是语义错误。教师不决定 gold 标签。",
        "",
        "随机生成的初版存在少数类覆盖不足，已在最终模型测试前补齐主要结果类别；"
        "最初短暂的训练被中止并保留在归档中。最终报告使用平衡后冻结的数据重新训练，"
        "补齐过程没有使用模型测试预测，也没有新增教师调用。",
        "",
        f"本轮训练 {selection['trainable_adapter_parameters']:,} 个 LoRA 参数及 "
        f"{selection['trainable_head_parameters']:,} 个评分头参数，"
        f"共两轮、约 {selection['training_seconds'] / 60:.1f} 分钟。"
        f"仅按开发集 NLL 选择第 {selection['selected_update']} 次更新："
        f"{selection['initial_dev_nll']:.3f} → {selection['best_dev_nll']:.3f}。",
        "",
        "## 结构性验证",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
        f"| 与 v0.2 相同的 16 道候选换序题 | 一致率 {checks['v02_matched_16']['decision_consistency']:.1%}；v0.2 为 68.75% |",
        f"| 新增 32 道候选换序题 | 一致率 {checks['fresh_32']['decision_consistency']:.1%} |",
        f"| 8 个多问题状态，单独问与一起问 | 最大概率差 {checks['question_isolation']['max_probability_error']:.3g} |",
        f"| 同模型共享状态与重新计算 | 最大概率差 {checks['cache_equivalence']['max_probability_error']:.3g} |",
        f"| 255 候选执行测试 | 概率有限、和为 {checks['large_candidate_contract']['probability_sum']:.9f}、输出属于候选集合 |",
        "",
        "255 候选测试验证运行和类型约束，不证明大候选集合上的校准或普遍准确率。"
        "候选独立评分提供了结构上的换序等变性；排序后再调用旧模型也能稳定输入顺序，"
        "所以这不是唯一能稳定外部接口的办法。新结构额外去除了固定字母映射和共享状态的重复计算。",
        "",
        "## 相同新测试题上的准确率",
        "",
        "| 任务 | N | v0.2 | 新结构未训练 | v0.3 训练后 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, values in comparison["models"]["branch_trained"]["groups"]["fresh"].items():
        current = values["calibrated"]
        previous = comparison["models"]["qwen_v02"]["groups"]["fresh"][task]["calibrated"]
        initial = comparison["models"]["branch_initial"]["groups"]["fresh"][task]["calibrated"]
        rows.append(
            f"| {task} | {current['n']} | {previous['accuracy']:.2%} | {initial['accuracy']:.2%} | {current['accuracy']:.2%} |"
        )
    rows += [
        "",
        "退款资格、信息缺失识别及部分阅读判断改善，但权限等级差和物流复合决策仍很弱。"
        "这些失败完整保留，没有用测试结果重新挑选检查点。"
        "本轮同时改变了结构和训练数据，因此准确率变化不是纯结构消融结论。",
        "",
        "## 旧测试集回归",
        "",
        "| 任务 | v0.2 | v0.3 |",
        "| --- | ---: | ---: |",
    ]
    for task, values in comparison["models"]["branch_trained"]["groups"]["regression"].items():
        a = comparison["models"]["qwen_v02"]["groups"]["regression"][task]["calibrated"]["accuracy"]
        b = values["calibrated"]["accuracy"]
        rows.append(f"| {task} | {a:.2%} | {b:.2%} |")
    rows += [
        "",
        "旧银行意图和部分真假判断出现退步。v0.3 保持为可选实验后端，不据此宣称全面优于旧版本。",
        "",
        "## 独立 DeepSeek 配对参考",
        "",
        "| 模型 | 正确 / 总数 | 准确率 |",
        "| --- | ---: | ---: |",
    ]
    for name, values in comparison["paired_reference"].items():
        correct = sum(v["correct"] for v in values.values())
        n = sum(v["n"] for v in values.values())
        rows.append(f"| {name} | {correct}/{n} | {correct / n:.2%} |")
    rows += [
        "",
        "配对集每个任务只有 4 题，共 60 题；仅提供方向性参考。DeepSeek 的输出不是 gold。"
        "新测试样本没有参与该轮训练/选择，但公开语料预训练污染无法排除。",
        "",
        "## 同输入、交错测量的延迟",
        "",
        "M3 Pro、BF16、4 个 CPU 线程，预热 2 次、每条路径计时 8 次，三条路径轮换顺序。"
        "包含各自 predict 调用的输入处理和结果构造，不含模型加载或网络。",
        "",
        "| 状态 | 问题数×每题候选 | v0.2 / ms | v0.3 共享 / ms | v0.3 重算 / ms | 共享相对重算 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for w in timing["workloads"]:
        p = w["paths"]
        rows.append(
            f"| {w['state_length']} | {w['questions']}×4 | {p['v02']['p50_ms']:.1f} | "
            f"{p['shared']['p50_ms']:.1f} | {p['recomputed']['p50_ms']:.1f} | "
            f"{w['shared_speedup_over_recomputed']:.2f}× |"
        )
    rows += [
        "",
        "共享与重算列是同一新模型的优化对照，不能把该倍数解释为相对 v0.2 或 Jev 的速度优势。"
        "新结构对每个候选进行一次分支计算，因此短状态、单题可能更慢；"
        "长状态、多问题更能获益。候选后缀计算不是免费的。",
        "",
        "## 仍未解决",
        "",
        "- 新任务族上的复合条件、数值差和优先级规则仍可能失败。",
        "- 概率仍需在实际业务分布上重新验证；平衡模拟数据不代表真实发生率。",
        "- 独立评分对依赖整个候选集合的相对问题、none-of-the-above 等尚未充分验证。",
        "- 单种子、小规模数据；确定性规则场景不能替代真实业务或广泛语言评测。",
        "- 当前使用标准 Transformers/PyTorch 内核，没有复现 Jev 的专用采样器或 RLCD。",
        "",
        "## 使用",
        "",
        "~~~bash",
        "uv sync --frozen --extra dev --extra qwen",
        "uv run --no-sync openjev-branch predict --input examples/refund.json",
        "uv run --no-sync openjev-branch serve --port 8081",
        "~~~",
        "",
        "接口为本机 /v1/decide。共享前缀只在一个请求内复用。"
        "代码、数据版本、校准参数、训练日志及权重哈希均已保存。",
    ]
    (ROOT / "reports/v03/RESULTS.md").write_text("\n".join(rows) + "\n")
    output = ROOT / "artifacts/openjev-branch-v0.3"
    if output.exists():
        shutil.copy2(ROOT / "reports/v03/RESULTS.md", output / "RESULTS.md")
        card = [
            "---",
            "base_model: Qwen/Qwen3-0.6B",
            "library_name: peft",
            "tags:",
            "- openjev",
            "- experimental",
            "- candidate-scoring",
            "---",
            "# OpenJev-Branch-v0.3",
            "",
            "Experimental Qwen3-0.6B LoRA plus a 1024-parameter scalar candidate head.",
            "A shared state prefix and isolated candidate branches replace A–Z label logits.",
            "The head is initialized from a pretrained Yes/No direction but independently trained.",
            "No vocabulary projection or autoregressive answer decoding is used in prediction.",
            "Base weights are separate and pinned in MANIFEST.json.",
            "725 training judgments from 486 state groups; 405 fresh judgments for evaluation.",
            "Data includes public annotations and exact-rule software-state simulations.",
            "See RESULTS.md for regressions, cache benchmarks and probability limitations.",
            "This is not a reproduction of TypeSafe's undisclosed architecture or RLCD.",
            "Base model and source code: Apache-2.0. Dataset terms remain separately documented",
            "in the project's THIRD_PARTY.md; no blanket dataset license is asserted here.",
        ]
        (output / "README.md").write_text("\n".join(card) + "\n")
        manifest = json.loads((output / "MANIFEST.json").read_text())
        manifest["files"] = {
            str(p.relative_to(output)): sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "MANIFEST.json"
        }
        write_json(output / "MANIFEST.json", manifest)
    print(ROOT / "reports/v03/RESULTS.md")


if __name__ == "__main__":
    main()
