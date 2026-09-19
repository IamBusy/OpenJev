# OpenJev

[English](README.md) · [实验结果](reports/v03/RESULTS.md) · [复现说明](docs/REPRODUCING.md)

OpenJev 是一个研究型开源项目：输入状态、问题和候选描述，直接得到结构化概率，
无需生成答案文本。支持 `choice`（候选选择）、`noul`（命题为真的概率）和
`score`（有序评分的期望等级）。

项目受 TypeSafe 的 Jev 启发，与 TypeSafe 无隶属关系。这里使用监督交叉熵训练和
独立校准集，没有声称复现 Jev 未公开的模型结构、权重或 RLCD 算法。

## 快速使用

建议使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
实测环境为 Apple M3 Pro；CPU 使用 FP32，本版本尚未实现 CUDA 加速。
基础模型约 1.2 GB，安装和运行需预留数 GB 磁盘及内存空间。

```bash
git clone https://github.com/IamBusy/OpenJev.git
cd OpenJev
uv sync --frozen --extra qwen --extra dev
uv run --no-sync openjev-branch download
uv run --no-sync openjev-branch predict --input examples/refund.json
uv run --no-sync openjev-branch serve --port 8081
```

下载命令会取得固定版本的 Qwen 基模，以及 GitHub Release 上的 LoRA、评分头和
校准参数，并检查哈希。推理和默认数据重建都不需要 DeepSeek 密钥。
服务位于 `http://127.0.0.1:8081/v1/decide`，只面向本机实验。

## 当前实验

v0.3 使用 Qwen3-0.6B，同一状态只编码一次，每个问题和候选独立评分，再在每题内
归一化。单题最多 255 个候选，每次请求最多 512 个候选分支。状态前缀上限为
768 tokens，候选分支为 192 tokens，包含提示词开销；超限会明确报错。

- 固定 60 题：v0.2 为 39/60，v0.3 为 45/60，DeepSeek 为 59/60。
- 固定 16 题候选换序一致率：68.75% → 100%。
- 较长合成状态、8 题 × 4 候选：本机热启动耗时约 1.38 秒 → 0.70 秒。
- 短状态单题反而变慢，部分银行分类、权限规则和物流任务退步。

这些结果来自小规模固定实验。结构和训练数据同时发生变化，不能把准确率变化
单独归因于结构；概率校准也不保证跨域有效。详见[完整结果与限制](reports/v03/RESULTS.md)。

## 开发与复现

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv build
```

[复现指南](docs/REPRODUCING.md)说明了公开数据下载、固定场景快照、训练选择、
校准和评测流程。CI 覆盖微型随机模型的缓存前向及梯度等价检查；需要完整模型和
数据的集成测试，在缺少本地文件时会明确跳过。

代码使用 Apache-2.0。公共数据保留各自许可证，详见[来源与署名](THIRD_PARTY.md)。
[模型说明](docs/MODEL_CARD.md) · [贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md)
