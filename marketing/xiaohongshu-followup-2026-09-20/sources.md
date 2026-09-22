# 事实核对与来源

核对日期：2026-09-20。供作者复核，正文及图片已保留必要范围。

## 参考内容

- 参考帖：[Jev刚火两天，开源就杀疯了：0.5B就能跑！](https://www.xiaohongshu.com/explore/6aaf83ef0000000012001034)
- 上一帖：[秘密研发2h，Jev-Vision开源了](https://www.xiaohongshu.com/explore/6aae8d7400000000280328ca)
- 只参考叙事结构。未复制他人配图、特色表达或长段文字。

## Nimble

- [作者仓库](https://github.com/bespokelabsai/nimble)
- 基座为 Qwen3.5-9B；以对照数据和硬参考标签进行 LoRA 训练；当前发布接口只接受文本。
- 90.12% = 292/324 条特定留出样本的参考标签匹配率。样本组成 162 对、来自六个 source family；参考标签为合成标签。这不是通用模型能力或概率校准的证明。
- 不把“只改一处事实”描述成已经证明学习了因果推理；不把“比某个未微调 27B 在这批样本上更高”推广成普遍超越 27B。

## Kev

- [当前仓库](https://github.com/jaredpalmer/kev)
- [早期 0.5B 权重与模型卡](https://huggingface.co/jaredpalmer/kev-0.5b)
- 0.5B 指 Qwen2.5-0.5B 路线的 v0.1；当前 README 另列 0.6B / 4B / 8B 预览。配图写成“0.5B 起步”而不是把 0.5B 当成整个项目的最新版本。
- 作者模型卡记载一次训练约 1.75 小时、M5 笔记本、能耗估计约 0.06 kWh；新帖不重复成本、能耗或“无缝平替”宣传，避免把接口兼容等同能力相同。

## OpenJev-Vision

- [视觉说明](https://github.com/IamBusy/OpenJev-Vision/blob/main/docs/VISION.md)
- [公开图片协议](https://github.com/IamBusy/OpenJev-Vision/blob/main/docs/VISION_PUBLIC_PROTOCOL.md)
- [合成场景协议](https://github.com/IamBusy/OpenJev-Vision/blob/main/docs/VISION_PROTOCOL.md)
- [结果报告](https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md)
- [权重](https://huggingface.co/IamBusy/OpenJev-Vision)
- [数据集](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1)

合成场景：小 CNN＋额外提供的 64 状态先验，输出固定世界空间的分布。图片编码一次，同一视图的多个事件查询重用该分布；换图必须重新编码。8,192 是已发布原创合成图像总量，包含不同划分和测试分布，不是训练图片数。

公开图片：冻结 DINOv2-small，训练小预测头；Pets 为 37 类线性头，CLEVR-4 对比联合、独立和低秩交互头。事件执行为符号运算，无附加语言模型，不支持任意类别或自由问答。

第 3、4 张配图使用 `examples/vision/demo-output.json` 的固定开发样例、joint seed 17 已记录模型输出，原始场景为 `examples/vision/scene-0.png` 与 `scene-1.png`。左侧红色概率：53.33895% → 98.70932%；更多证据视图中左右同色为 2.29004%、同时红色为 1.01662%。图中显示两位小数。这是样例，不是测试准确率。

第 6 张配图使用 CLEVR-4 未见颜色／形状组合：80 张图、20 种留出组合、三种子均值±样本标准差。joint：0.00±0.00%；independent：63.33±1.91%；binding：60.00±3.75%。各方法共用冻结特征；这些结果不代表所有联合概率结构的能力。

未做 Nimble、Kev、OpenJev-Vision 的同题横向评测；也未找到本项目与 Qwen-VL 的同条件比较。因此正文、图片均不声称更强、更准、更快或可直接替代。

## 素材权限

本次为原创排版和原仓库原创合成场景的组合。未使用参考帖配图、第三方头像或 logo。代码与原创合成图像按原仓库 Apache-2.0 许可。
