# Show HN 事实清单：供作者自行组织文字

状态：未代写或发布 HN 帖文。

HN Guidelines 明确要求 “Don't post generated text or AI-edited text.”
因此这份文件仅整理可以核实的项目事实与来源；作者需要自己写标题和说明，并亲自参与讨论。

规则：https://news.ycombinator.com/newsguidelines.html
Show HN：https://news.ycombinator.com/showhn.html
临时发布限制：https://news.ycombinator.com/showlim

## 项目事实

- 名称：OpenJev-Vision。
- 类型：视觉概率决策的开源研究工具。
- 合成实验：64×192 图片、给定 64-world 先验、小型训练 CNN；事件查询在共享分布上计算。
- 在线演示：已发布权重的 浏览器本地推理；两个固定开发视图；支持继续查询同一个分布。
- 独立的公开图像实验：冻结 DINOv2-small 特征，训练 Pets/CLEVR-4 预测头。
- 训练种子：17、23、42；模型选择、校准、测试分开。
- 范围：固定词汇/事件定义；不支持任意场景和自由语言视觉问答。
- 不属于 TypeSafe，不复现 Jev 未公开的架构、权重或 RLCD。
- 代码/原创合成数据 Apache-2.0；公共图像及其预测头保留各自许可。

## 可讨论的实验结果

- 分布内保留相关性，对比从同一个模型边缘化再独立化，有助于复合事件概率预测。
- 直接预测联合分布的模型在未见依赖结构上退化。
- CLEVR-4 的 80 张未见组合图像：三种子均值±样本标准差，独立属性 63.33%±1.91pp，低秩组合 60.00%±3.75pp，平坦联合类别 0%±0pp。
- 上述结果不能推出所有联合概率架构都无法泛化。

## 可引用链接

- https://github.com/IamBusy/OpenJev-Vision
- https://huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo
- https://huggingface.co/IamBusy/OpenJev-Vision
- https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1
- https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md

发布时链接到可体验的项目；避免拉票、删除重发、夸张标题。原作者应能在场回答问题。
