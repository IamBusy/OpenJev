# Jev火了，我们把思路搬进了图像

Jev 这几天真火了。Vercel 9 月 18 日刚宣布，它成为 AI Gateway 历史上采用速度最快的模型。[来源](https://vercel.com/blog/ai-gateway-jev-model-launch)

我最感兴趣的是它的输出方式：直接给出选择、评分和概率，让程序接着做决策。

还有个细节：Jev 官方的 Doom 演示，输入其实是文本化的游戏状态。[官方说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

这让我冒出一个念头：那直接输入图片呢？👀

于是，我们把受 Jev 启发的独立视觉实验开源了：OpenJev-Vision。

核心思路：一张图编码一次，多个问题共享同一个概率分布。

比如在合成场景里：
🔴 左边的物体是红色的概率？
🔵 左右两个物体颜色相同的概率？
🧩 两个物体同时是红色的概率？

这些问题都从同一份概率表示里得到答案。换一张遮挡更少的图，还能观察补充视觉证据后，判断如何变化。

这次把能动手的东西都放出来了：
✅ 8,192 张原创合成图像与精确概率目标
✅ 宠物照片、颜色／形状组合实验
✅ 训练权重、推理示例、训练和评测代码
✅ 三组随机种子的结果与失败案例

目前是支持预定义类别与事件的研究版 v0.1；合成场景需要额外先验。项目与 TypeSafe 无隶属关系，采用独立实现。

如果你也被 Jev 的「直接做概率决策」吸引，欢迎来看看这个视觉方向的小实验。代码、数据、权重都能拿走研究 🌱

欢迎 Star，也欢迎一起提问题、改模型！

GitHub：[IamBusy/OpenJev-Vision](https://github.com/IamBusy/OpenJev-Vision)
模型权重：[IamBusy/OpenJev-Vision](https://huggingface.co/IamBusy/OpenJev-Vision)

#Jev #OpenJevVision #开源项目 #人工智能 #计算机视觉 #多模态 #独立开发

---

## 配图顺序

1. 封面：OpenJev-Vision；「Jev 火了／我们把思路／搬进了图像」；「代码 · 数据 · 权重 已公开」；「受 Jev 启发的独立开源实验」。
2. [遮挡前后示例](../../reports/vision-v01/demo.png)：说明补充视觉证据后，概率如何变化。蓝色为模型预测，绿色为合成过程的精确后验；这是固定开发集示例。
3. [完整对比图](../../reports/vision-v01/comparison.png)：展示基线结果，配文注明「小规模受控实验，完整结果与失败案例均公开」。

事实依据：[中文视觉说明](../VISION.zh-CN.md)、[完整实验报告](../../reports/vision-v01/RESULTS.md)。
