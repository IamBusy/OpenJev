"""Compose original editorial cards from recorded experiment outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / "images"
VECTOR = OUT / "editable"
ASSETS.mkdir(exist_ok=True)
VECTOR.mkdir(exist_ok=True)
plt.rcParams["svg.fonttype"] = "none"
W, H = 1080, 1440
BG, INK, MUTED = "#F4F2EB", "#192D38", "#596973"
BLUE, TEAL, ORANGE, LINE = "#244CE7", "#077F74", "#CF563D", "#D7DBD6"
FONT = {
    "zh": "/System/Library/Fonts/STHeiti Light.ttc",
    "bold": "/System/Library/Fonts/STHeiti Medium.ttc",
    "en": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "num": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
}
REC = json.loads((ROOT / "examples/vision/demo-output.json").read_text())
QA = []


def text(ax, x, y, s, size=32, color=INK, kind="zh", **kw):
    return ax.text(
        x,
        y,
        s,
        fontsize=size * 0.72,
        color=color,
        fontproperties=FontProperties(fname=FONT[kind]),
        va="top",
        linespacing=1.2,
        **kw,
    )


def box(ax, x, y, w, h, color="white", radius=26, edge=LINE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=color,
            edgecolor=edge,
            linewidth=1,
        )
    )


def base(n, category, dark=False):
    bg = INK if dark else BG
    f = plt.figure(figsize=(10.8, 14.4), dpi=100, facecolor=bg)
    ax = f.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    fg = BG if dark else INK
    mute = "#B8C6C9" if dark else MUTED
    text(ax, 68, 59, "Mr.Ming AI / 实验笔记", 25, fg, "bold")
    text(ax, 1012, 61, category, 23, mute, ha="right")
    ax.plot([68, 1012], [113, 113], color="#50616A" if dark else LINE, lw=1)
    ax.plot([68, 1012], [1338, 1338], color="#50616A" if dark else LINE, lw=1)
    text(ax, 68, 1365, "OpenJev-Vision · 独立开源研究", 22, mute)
    text(ax, 1012, 1363, f"{n:02d} / 08", 23, fg, "num", ha="right")
    return f, ax


def title(ax, lines, size=80, color=INK, y=168):
    text(ax, 68, y, lines, size, color, "bold")


def scene(f, x, y, width, view):
    a = f.add_axes([x / W, 1 - (y + width / 3) / H, width / W, width / 3 / H])
    a.imshow(Image.open(ROOT / f"examples/vision/scene-{view}.png"), interpolation="nearest")
    a.axis("off")


def save(f, name):
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    issues = []
    for ax in f.axes:
        for item in ax.texts:
            b = item.get_window_extent(renderer)
            if b.x0 < 30 or b.x1 > W - 30 or b.y0 < 25 or b.y1 > H - 25:
                issues.append({"text": item.get_text(), "bounds": list(b.bounds)})
    QA.append({"file": f"{name}.png", "canvas": [W, H], "text_overflow": issues})
    f.savefig(ASSETS / f"{name}.png", dpi=100, facecolor=f.get_facecolor())
    f.savefig(VECTOR / f"{name}.svg", facecolor=f.get_facecolor())
    plt.close(f)


def cover():
    f, ax = base(1, "从热点到自己的实验")
    text(ax, 68, 170, "Jev 之后，", 89, INK, "bold")
    text(ax, 68, 286, "我想让 AI", 104, INK, "bold")
    text(ax, 68, 419, "看图做判断。", 108, BLUE, "bold")
    text(ax, 72, 582, "从 Nimble、Kev，", 35, MUTED)
    text(ax, 72, 634, "到我的视觉概率实验", 35, MUTED)
    box(ax, 68, 742, 944, 431, INK, edge=INK)
    labels = [
        ("01", "Nimble", "文本判断"),
        ("02", "Kev", "本地决策"),
        ("03", "OpenJev-Vision", "视觉概率"),
    ]
    for i, (num, name, label) in enumerate(labels):
        y = 787 + i * 127
        text(ax, 103, y + 7, num, 29, "#92A7FF", "num")
        text(ax, 169, y, name, 43 if i < 2 else 40, "white", "num")
        text(ax, 966, y + 6, label, 31, "#D4DFE4", ha="right")
        if i < 2:
            ax.plot([103, 966], [y + 85, y + 85], color="#3D515E", lw=1)
    text(ax, 70, 1230, "代码 · 数据 · 权重 · 失败案例", 31, TEAL, "bold")
    save(f, "01-cover")


def projects():
    f, ax = base(2, "三个项目 / 三个切入点")
    title(ax, "同一个热潮，\n不同的切入点。", 80)
    cards = [
        (
            427,
            "01 / Bespoke Nimble",
            "9B · 文本判断",
            "只改一个关键事实的对照样本，",
            "训练模型区分不同条件下的答案。",
            BLUE,
            "#E9EDFF",
        ),
        (
            715,
            "02 / Kev",
            "0.5B 起步 · 本地决策",
            "早期版本展示笔记本训练的可能性。",
            "现另有 0.6B / 4B / 8B 预览版。",
            TEAL,
            "#E6F0EA",
        ),
        (
            1003,
            "03 / 我的 OpenJev-Vision",
            "视觉概率 · 独立实验",
            "一张图得到一个概率分布，",
            "从中回答多个预定义事件问题。",
            INK,
            "white",
        ),
    ]
    for y, small, main, a, b, c, bg in cards:
        box(ax, 68, y, 944, 258, bg)
        text(ax, 100, y + 28, small, 27, c, "bold")
        text(ax, 100, y + 78, main, 42, INK, "bold")
        text(ax, 100, y + 147, a, 30, MUTED)
        text(ax, 100, y + 191, b, 30, MUTED)
    text(ax, 70, 1290, "项目各有任务范围；这里不做性能排名。", 23, MUTED)
    save(f, "02-three-projects")


def questions():
    f, ax = base(3, "视觉概率 / 固定开发样例")
    title(ax, "一张图，\n问的不止一件事。", 78)
    box(ax, 68, 416, 944, 374)
    text(ax, 100, 442, "输入：合成图像 + 额外提供的先验", 28, MUTED)
    scene(f, 112, 498, 856, 1)
    keys = [
        ("左边是红色？", "left_red"),
        ("左右颜色相同？", "same_color"),
        ("左右同时是红色？", "both_red"),
    ]
    for i, (label, key) in enumerate(keys):
        y = 845 + i * 105
        p = REC["views"][1]["model"][key]["noul"]
        text(ax, 80, y, label, 36, INK, "bold")
        text(ax, 997, y - 5, f"{p:.2%}", 45, BLUE, "num", ha="right")
        ax.add_patch(Rectangle((80, y + 61), 917, 8, facecolor="#DFE4E3", edgecolor="none"))
        ax.add_patch(Rectangle((80, y + 61), 917 * p, 8, facecolor=BLUE, edgecolor="none"))
    text(ax, 72, 1192, "这些答案，都从同一份分布里计算。", 33, TEAL, "bold")
    text(ax, 72, 1255, "固定开发样例的模型估计，不代表整体准确率。", 24, MUTED)
    save(f, "03-one-image-many-questions")


def evidence():
    f, ax = base(4, "观察 / 遮挡与证据")
    title(ax, "换一张图，\n判断怎么变？", 82)
    text(ax, 71, 391, "同一个样例：左边是红色的概率", 34, MUTED)
    for i, label in enumerate(["视图 A / 存在遮挡", "视图 B / 更多可见证据"]):
        y = 489 + i * 370
        box(ax, 68, y, 944, 320)
        text(ax, 99, y + 28, label, 29, TEAL if i else MUTED, "bold")
        scene(f, 100, y + 92, 572, i)
        p = REC["views"][i]["model"]["left_red"]["noul"]
        text(ax, 712, y + 130, f"{p:.2%}", 60, BLUE, "num")
        text(ax, 715, y + 218, "模型估计", 24, MUTED)
    text(ax, 72, 1230, "换图需重新编码；每张图内的查询可共享分布。", 26, INK)
    text(ax, 72, 1280, "同一固定开发样例、同一先验；不是连续视频记忆。", 22, MUTED)
    save(f, "04-more-visual-evidence")


def implementation():
    f, ax = base(5, "回应 / 上一条的技术追问")
    title(ax, "DINO + 分类头？\n这部分，确实是。", 75)
    text(ax, 71, 385, "公开图片实验的流程", 32, MUTED)
    steps = [
        ("01", "冻结 DINOv2-small", "提取已有视觉特征"),
        ("02", "训练小预测头", "得到固定类别上的概率分布"),
        ("03", "按事件计算概率", "例如：把猫品种的概率加起来"),
    ]
    for i, (num, main, sub) in enumerate(steps):
        y = 480 + i * 214
        box(ax, 68, y, 944, 170, "#E9EDFF" if i == 2 else "white")
        text(ax, 99, y + 35, num, 33, BLUE, "num")
        text(ax, 179, y + 28, main, 40, INK, "bold")
        text(ax, 181, y + 99, sub, 28, MUTED)
        if i < 2:
            ax.annotate(
                "",
                xy=(540, y + 207),
                xytext=(540, y + 175),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2),
            )
    text(ax, 72, 1129, "重点：共享查询、校准与可复现实验。", 34, TEAL, "bold")
    text(ax, 72, 1201, "没有额外语言模型；类别和事件需预先定义。", 27, MUTED)
    text(ax, 72, 1250, "合成场景另用小 CNN + 先验，是另一组实验。", 25, MUTED)
    save(f, "05-what-is-actually-built")


def results():
    f, ax = base(6, "结果 / 一个重要的失败")
    title(ax, "失败，\n也要放出来。", 85)
    text(ax, 71, 394, "CLEVR-4 · 未见颜色 / 形状组合", 32, MUTED)
    text(ax, 71, 449, "组合分类准确率（%），越高越好", 27, MUTED)
    data = [
        ("joint 整体分类头", 0, 0, ORANGE),
        ("independent 独立属性", 63.33, 1.91, BLUE),
        ("binding 低秩交互", 60, 3.75, TEAL),
    ]
    for i, (label, val, sd, color) in enumerate(data):
        y = 543 + i * 175
        text(ax, 72, y, label, 31, INK, "bold")
        text(ax, 1007, y - 4, f"{val:.2f} ± {sd:.2f}", 34, color, "num", ha="right")
        ax.add_patch(Rectangle((73, y + 66), 936, 40, facecolor="#E0E4DF", edgecolor="none"))
        if val:
            ax.add_patch(
                Rectangle((73, y + 66), 936 * val / 100, 40, facecolor=color, edgecolor="none")
            )
        else:
            ax.plot([73, 73], [y + 62, y + 110], color=color, lw=3)
    for x, s in [(73, "0"), (541, "50"), (1009, "100%")]:
        text(
            ax,
            x,
            1011,
            s,
            22,
            MUTED,
            "en",
            ha="left" if x == 73 else "right" if x == 1009 else "center",
        )
    box(ax, 68, 1069, 944, 154, "#F3E6DE")
    text(ax, 100, 1097, "保留一个分布，", 35, INK, "bold")
    text(ax, 100, 1148, "并不自动带来组合泛化。", 35, INK, "bold")
    text(ax, 72, 1253, "80 张未见组合图 · 20 种留出组合 · 3 种子均值 ± 标准差", 23, MUTED)
    text(ax, 72, 1294, "仅针对本次预测头；来源：仓库 vision-v01 实验报告。", 22, MUTED)
    save(f, "06-publish-the-failures")


def materials():
    f, ax = base(7, "开源 / 可以直接动手")
    title(ax, "代码之外，\n把实验也交出来。", 79)
    cards = [
        (68, 448, "8,192", "原创合成图像", "含训练与各评测划分", BLUE),
        (553, 448, "3 组", "随机种子评测", "成功与失败一起报告", TEAL),
        (68, 756, "可下载", "训练权重", "先跑示例，再动手训练", TEAL),
        (553, 756, "可复现", "基线与代码", "公开数据、配置与教程", BLUE),
    ]
    for x, y, big, a, b, c in cards:
        box(ax, x, y, 459, 276)
        text(ax, x + 29, y + 31, big, 62, c, "num" if x == 68 and y == 448 else "bold")
        text(ax, x + 29, y + 133, a, 35, INK, "bold")
        text(ax, x + 29, y + 207, b, 24, MUTED)
    text(ax, 72, 1098, "准备数据 → 训练 → 校准 → 评测 → 推理", 31, INK, "bold")
    text(ax, 72, 1173, "当前范围：预定义类别和事件；合成场景需先验。", 26, MUTED)
    text(ax, 72, 1226, "还没有与 Qwen-VL 的同条件对比。", 27, MUTED)
    text(ax, 72, 1280, "研究原型；尚不支持任意图片自由问答。", 24, MUTED)
    save(f, "07-the-open-materials")


def invitation():
    f, ax = base(8, "继续 / 一起把问题测清楚", True)
    title(ax, "下一步，\n你想先测什么？", 84, BG)
    options = ["与 Qwen-VL 做同条件对比", "加入更复杂的组合事件", "在自己的固定任务上复现"]
    for i, s in enumerate(options):
        y = 452 + i * 112
        ax.add_patch(Circle((93, y + 22), 22, facecolor="#91A7FF", edgecolor="none"))
        text(ax, 93, y + 6, str(i + 1), 26, INK, "num", ha="center")
        text(ax, 140, y, s, 36, BG, "bold")
    box(ax, 68, 835, 944, 242, "#F4F2EB", edge="#F4F2EB")
    text(ax, 101, 869, "GitHub 搜索", 26, MUTED, "bold")
    text(ax, 101, 924, "IamBusy/OpenJev-Vision", 49, BLUE, "num")
    text(ax, 103, 1008, "代码 · 权重 · 数据 · 教程 · 完整结果", 28, INK)
    text(ax, 72, 1140, "延伸阅读", 24, "#B8C6C9")
    text(ax, 72, 1188, "Nimble / bespokelabsai/nimble", 28, BG, "en")
    text(ax, 72, 1233, "Kev / jaredpalmer/kev", 28, BG, "en")
    text(ax, 72, 1290, "受 Jev 启发的独立实现，与 TypeSafe 无隶属关系。", 23, "#B8C6C9")
    save(f, "08-join-the-experiment")


for make in [cover, projects, questions, evidence, implementation, results, materials, invitation]:
    make()

# Contact sheet for visual inspection; cards themselves remain original-size exports.
sheet = Image.new("RGB", (1200, 848), "#DFE2DC")
d = ImageDraw.Draw(sheet)
font = ImageFont.truetype(FONT["en"], 20)
for i, path in enumerate(sorted(ASSETS.glob("*.png"))):
    im = Image.open(path).convert("RGB")
    assert im.size == (W, H)
    im.thumbnail((285, 380), Image.Resampling.LANCZOS)
    x, y = 9 + (i % 4) * 300, 10 + (i // 4) * 424
    sheet.paste(im, (x, y))
    d.text((x, y + 387), path.name[:2], fill=INK, font=font)
sheet.save(OUT / "preview.jpg", quality=93)
(OUT / "visual-qa.json").write_text(json.dumps(QA, ensure_ascii=False, indent=2) + "\n")
assert not any(q["text_overflow"] for q in QA), QA
print(f"Created {len(QA)} cards: {ASSETS}")
