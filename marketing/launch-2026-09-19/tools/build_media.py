"""Render publication graphics from the repository's recorded research outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "marketing/launch-2026-09-19/assets"
OUT.mkdir(parents=True, exist_ok=True)
FONTS = {
    "regular": FontProperties(fname="/System/Library/Fonts/Supplemental/Arial.ttf"),
    "bold": FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    "zh": FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
}
BG, INK, MUTED, TEAL, LINE = "#f5f5ed", "#102f31", "#50676a", "#0d877b", "#d6e2dd"
REC = json.loads((ROOT / "examples/vision/demo-output.json").read_text())
KEYS = ["left_red", "right_red", "same_color", "both_red"]
LABELS = {
    "en": ["Left is red", "Right is red", "Same color", "Both are red"],
    "zh": ["左侧是红色", "右侧是红色", "左右颜色相同", "左右都是红色"],
}


def fig(w=1600, h=900):
    f = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
    ax = f.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return f, ax


def text(ax, x, y, s, size=24, color=INK, bold=False, zh=False, **kw):
    ax.text(
        x,
        y,
        s,
        fontsize=size * 0.72,
        color=color,
        fontproperties=FONTS["zh" if zh else "bold" if bold else "regular"],
        va="top",
        **kw,
    )


def panel(ax, x, y, w, h, color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=16",
            facecolor=color,
            edgecolor=LINE,
            linewidth=1,
        )
    )


def scene(f, x, y, w, view, total_w=1600, total_h=900):
    a = f.add_axes([x / total_w, 1 - (y + w / 3) / total_h, w / total_w, w / 3 / total_h])
    a.imshow(Image.open(ROOT / f"examples/vision/scene-{view}.png"), interpolation="nearest")
    a.axis("off")


def footer(ax, h=900, zh=False):
    ax.plot([64, 1536], [h - 88, h - 88], color=LINE, linewidth=1)
    text(ax, 64, h - 63, "github.com/IamBusy/OpenJev-Vision", 20, bold=True)
    text(
        ax,
        1536,
        h - 63,
        "固定场景词汇 · 实验性研究" if zh else "FIXED SCENE VOCABULARY · EXPERIMENTAL RESEARCH",
        16,
        MUTED,
        zh=zh,
        ha="right",
    )


def save(f, name, vector=False):
    f.savefig(OUT / f"{name}.png", dpi=100, facecolor=BG)
    if vector:
        f.savefig(OUT / f"{name}.svg", facecolor=BG)
    plt.close(f)


def cover(lang):
    zh = lang == "zh"
    f, ax = fig()
    text(ax, 64, 45, "OpenJev-Vision", 27, bold=True)
    text(ax, 1536, 49, "OPEN RESEARCH / 01", 18, TEAL, bold=True, ha="right")
    text(
        ax,
        64,
        118,
        "一张图，一次编码。\n从同一分布回答多个问题。"
        if zh
        else "One image encoding.\nSeveral probability queries.",
        64,
        bold=True,
        zh=zh,
        linespacing=1.22,
    )
    text(
        ax,
        64,
        295,
        "代码、权重、数据与失败案例，全部公开。"
        if zh
        else "Open code, weights, data — and the failures worth studying.",
        26,
        MUTED,
        zh=zh,
    )
    panel(ax, 64, 371, 698, 370)
    panel(ax, 798, 371, 738, 370)
    text(
        ax,
        90,
        399,
        "视觉证据 / 固定开发样例" if zh else "VISUAL EVIDENCE / FIXED DEVELOPMENT EXAMPLE",
        18,
        TEAL,
        bold=True,
        zh=zh,
    )
    scene(f, 94, 452, 636, 1)
    text(
        ax,
        90,
        699,
        "小型 CNN + 提供的先验 → 64 个可能场景"
        if zh
        else "Small CNN + supplied prior → 64 possible scenes",
        18,
        MUTED,
        zh=zh,
    )
    text(
        ax,
        829,
        399,
        "模型估计 / 同一个共享分布" if zh else "MODEL ESTIMATES / ONE SHARED DISTRIBUTION",
        18,
        TEAL,
        bold=True,
        zh=zh,
    )
    for j, key in enumerate(KEYS):
        y = 450 + j * 63
        val = REC["views"][1]["model"][key]["noul"]
        text(ax, 830, y, LABELS[lang][j], 23, zh=zh)
        text(ax, 1505, y, f"{100 * val:.2f}%", 24, bold=True, ha="right")
        ax.add_patch(Rectangle((830, y + 34), 675, 7, facecolor="#e7efea", edgecolor="none"))
        ax.add_patch(Rectangle((830, y + 34), 675 * val, 7, facecolor=TEAL, edgecolor="none"))
    footer(ax, zh=zh)
    save(f, f"cover-{lang}", True)


def evidence(view, lang="en", highlight=False):
    zh = lang == "zh"
    f, ax = fig()
    text(ax, 64, 45, "OpenJev-Vision", 27, bold=True)
    text(ax, 1536, 49, "RECORDED MODEL OUTPUTS", 17, TEAL, bold=True, ha="right")
    title = (
        ("遮挡存在时，保留不确定性。" if view == 0 else "看见更多证据，重新估计概率。")
        if zh
        else (
            "Hidden evidence leaves uncertainty."
            if view == 0
            else "More evidence changes the distribution."
        )
    )
    text(ax, 64, 125, title, 56, bold=True, zh=zh)
    text(
        ax,
        64,
        210,
        "同一个固定开发样例的两个视图；使用各自图像与随附先验。"
        if zh
        else "Two views of the same fixed development example, each with its supplied prior.",
        25,
        MUTED,
        zh=zh,
    )
    panel(ax, 64, 298, 680, 443)
    panel(ax, 780, 298, 756, 443)
    text(
        ax,
        94,
        330,
        "遮挡视图"
        if zh and view == 0
        else "更多可见证据"
        if zh
        else "OCCLUDED VIEW"
        if view == 0
        else "MORE EVIDENCE",
        20,
        TEAL,
        bold=True,
        zh=zh,
    )
    scene(f, 96, 409, 616, view)
    text(
        ax,
        96,
        671,
        "每张图编码一次，再回答多个问题"
        if zh
        else "Encode this view once. Query it several times.",
        21,
        MUTED,
        zh=zh,
    )
    for j, key in enumerate(KEYS):
        y = 337 + j * 91
        val = REC["views"][view]["model"][key]["noul"]
        text(ax, 810, y, LABELS[lang][j], 26, zh=zh)
        text(ax, 1504, y, f"{100 * val:.2f}%", 30, bold=True, ha="right")
        ax.add_patch(Rectangle((810, y + 48), 694, 12, facecolor="#e7efea", edgecolor="none"))
        ax.add_patch(Rectangle((810, y + 48), 694 * val, 12, facecolor=TEAL, edgecolor="none"))
    footer(ax, zh=zh)
    save(f, f"evidence-{view}-{lang}")


def process():
    f, ax = fig()
    text(ax, 64, 45, "OpenJev-Vision", 27, bold=True)
    text(ax, 64, 126, "Keep a distribution. Compose the questions.", 55, bold=True)
    text(
        ax,
        64,
        217,
        "This demo runs a trained CNN; the event queries use explicit, fixed semantics.",
        26,
        MUTED,
    )
    for x, w, title, body in [
        (64, 410, "01 / SEE", "64 × 192 synthetic image\n+ supplied scene prior"),
        (580, 400, "02 / RETAIN", "A probability distribution\nover 64 possible scenes"),
        (1086, 450, "03 / QUERY", "red(left)\nsame_color(left, right)\nred(left) and red(right)"),
    ]:
        panel(ax, x, 350, w, 285)
        text(ax, x + 28, 389, title, 26, TEAL, bold=True)
        text(ax, x + 28, 458, body, 26, linespacing=1.6)
    for x in (490, 1000):
        ax.annotate(
            "",
            xy=(x + 66, 490),
            xytext=(x, 490),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
        )
    text(
        ax,
        64,
        714,
        "Additional queries reuse the distribution. New visual evidence needs a new encoding.",
        25,
        MUTED,
    )
    footer(ax)
    save(f, "how-it-works", True)


def result():
    f, ax = fig()
    text(ax, 64, 45, "OpenJev-Vision / research notes", 27, bold=True)
    text(ax, 64, 123, "Where the simple joint classifier fails.", 58, bold=True)
    text(
        ax,
        64,
        217,
        "CLEVR-4 unseen color / shape pairs · accuracy (%) · higher is better",
        26,
        MUTED,
    )
    labels = ["Flat joint head", "Independent attributes", "Low-rank binding"]
    vals = [0, 63.33, 60]
    std = [0, 1.91, 3.75]
    for j, (label, val, sd) in enumerate(zip(labels, vals, std)):
        y = 339 + j * 113
        text(ax, 64, y, label, 30, bold=True)
        ax.add_patch(Rectangle((600, y + 4), 680, 31, facecolor="#e5ece7", edgecolor="none"))
        if val:
            ax.add_patch(
                Rectangle(
                    (600, y + 4),
                    680 * val / 100,
                    31,
                    facecolor=TEAL if j == 1 else "#93aaa5",
                    edgecolor="none",
                )
            )
        text(ax, 1510, y, f"{val:.2f} ± {sd:.2f}", 30, bold=True, ha="right")
    text(
        ax,
        64,
        696,
        "80 held-out images · 20 withheld pair labels · 3 seeds (mean ± sample SD)",
        24,
        MUTED,
    )
    text(
        ax,
        64,
        742,
        "A failure of this flat label-space baseline; not a verdict on every joint architecture.",
        23,
        MUTED,
    )
    footer(ax)
    save(f, "research-result", True)


def outro():
    f, ax = fig()
    text(ax, 64, 45, "OpenJev-Vision", 27, bold=True)
    text(
        ax,
        64,
        158,
        "Try it. Reproduce it.\nFind the next failure case.",
        73,
        bold=True,
        linespacing=1.15,
    )
    text(
        ax,
        64,
        387,
        "Live browser demo · downloadable weights · data · three-seed evaluations",
        28,
        MUTED,
    )
    panel(ax, 64, 492, 1472, 179)
    text(ax, 94, 521, "START HERE", 19, TEAL, bold=True)
    text(ax, 94, 568, "huggingface.co/spaces/IamBusy/OpenJev-Vision-Demo", 31, bold=True)
    text(ax, 64, 726, "Fixed-vocabulary visual research. Independent of TypeSafe.", 24, MUTED)
    footer(ax)
    save(f, "try-it")


if __name__ == "__main__":
    for lang in ("en", "zh"):
        cover(lang)
        for view in (0, 1):
            evidence(view, lang)
    process()
    result()
    outro()
    for lang in ("en", "zh"):
        frames = [Image.open(OUT / f"evidence-{v}-{lang}.png").resize((960, 540)) for v in (0, 1)]
        frames[0].save(
            OUT / f"demo-{lang}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=[3500, 4500],
            loop=0,
        )
    paths = [
        "cover-en",
        "how-it-works",
        "evidence-0-en",
        "evidence-1-en",
        "research-result",
        "try-it",
    ]
    lines = []
    for name in paths:
        lines.extend([f"file '{name}.png'", "duration 7"])
    lines.append("file 'try-it.png'")
    (OUT / "video-frames.txt").write_text("\n".join(lines) + "\n")
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "source_commit": "4fa973e5212881d704def6818b0def190b094f12",
                "example": "Fixed development episode 0, views 0 and 1",
                "model": "synthetic-joint seed 17, calibrated",
                "probabilities_source": "examples/vision/demo-output.json",
                "benchmark_source": "reports/vision-v01/RESULTS.md",
                "video": "Graphic explainer with recorded model outputs, not a screen recording",
                "original_synthetic_images_license": "Apache-2.0",
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT)
