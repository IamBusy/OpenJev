"""Live CPU demo of the published, fixed-vocabulary synthetic vision model."""

import hashlib
import html
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from model import SceneModel
from PIL import Image
from query import answer_questions

ROOT = Path(__file__).parent
REPO = "IamBusy/OpenJev-Vision"
REVISION = "8cf6cbd39a7dc72840c9b52810a3f672786566b0"
QUESTIONS = {
    "Left object is red": {"type": "noul", "event": "red(left)"},
    "Right object is red": {"type": "noul", "event": "red(right)"},
    "Left and right have the same color": {"type": "noul", "event": "same_color(left, right)"},
    "Both left and right are red": {"type": "noul", "event": "red(left) and red(right)"},
}
CSS = """
.gradio-container {max-width:1120px!important; margin:auto!important; font-family:Inter,Arial,sans-serif!important}
.hero {padding:30px 0 22px;border-bottom:1px solid #dbe5e4;margin-bottom:20px}
.eyebrow {font-size:12px;letter-spacing:.13em;color:#14766b;font-weight:700}
.hero h1 {font-size:42px;line-height:1.1;letter-spacing:-1.5px;margin:12px 0;color:#102c2c}
.hero p {font-size:17px;line-height:1.6;max-width:760px;color:#496060}
.result-row {margin:18px 0;color:#183939}
.result-label {display:flex;justify-content:space-between;gap:15px;font-size:14px}
.track {height:9px;background:#e8efed;border-radius:6px;margin-top:8px;overflow:hidden}
.fill {height:100%;background:#178b7e;border-radius:6px}
.result-foot {font-size:12px;color:#5d7170;margin-top:22px}
.answer {font-size:23px;color:#14766b;padding:15px 0}
.status {font-size:13px;color:#14766b;padding:8px 0}
"""


@lru_cache(maxsize=1)
def get_model():
    torch.set_num_threads(2)
    checkpoint = ROOT / ".model"
    checkpoint.mkdir(exist_ok=True)
    manifest = json.loads(
        Path(hf_hub_download(REPO, "FILE_MANIFEST.json", revision=REVISION)).read_text()
    )
    for source, destination in (
        ("synthetic-joint.safetensors", "model.safetensors"),
        ("synthetic-joint.config.json", "config.json"),
        ("synthetic-joint.calibration.json", "calibration.json"),
    ):
        data = Path(hf_hub_download(REPO, source, revision=REVISION)).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest[source]:
            raise ValueError("Published model file failed hash verification")
        (checkpoint / destination).write_bytes(data)
    return SceneModel(checkpoint, device="cpu")


def bars(posterior):
    answers = answer_questions(posterior, QUESTIONS)
    rows = []
    for label, answer in answers.items():
        value = answer["noul"] * 100
        rows.append(
            f'<div class="result-row"><div class="result-label">'
            f"<span>{html.escape(label)}</span><strong>{value:.2f}%</strong></div>"
            f'<div class="track"><div class="fill" style="width:{value:.4f}%">'
            "</div></div></div>"
        )
    return "".join(rows) + (
        '<div class="result-foot">Model estimates from one shared distribution. '
        "These are not certainty guarantees.</div>"
    )


def encode(view):
    index = 0 if view == "Occluded view" else 1
    image = Image.open(ROOT / f"scene-{index}.png").convert("RGB")
    prior = json.loads((ROOT / f"scene-{index}-prior.json").read_text())
    posterior = get_model().posterior(image, prior)
    return (
        image.resize((768, 256), Image.Resampling.NEAREST),
        posterior.tolist(),
        bars(posterior),
        '<div class="status">1 image encoding · 64 possible scenes · 4 questions answered</div>',
        "Change the event below to query this same distribution.",
    )


def ask(event, posterior):
    if posterior is None:
        return "Please select a view and wait for the model to load."
    try:
        probability = answer_questions(posterior, {"answer": {"type": "noul", "event": event}})[
            "answer"
        ]["noul"]
    except (ValueError, KeyError, TypeError) as exc:
        return f"Unsupported event: {html.escape(str(exc))}"
    return (
        f'<div class="answer">P({html.escape(event)}) = {probability:.2%}</div>'
        '<div class="result-foot">Answered from the stored distribution. '
        "No additional image encoding.</div>"
    )


with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="teal"), title="OpenJev-Vision") as demo:
    gr.HTML("""<div class="hero"><div class="eyebrow">OPEN RESEARCH · LIVE CPU DEMO</div>
    <h1>One image. A distribution.<br>Several questions.</h1>
    <p>Choose a view, then compose questions about it. A small trained CNN reads the
    image and a supplied prior once; each answer is computed from the same
    distribution over 64 possible scenes.</p></div>""")
    posterior_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            view = gr.Radio(
                ["Occluded view", "More evidence"],
                value="Occluded view",
                label="1 · Choose the visual evidence",
            )
            image = gr.Image(
                label="Fixed development example · left / center / right",
                interactive=False,
                height=230,
            )
            status = gr.HTML()
            gr.Markdown(
                "The gray panels hide objects. Both views come from the **same fixed development episode**. The supplied prior is kept with its example."
            )
        with gr.Column(scale=1):
            gr.Markdown("### Four questions. One distribution.")
            results = gr.HTML()
    gr.Markdown("### 2 · Ask another question without encoding the image again")
    event = gr.Textbox(value="red(left) and square(right)", label="Event expression", max_lines=2)
    submit = gr.Button("Query the stored distribution", variant="primary")
    answer = gr.HTML()
    gr.Examples(
        [
            ["red(left)"],
            ["not red(left)"],
            ["same_color(left, right)"],
            ["red(left) and square(right)"],
            ["red(left) or red(center) or red(right)"],
        ],
        inputs=event,
    )
    with gr.Accordion("Scope, model and reproducibility", open=False):
        gr.Markdown(f"""
This demo runs the **published synthetic-joint seed-17 checkpoint** on CPU.
Its model files are pinned to `{REVISION}` and verified against the published hashes.
The examples are fixed development views, not selected test successes.

Supported scene vocabulary: three positions (left, center, right), two colors
(red, blue), and two shapes (circle, square). You can combine predicates with
`and`, `or`, `not`, `same_color`, and `same_shape`.

This is a fixed-vocabulary research demo. It does not accept arbitrary photographs
or understand arbitrary natural-language questions. The full project also has
separate DINOv2-based Pets and CLEVR-4 experiments. Joint prediction degrades on
new dependency structures; the published report includes those failures.

[Source and quick start](https://github.com/IamBusy/OpenJev-Vision) ·
[Weights](https://huggingface.co/IamBusy/OpenJev-Vision) ·
[Dataset](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) ·
[Results](https://github.com/IamBusy/OpenJev-Vision/blob/main/reports/vision-v01/RESULTS.md)

Independent project inspired by TypeSafe's Jev; not affiliated with TypeSafe.
Code and the original synthetic scenes are Apache-2.0.
""")
    view.change(encode, inputs=view, outputs=[image, posterior_state, results, status, answer])
    submit.click(ask, inputs=[event, posterior_state], outputs=answer)
    event.submit(ask, inputs=[event, posterior_state], outputs=answer)
    demo.load(encode, inputs=view, outputs=[image, posterior_state, results, status, answer])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get(
            "GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
        ),
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
    )
