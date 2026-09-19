import json

import numpy as np
import pytest
import torch

from openjev.vision.data import build_split
from openjev.vision.model import SceneModel, VisualPosterior, image_tensor, save_checkpoint
from openjev.vision.query import answer_questions, compile_event, question_from_text
from openjev.vision.render import render_scene
from openjev.vision.world import WORLDS, factorize, posterior


def test_missing_evidence_preserves_prior():
    prior = np.arange(1, 65, dtype=float)
    prior /= prior.sum()
    np.testing.assert_allclose(posterior(prior, [-1] * 6), prior)


def test_observation_likelihood_has_correct_odds():
    p = posterior(np.ones(64), [1, -1, -1, -1, -1, -1], sensor_error=0.1)
    assert p @ WORLDS[:, 0] == pytest.approx(0.9)


def test_factorization_loses_correlation_not_marginals():
    p = np.zeros(64)
    p[0] = p[-1] = 0.5
    q = factorize(p)
    np.testing.assert_allclose(p @ WORLDS, q @ WORLDS)
    event = compile_event("red(left) and red(right)")
    assert p @ event == pytest.approx(0.5)
    assert q @ event == pytest.approx(0.25)


def test_query_semantics_and_partitions():
    p = np.ones(64) / 64
    questions = {
        "yes": {"type": "noul", "event": "red(left)"},
        "no": {"type": "noul", "event": "not red(left)"},
        "count": "How many objects are red?",
        "choice": {"type": "choice", "criteria": {"red": "red(left)", "blue": "blue(left)"}},
    }
    result = answer_questions(p, questions)
    assert result["yes"]["noul"] + result["no"]["noul"] == pytest.approx(1)
    assert result["count"]["score"] == pytest.approx(1.5)
    assert result["count"]["probabilities"]["3"] == pytest.approx(0.125)
    assert result["choice"]["probabilities"] == {"red": 0.5, "blue": 0.5}
    assert answer_questions(p, dict(reversed(list(questions.items())))) == result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('touch bad')",
        "red(left.__class__)",
        "WORLDS[0]",
        "red(left, right)",
        "red(foo)",
        "1",
        "red(left, x=1)",
    ],
)
def test_no_python_execution(expression):
    with pytest.raises(ValueError):
        compile_event(expression)


def test_invalid_partition_rejected():
    with pytest.raises(ValueError, match="partition"):
        answer_questions(
            np.ones(64),
            {"x": {"type": "choice", "criteria": {"a": "red(left)", "b": "red(right)"}}},
        )


def test_language_parser_is_explicitly_bounded():
    assert question_from_text("Is the left object red?")["event"] == "red(left)"
    with pytest.raises(ValueError, match="controlled"):
        question_from_text("What does the image mean?")


def test_renderer_only_depends_on_observations():
    a = np.asarray(render_scene([-1, -1, 1, 0, 0, 1], seed=12))
    b = np.asarray(render_scene([-1, -1, 1, 0, 0, 1], seed=12))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (64, 192, 3)
    assert not np.array_equal(a, np.asarray(render_scene([-1] * 6, seed=13)))


def test_episode_split_and_posterior():
    a = build_split(5, 7, "train", 0.08)
    b = build_split(5, 8, "dev", 0.08)
    assert not set(a["groups"]) & set(b["groups"])
    assert len(a["images"]) == 10
    for i in range(10):
        np.testing.assert_allclose(
            a["targets"][i], posterior(a["priors"][i], a["observed"][i]), atol=1e-7
        )
    assert a["groups"][0] == a["groups"][1]
    assert a["visibility"][0] < 3 and a["visibility"][1] == 3
    # Appearance is paired; all common observed slots have identical pixels.
    for slot in range(3):
        if a["observed"][0, 2 * slot] >= 0:
            np.testing.assert_array_equal(
                a["images"][0, :, slot * 64 : (slot + 1) * 64],
                a["images"][1, :, slot * 64 : (slot + 1) * 64],
            )


@pytest.mark.parametrize("variant", ["joint", "independent", "evidence"])
def test_models_normalized_differentiable_and_reloadable(variant, tmp_path):
    torch.manual_seed(17)
    model = VisualPosterior(variant, hidden_dim=16)
    images = image_tensor(np.asarray(render_scene([1, 0, -1, -1, 0, 1])))
    priors = torch.ones(1, 64) / 64
    outputs = model(images, priors)
    logp = model.log_posterior(outputs, priors)
    assert torch.isfinite(logp).all()
    assert logp.exp().sum().item() == pytest.approx(1, abs=1e-5)
    loss = -logp[:, 3].mean()
    loss.backward()
    assert model.encoder.net[0].weight.grad.abs().sum() > 0
    save_checkpoint(tmp_path, model, {"seed": 17})
    loaded = SceneModel(tmp_path, "cpu")
    np.testing.assert_allclose(
        loaded.posterior(np.asarray(render_scene([1, 0, -1, -1, 0, 1]))),
        logp.detach().exp().numpy()[0],
        rtol=1e-5,
    )
    assert json.loads((tmp_path / "config.json").read_text())["variant"] == variant


def test_slot_cropping_keeps_slots_separate():
    model = VisualPosterior("evidence", hidden_dim=16).eval()
    image = np.asarray(render_scene([0, 0, 0, 0, 0, 0]))
    other = image.copy()
    other[:, 128:] = 0
    with torch.inference_mode():
        a = model.encoder(image_tensor(image))
        b = model.encoder(image_tensor(other))
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2], b[:, 2])
