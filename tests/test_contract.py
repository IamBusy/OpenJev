import numpy as np
import pytest
import torch
from pydantic import ValidationError

from openjev.metrics import nll, probabilities, reliability
from openjev.model import DecisionHead
from openjev.schema import Question, Record, Request, query_text


def test_noul_means_probability_of_truth_not_abstention():
    question = Question(type="noul", instructions="The customer requests a refund.")
    names, descriptions = question.options()
    assert names == ["false", "true"]
    assert "false" in descriptions[0] and "true" in descriptions[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "choice", "instructions": "Choose", "criteria": {"a": "only one"}},
        {"type": "score", "instructions": "Score", "criteria": ["low"]},
        {"type": "noul", "instructions": "True?", "criteria": {"yes": "yes", "no": "no"}},
        {"type": "choice", "instructions": "Choose", "criteria": {"a": "same", "b": "same"}},
        {"type": "choice", "instructions": " ", "criteria": {"a": "a", "b": "b"}},
    ],
)
def test_invalid_contract_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        Question(**kwargs)


def test_non_finite_state_is_rejected():
    with pytest.raises(ValidationError):
        Request(state={"x": float("nan")}, questions={"q": {"type": "noul", "instructions": "x"}})


def test_question_ids_and_other_questions_are_not_model_inputs():
    q = Question(type="noul", instructions="A complete proposition")
    assert query_text({"b": 2, "a": 1}, q) == query_text({"a": 1, "b": 2}, q)
    assert query_text("State", q) == "A complete proposition\nState"


def test_candidate_permutation_and_mask_are_invariant():
    torch.manual_seed(17)
    head = DecisionHead(8, hidden_dim=16).eval()
    # Exercise a nonzero trained residual, not just the cosine initialization.
    torch.nn.init.normal_(head.residual[-1].weight)
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    options = torch.nn.functional.normalize(torch.randn(2, 5, 8), dim=-1)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    types = torch.tensor([0, 2])
    perm = torch.tensor([2, 0, 4, 1, 3])
    a = head(q, options, mask, types).softmax(-1)
    b = head(q, options[:, perm], mask[:, perm], types).softmax(-1)
    torch.testing.assert_close(a[:, perm], b)
    assert a[0, 3:].sum() == 0
    torch.testing.assert_close(a.sum(-1), torch.ones(2))
    one = head(q[:1], options[:1], mask[:1], types[:1]).softmax(-1)
    torch.testing.assert_close(one, a[:1])


def test_proper_score_and_temperature_math():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_allclose(probabilities(np.log(p)), p)
    optimum = nll(np.log(p), p)
    assert nll(np.log(np.array([[0.5, 0.5], [0.5, 0.5]])), p) > optimum
    assert probabilities(np.log(p), 2).max() < p.max()
    ece, bins = reliability(np.array([0.5, 0.5, 1.0]), np.array([0.0, 1.0, 1.0]))
    assert ece == 0
    assert sum(x["count"] for x in bins) == 3


def test_bad_target_cannot_enter_training():
    with pytest.raises(ValidationError):
        Record(
            id="a",
            group_id="g",
            source="test",
            source_id="a",
            split="train",
            task="x",
            state="state",
            question=Question(type="noul", instructions="True?"),
            target=[0.8, 0.8],
            label="true",
        )
