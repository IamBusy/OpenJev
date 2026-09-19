"""Validated public contract; probability quality is evaluated separately."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=4096)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def valid_criteria(self):
        if not self.instructions.strip():
            raise ValueError("instructions cannot be blank")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("choice needs 2–255 named candidates")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score needs 2–10 ordered level descriptions")
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) != {"false", "true"}:
                raise ValueError("noul criteria must describe exactly false and true")
        if self.criteria is not None:
            values = (
                list(self.criteria.values()) if isinstance(self.criteria, dict) else self.criteria
            )
            if any(not x.strip() or len(x) > 4096 for x in values):
                raise ValueError("candidate descriptions must contain 1–4096 characters")
            if len(set(values)) != len(values):
                raise ValueError("candidate descriptions must be distinct")
            if isinstance(self.criteria, dict) and any(not x.strip() for x in self.criteria):
                raise ValueError("candidate names cannot be blank")
        return self

    def options(self) -> tuple[list[str], list[str]]:
        if self.type == "choice":
            return list(self.criteria), list(self.criteria.values())
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))], list(self.criteria)
        if self.criteria is not None:
            return ["false", "true"], [self.criteria["false"], self.criteria["true"]]
        return ["false", "true"], [
            f"The following statement is false: {self.instructions}",
            f"The following statement is true: {self.instructions}",
        ]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Any
    questions: dict[str, Question] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def json_state(self):
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError("state must be text, an object, or an array")
        text = state_text(self.state)
        if not text.strip() or len(text) > 100000:
            raise ValueError("state must contain 1–100000 characters")
        if any(not k.strip() for k in self.questions):
            raise ValueError("question IDs cannot be blank")
        return self


def state_text(state: Any) -> str:
    return (
        state
        if isinstance(state, str)
        else json.dumps(
            state, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    )


def query_text(state: Any, question: Question) -> str:
    # IDs are intentionally absent. No other question is included.
    return f"{question.instructions}\n{state_text(state)}"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    group_id: str
    source: str
    source_id: str
    split: str
    task: str
    state: str
    question: Question
    target: list[float]
    label: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_target(self):
        import math

        _, options = self.question.options()
        if len(self.target) != len(options):
            raise ValueError("target must match candidate count")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.target):
            raise ValueError("targets must be finite probabilities")
        if abs(sum(self.target) - 1) > 1e-6:
            raise ValueError("target probabilities must sum to one")
        return self
