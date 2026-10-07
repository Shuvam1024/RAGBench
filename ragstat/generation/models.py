"""Provider-independent text and usage records."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

Count = Annotated[StrictInt, Field(ge=0)]


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_tokens: Count
    cached_input_tokens: Count = 0
    output_tokens: Count

    @model_validator(mode="after")
    def cached_subset(self) -> Self:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        return self


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    model: str
    usage: Usage | None = None
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    correctness: Annotated[StrictInt, Field(ge=0, le=4)]
    faithfulness: Annotated[StrictInt, Field(ge=0, le=4)]
    explanation: str = Field(min_length=1, max_length=4000)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rubric_version: str = "ragbench-judge-v1"
    scores: JudgeScore
    response: GeneratedAnswer
