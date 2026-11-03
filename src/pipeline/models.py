from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Error(Model):
    kind: str
    message: str


class Review(Model):
    product_id: str
    text: str
    title: str | None = None
    rating: float | None = None


class Product(Model):
    product_id: str
    source: str
    category: str
    review_count_bin: str
    dataset_role: Literal["main_evaluation", "threshold_calibration"]
    reviews: list[Review]
    source_text: str
    source_hash: str


class GenerationMeta(Model):
    provider: str
    model: str
    revision: str
    prompt_hash: str
    request: dict[str, Any]
    request_hash: str
    cached: bool
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Summary(Model):
    summary_id: str
    product_id: str
    source: str
    category: str
    review_count_bin: str
    dataset_role: Literal["main_evaluation", "threshold_calibration"]
    condition: str
    summarizer_model: str
    is_primary_panel: bool
    is_model_ablation: bool
    source_text: str
    source_hash: str
    source_reviews: list[Review]
    text: str | None = None
    error: Error | None = None
    generation: GenerationMeta

    @model_validator(mode="after")
    def one_result(self) -> "Summary":
        if (self.text is None) == (self.error is None):
            raise ValueError("Exactly one of text or error is required.")

        return self


class Edit(Model):
    before: str
    after: str


class Modification(Model):
    attempt_id: str
    modification_id: str | None = None
    parent_summary_id: str
    product_id: str
    dataset_role: Literal["main_evaluation", "threshold_calibration"]
    summary_condition: str
    condition: str
    family: str
    severity: int
    direction: str
    aspect_group: str
    generator: str
    modifier: str
    # The prompt that produced this attempt, so a resumed run can tell a stored
    # record apart from one the current configuration would produce. Records
    # written before this field existed carry None and are re-run once.
    prompt_hash: str | None = None
    # Retained only so records written before this field became optional still
    # load. The reviews a modification is scored against are read from its
    # parent summary, never from here, and copying them per attempt made the
    # file roughly ten times larger than the records it holds.
    source_text: str | None = None
    parent_text: str
    text: str | None = None
    edits: list[Edit] = Field(default_factory=list)
    details: dict[str, Any] | None = None
    changed_word_percentage: float | None = None
    length_reduction: float | None = None
    similarity: float | None = None
    valid: bool
    error: str | None = None
    summary_is_judge_generated: bool
    modification_is_judge_generated: bool


class MetricResult(Model):
    metric: str
    score: float
    higher_is_faithful: bool
    probabilities: dict[str, float] | None = None
    raw_answer: str | None = None
    model_revision: str | None = None
    # The judge that produced this score, for metrics that ask a language model
    # rather than compute a number. `model_revision` also carries it, but only
    # as the leading part of a string that bundles it with backend versions, so
    # comparing judges meant parsing that string apart. Kept separate here so a
    # second judge can be grouped against the first directly. None for metrics
    # with no judge (SUPERT, SummaC, BARTScore), and on scores written before
    # this field existed - read it with a `model_revision` fallback.
    judge_model: str | None = None
    prompt_hash: str | None = None
    details: dict[str, Any] | None = None


class Score(Model):
    score_id: str
    record_id: str
    parent_summary_id: str
    product_id: str
    dataset_role: Literal["main_evaluation", "threshold_calibration"]
    summary_condition: str
    modification_condition: str
    summarizer_model: str
    generator: str
    modifier: str
    summary_is_judge_generated: bool
    modification_is_judge_generated: bool
    source_hash: str
    summary_hash: str
    seed: int
    result: MetricResult
