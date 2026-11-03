from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Any

if __package__:
    import pandas as pd
    from tqdm.auto import tqdm

    from .common import (
        active_geval_judge,
        append_jsonl,
        digest,
        geval_judge_app_name,
        metric_score_ids,
        read_jsonl,
        score_id_for,
    )
    from .llms import (
        BatchPending,
        BatchSession,
        DeferredRequest,
        ResponseCache,
        build_qwen_score_request,
        client,
        is_retryable_error,
        model_identity,
    )
    from .models import MetricResult, Modification, Score, Summary
    from .reporting import (
        format_score_summary,
        print_scoring_plan,
        report_scoring_status,
        stage_progress_bar,
    )


METRIC_APPLICABILITY = {
    "geval_consistency": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "geval_relevance": {
        "primary": ["omission", "review_filtering"],
        "secondary": [],
    },
    "summac_conv": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "summac_zs": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "qafacteval": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "supert": {
        "primary": ["omission"],
        "secondary": ["review_filtering", "factual_misrepresentation"],
    },
    "bartscore_faithfulness": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": ["omission", "review_filtering"],
    },
    "bartscore_coverage": {
        "primary": ["omission", "review_filtering"],
        "secondary": [],
    },
    "alignscore": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "minicheck": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "alignscore_default": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "minicheck_default": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "minicheck_7b": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "minicheck_7b_default": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "summac_conv_default": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "summac_zs_default": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
    "factgraph": {
        "primary": ["factual_misrepresentation", "quantifier_change"],
        "secondary": [],
    },
}
SUPERT_SENTENCE_TRANSFORMERS_VERSION = "3.4.1"


def metric_applicability(metric: str) -> dict[str, list[str]]:
    """Return the pre-declared experimental role of one metric."""

    try:
        return METRIC_APPLICABILITY[metric]
    except KeyError as exception:
        raise ValueError(
            f"No applicability declaration for metric: {metric}"
        ) from exception


def split_sentences(text: str) -> list[str]:
    """Split formatted reviews without adding another tokenizer dependency."""

    sentences: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped or re.fullmatch(r"Review \d+:", stripped):
            continue

        sentences.extend(
            candidate.strip()
            for candidate in re.split(r"(?<=[.!?])\s+", stripped)
            if candidate.strip()
        )

    return sentences


def split_source_documents(source: str) -> list[list[str]]:
    """Split a formatted source back into one sentence list per review.

    `format_reviews` joins reviews with a blank line, so that separator
    recovers the documents the source was assembled from. Metrics that are
    defined over a *set* of documents need this: handed the whole source as one
    string they cannot tell 48 reviews from one long article.
    """

    blocks = [block for block in re.split(r"\n\s*\n", source) if block.strip()]
    documents = [split_sentences(block) for block in blocks]

    return [sentences for sentences in documents if sentences]


def chunk_sentences(
    sentences: list[str],
    maximum_characters: int = 3_000,
) -> list[str]:
    """Build BART-sized source chunks while preserving sentence boundaries."""

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        if current and current_length + len(sentence) + 1 > maximum_characters:
            chunks.append(" ".join(current))
            current = []
            current_length = 0

        current.append(sentence)
        current_length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def probability_weighted_score(
    logprobabilities: dict[str, float],
) -> tuple[float, dict[str, float]]:
    label_logprobabilities = {
        str(label): float(logprobabilities[str(label)])
        for label in range(1, 6)
        if str(label) in logprobabilities
    }

    if set(label_logprobabilities) != set("12345"):
        raise ValueError(
            "The response did not contain log probabilities for all labels 1-5. "
            f"Observed labels: {sorted(label_logprobabilities)}"
        )

    highest_logprobability = max(label_logprobabilities.values())
    unnormalized_probabilities = {
        label: math.exp(logprobability - highest_logprobability)
        for label, logprobability in label_logprobabilities.items()
    }
    probability_sum = sum(unnormalized_probabilities.values())
    normalized_probabilities = {
        label: probability / probability_sum
        for label, probability in unnormalized_probabilities.items()
    }
    weighted_score = sum(
        int(label) * probability
        for label, probability in normalized_probabilities.items()
    )

    return weighted_score, normalized_probabilities


def extract_label_logprobabilities(raw_response: dict[str, Any]) -> dict[str, float]:
    if "top_logprobs" in raw_response:
        candidates = raw_response["top_logprobs"].items()
    else:
        top_logprobs = raw_response["choices"][0]["logprobs"]["content"][0][
            "top_logprobs"
        ]
        candidates = (
            (candidate["token"], candidate["logprob"]) for candidate in top_logprobs
        )

    return {
        str(token).strip(): float(logprobability)
        for token, logprobability in candidates
        if str(token).strip() in set("12345")
    }


@lru_cache(maxsize=None)
def load_geval_prompt(path: str) -> tuple[str, str]:
    """Read and hash one G-Eval prompt. Cached: a run reads each file once.

    Reading it once also fixes the prompt for the whole run, so a file edited
    while a long scoring pass is in flight cannot score part of the corpus with
    one prompt and the rest with another.
    """

    prompt_path = Path(path)

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing G-Eval prompt: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    return template, digest(template)


class Scorer(ABC):
    """Superclass for G-Eval and the isolated secondary metrics."""

    name: str

    @abstractmethod
    def score(self, source: str, summary: str) -> MetricResult:
        """Score one source/summary pair."""

    @abstractmethod
    def preflight(self) -> None:
        """Verify metric resources before any batch requests are sent."""

    def close(self) -> None:
        """Release optional worker resources."""

    @property
    def batch_size(self) -> int:
        """Return how many source/summary pairs can be scored together."""

        return 1

    def score_many(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[MetricResult]:
        """Score a small batch, using the single-item implementation by default."""

        return [self.score(source=source, summary=summary) for source, summary in pairs]

    @property
    def revision(self) -> str:
        """Return the version included in deterministic score IDs."""

        return self.name

    @property
    def backend(self) -> str:
        """Return where this metric actually executes."""

        return type(self).__name__


class GEvalScorer(Scorer):
    def __init__(
        self,
        name: str,
        specification: dict[str, Any],
        config: dict[str, Any],
        batch_session: BatchSession,
        worker: ModalMetricWorker | None = None,
        mock: bool = False,
    ) -> None:
        self.name = name
        self.specification = specification
        # The judge is chosen by `geval_judge`, independently of the Qwen that
        # generates `qwen_faithful` summaries, so the scoring model can be
        # swapped without changing anything the summaries were produced by.
        self.judge_name, self.model = active_geval_judge(
            config,
            config.get("geval_judge_override"),
        )
        self.decoding = config["decoding"]
        modal_config = config["modal"]
        self.backend_revision = ";".join(
            (
                model_identity(self.model),
                f"torch=={modal_config['qwen_score_torch_version']}",
                (f"transformers=={modal_config['qwen_score_transformers_version']}"),
                (f"flash-attn=={modal_config['qwen_score_flash_attention_version']}"),
                (
                    "fla=="
                    f"{modal_config['qwen_score_flash_linear_attention_version']}"
                ),
                (
                    "causal-conv1d=="
                    f"{modal_config['qwen_score_causal_conv1d_version']}"
                ),
                str(modal_config["qwen_score_backend_version"]),
                # Mirrors QWEN_SCORE_REVISION in modal_app.py: present only for
                # a judge whose answer is started with the form line.
                *(
                    (f"answer-prefix={self.model['answer_prefix']}",)
                    if self.model.get("answer_prefix")
                    else ()
                ),
                # Mirrors JUDGE_GPU: present only for a judge that names a GPU.
                *((f"gpu={self.model['gpu']}",) if self.model.get("gpu") else ()),
            )
        )
        self.worker = worker
        self.mock = mock

        if mock:
            self.client = client(
                provider=self.model["provider"],
                mock=True,
                batch_session=batch_session,
            )
            self.cache = ResponseCache(config["cache"])
        else:
            self.client = None
            self.cache = None

    @property
    def batch_size(self) -> int:
        return int(self.specification.get("modal_batch_size", 1))

    @property
    def judge_model(self) -> str:
        """The model this metric asks, apart from the backend it runs on."""

        return model_identity(self.model)

    @property
    def revision(self) -> str:
        return self.backend_revision

    @property
    def backend(self) -> str:
        return "mock" if self.mock else "modal:QwenGEvalService"

    def preflight(self) -> None:
        if self.mock:
            return

        if "$" in self.model.get("revision", ""):
            raise ValueError("Missing pinned QWEN_REVISION.")

        if self.worker is None:
            raise RuntimeError("G-Eval requires the Modal Qwen scoring worker.")

        self.worker.start()

        if self.worker.ready_response is None:
            raise RuntimeError("Qwen G-Eval worker returned no version information.")

        observed_revision = self.worker.ready_response.get("revision")

        if observed_revision != self.revision:
            raise RuntimeError(
                "Qwen G-Eval revision mismatch: configured "
                f"{self.revision!r}, worker reported {observed_revision!r}."
            )

    def _render_prompt(
        self,
        source: str,
        summary: str,
    ) -> tuple[str, str]:
        template, prompt_hash = load_geval_prompt(self.specification["prompt"])
        prompt = template.replace("{reviews}", source)
        prompt = prompt.replace("{summary}", summary)

        return prompt, prompt_hash

    def score(self, source: str, summary: str) -> MetricResult:
        if not self.mock:
            return self.score_many([(source, summary)])[0]

        prompt, prompt_hash = self._render_prompt(source, summary)
        prompt_path = Path(self.specification["prompt"])

        if self.client is None or self.cache is None:
            raise RuntimeError("Mock G-Eval client was not initialized.")

        request = build_qwen_score_request(
            model=self.model,
            prompt=prompt,
            decoding=self.decoding,
        )
        cache_key = {
            "revision": model_identity(self.model),
            "prompt_file": str(prompt_path),
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "request": request,
        }

        response, _ = self.cache.call(
            key=cache_key,
            llm_client=self.client,
            payload=request,
        )
        logprobabilities = extract_label_logprobabilities(response.raw)
        score, probabilities = probability_weighted_score(logprobabilities)

        return MetricResult(
            metric=self.name,
            score=score,
            higher_is_faithful=self.specification["higher_is_faithful"],
            probabilities=probabilities,
            raw_answer=response.text.strip(),
            model_revision=self.revision,
            judge_model=self.judge_model,
            prompt_hash=prompt_hash,
            details={
                "applicability": metric_applicability(self.name),
                "usage": response.raw.get("usage", {}),
            },
        )

    def score_many(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[MetricResult]:
        if self.mock:
            return [
                self.score(source=source, summary=summary) for source, summary in pairs
            ]

        if self.worker is None:
            raise RuntimeError("G-Eval worker is unavailable.")

        rendered = [self._render_prompt(source, summary) for source, summary in pairs]
        responses = self.worker.request_many(
            [
                {
                    "prompt": prompt,
                }
                for prompt, _ in rendered
            ]
        )

        return [
            MetricResult(
                metric=self.name,
                score=float(response["score"]),
                higher_is_faithful=self.specification["higher_is_faithful"],
                probabilities={
                    str(label): float(probability)
                    for label, probability in response["probabilities"].items()
                },
                raw_answer=str(response["raw_answer"]),
                model_revision=self.revision,
                judge_model=self.judge_model,
                prompt_hash=prompt_hash,
                details={
                    "applicability": metric_applicability(self.name),
                    **response.get("details", {}),
                },
            )
            for response, (_, prompt_hash) in zip(
                responses,
                rendered,
                strict=True,
            )
        ]

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class ModalMetricWorker:
    """Client for one persistent GPU metric class in the deployed Modal app."""

    def __init__(
        self,
        app_name: str,
        class_name: str,
    ) -> None:
        self.app_name = app_name
        self.class_name = class_name
        self.service: Any | None = None
        self.ready_response: dict[str, Any] | None = None

    def start(self) -> None:
        if self.service is not None:
            return

        try:
            import modal
        except ImportError as exception:
            raise RuntimeError(
                "Modal scoring requires the optional dependency. "
                "Run `uv sync --extra modal`."
            ) from exception

        remote_class = modal.Cls.from_name(
            self.app_name,
            self.class_name,
        )
        self.service = remote_class()
        response = self.service.ready.remote()

        if response.get("status") != "ready":
            raise RuntimeError(
                response.get("error", "Modal metric service did not become ready.")
            )

        self.ready_response = response

    def request_many(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.start()

        if self.service is None:
            raise RuntimeError("Modal metric service is unavailable.")

        responses = self.service.score_batch.remote(payloads)

        if len(responses) != len(payloads):
            raise RuntimeError(
                f"{self.class_name} returned {len(responses)} results for "
                f"{len(payloads)} requests."
            )

        for response in responses:
            if "error" in response:
                raise RuntimeError(response["error"])

        return responses

    def close(self) -> None:
        self.service = None
        self.ready_response = None


class ModalMetricScorer(Scorer):
    def __init__(
        self,
        name: str,
        specification: dict[str, Any],
        worker: ModalMetricWorker,
    ) -> None:
        self.name = name
        self.specification = specification
        self.worker = worker

    @property
    def batch_size(self) -> int:
        return int(self.specification.get("modal_batch_size", 1))

    @property
    def revision(self) -> str:
        return self.specification["revision"]

    @property
    def backend(self) -> str:
        return f"modal:{self.worker.class_name}"

    def preflight(self) -> None:
        self.worker.start()

        if self.worker.ready_response is None:
            raise RuntimeError(f"{self.name} worker returned no version information.")

        observed_revision = self.worker.ready_response.get("revision")

        if observed_revision != self.revision:
            raise RuntimeError(
                f"{self.name} revision mismatch: configured {self.revision!r}, "
                f"worker reported {observed_revision!r}."
            )

    def score(self, source: str, summary: str) -> MetricResult:
        return self.score_many([(source, summary)])[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[MetricResult]:
        requests = [
            {
                "source_text": source,
                "summary_text": summary,
                "parameters": self.specification.get("parameters", {}),
            }
            for source, summary in pairs
        ]
        responses = self.worker.request_many(requests)

        return [
            MetricResult(
                metric=self.name,
                score=float(response["score"]),
                higher_is_faithful=self.specification["higher_is_faithful"],
                model_revision=self.specification["revision"],
                details={
                    "applicability": metric_applicability(self.name),
                    **response.get("details", {}),
                },
            )
            for response in responses
        ]

    def close(self) -> None:
        self.worker.close()


def resolve_metrics(
    config: dict[str, Any],
    selected: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Resolve repeatable metric selectors into the metrics this run scores.

    Selecting nothing scores every metric `experiment.yaml` enables. Naming
    metrics restricts the run to those alone; scores already saved for the
    others are untouched and are still republished, because the output file is
    rebuilt from the whole checkpoint rather than from this run's work.
    """

    available = config["metrics"]

    if not selected:
        return dict(available)

    names = list(dict.fromkeys(selected))
    unsupported = sorted(set(names) - set(available))

    if unsupported:
        raise ValueError(
            "Unknown metrics: "
            + ", ".join(unsupported)
            + ". Choose from: "
            + ", ".join(available)
        )

    disabled = [name for name in names if not available[name]["enabled"]]

    if disabled:
        raise ValueError(
            "These metrics are disabled in experiment.yaml: "
            + ", ".join(disabled)
            + ". Enable them there before selecting them."
        )

    return {name: available[name] for name in names}


def configured_scorers(
    config: dict[str, Any],
    batch_session: BatchSession,
    mock: bool = False,
) -> list[Scorer]:
    scorers: list[Scorer] = []
    workers: dict[str, ModalMetricWorker] = {}
    modal_app_name = config["modal"]["app_name"]
    # G-Eval may be served by its own deployment, so that a second judge can
    # run without replacing the first. The other metrics always live in the
    # main app.
    _, judge_model = active_geval_judge(config, config.get("geval_judge_override"))
    geval_app_name = geval_judge_app_name(config, judge_model)

    try:
        for name, specification in config["metrics"].items():
            if not specification["enabled"]:
                continue

            if mock and not name.startswith("geval_"):
                continue

            if name.startswith("geval_"):
                worker = None

                if not mock:
                    worker = workers.setdefault(
                        "QwenGEvalService",
                        ModalMetricWorker(
                            app_name=geval_app_name,
                            class_name="QwenGEvalService",
                        ),
                    )

                scorer: Scorer = GEvalScorer(
                    name=name,
                    specification=specification,
                    config=config,
                    batch_session=batch_session,
                    worker=worker,
                    mock=mock,
                )
            else:
                class_name = specification["modal_class"]
                worker = workers.setdefault(
                    class_name,
                    ModalMetricWorker(
                        app_name=modal_app_name,
                        class_name=class_name,
                    ),
                )
                scorer = ModalMetricScorer(
                    name=name,
                    specification=specification,
                    worker=worker,
                )

            scorers.append(scorer)
    except Exception:
        for worker in workers.values():
            worker.close()

        raise

    return scorers


def scoring_batches(
    source_items: list[dict[str, Any]],
    scorer: Scorer,
) -> list[list[dict[str, Any]]]:
    """Batch one source group with an optional sentence-pair work limit."""

    parameters = getattr(scorer, "specification", {}).get("parameters", {})
    maximum_sentence_pairs = parameters.get("maximum_sentence_pairs_per_batch")
    if maximum_sentence_pairs is None:
        return [
            source_items[start : start + scorer.batch_size]
            for start in range(0, len(source_items), scorer.batch_size)
        ]

    maximum_sentence_pairs = int(maximum_sentence_pairs)
    if maximum_sentence_pairs < 1:
        raise ValueError("maximum_sentence_pairs_per_batch must be positive.")

    source_sentence_count = max(
        1,
        len(split_sentences(source_items[0]["parent"].source_text)),
    )
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_sentence_pairs = 0

    for item in source_items:
        summary_sentence_count = max(1, len(split_sentences(item["text"])))
        item_sentence_pairs = source_sentence_count * summary_sentence_count
        exceeds_work_limit = (
            current
            and current_sentence_pairs + item_sentence_pairs > maximum_sentence_pairs
        )
        if current and (len(current) >= scorer.batch_size or exceeds_work_limit):
            batches.append(current)
            current = []
            current_sentence_pairs = 0

        current.append(item)
        current_sentence_pairs += item_sentence_pairs

    if current:
        batches.append(current)
    return batches


# The parquet is republished from scratch, so its cost grows with the number of
# scores already saved. Publishing after every batch made a long run quadratic
# in its own output; the checkpoint JSONL is appended per score regardless, so
# nothing is lost between publications.
SCORE_PUBLISH_INTERVAL_SECONDS = 60.0


def write_score_output(records: list[Score], output_path: Path) -> None:
    """Atomically publish all durable score records as a readable parquet."""

    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_records = [record.model_dump(mode="json") for record in records]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.json_normalize(serialized_records).to_parquet(
        temporary_path,
        index=False,
    )
    temporary_path.replace(output_path)


def check_prompt_drift(
    scorers: list[Scorer],
    records: list[Score],
    state_path: Path,
) -> None:
    """Refuse to extend a checkpoint whose G-Eval prompt has since changed.

    A score is identified by its record, its metric, and the metric revision -
    model and backend versions, but not the prompt, because the same prompt is
    what makes two G-Eval scores comparable in the first place. An edited
    prompt therefore reuses the identifiers of the scores it invalidates: the
    saved ones are skipped as complete and the rest are produced under the new
    prompt, leaving one column of results that came from two rubrics. Stopping
    here is the only point at which that is still visible.
    """

    # Only scores under the revision this run would extend can collide with
    # it. Scores left behind by an earlier model carry different IDs and are
    # already filtered out of the analysis, so an old prompt among them is
    # history rather than a conflict.
    saved_prompt_hashes: dict[tuple[str, str | None], set[str]] = {}

    for record in records:
        if record.result.prompt_hash is None:
            continue

        key = (record.result.metric, record.result.model_revision)
        saved_prompt_hashes.setdefault(key, set()).add(record.result.prompt_hash)

    for scorer in scorers:
        if not isinstance(scorer, GEvalScorer):
            continue

        _, prompt_hash = load_geval_prompt(scorer.specification["prompt"])
        superseded = saved_prompt_hashes.get(
            (scorer.name, scorer.revision),
            set(),
        ) - {prompt_hash}

        if superseded:
            raise ValueError(
                f"{state_path} holds {scorer.name} scores produced by a "
                f"different {scorer.specification['prompt']} "
                f"(saved {sorted(superseded)}, current {prompt_hash!r}). "
                "Scores from two prompts are not comparable and would be mixed "
                "silently, because the prompt is not part of a score's "
                "identity. Restore the prompt, or clear the saved scores "
                "(`uv run reset --stage scores --yes`) before re-scoring."
            )


def preflight_scorer(scorer: Scorer) -> None:
    """Start and verify a metric backend only when the metric has pending work."""

    backend = scorer.backend
    tqdm.write(
        f"[{scorer.name}] Preparing {backend} (batch size {scorer.batch_size})..."
    )
    try:
        scorer.preflight()
    except Exception as exception:
        tqdm.write(
            f"[{scorer.name}] Preflight failed: {type(exception).__name__}: {exception}"
        )
        raise
    tqdm.write(f"[{scorer.name}] Backend ready.")


def is_retryable_scoring_error(error: str) -> bool:
    """Classify transient scoring failures for status reporting."""

    return is_retryable_error(error)


def run(
    config: dict[str, Any],
    summaries_path: str | None = None,
    modifications_path: str | None = None,
    output: str | None = None,
    dry_run: bool = False,
    state_name: str = "scores.jsonl",
    only_modified_parents: bool = False,
    selected_metrics: list[str] | tuple[str, ...] | None = None,
) -> list[Score]:
    """Score faithful summaries and every valid modification of them.

    `state_name` selects the resume checkpoint. A side run scoring a different
    modification set must not share the main checkpoint, or its scores would be
    published into the main parquet, which is rebuilt from that state.

    `only_modified_parents` narrows the faithful baseline to the summaries the
    given modifications actually descend from, so a side run stays
    self-contained without re-scoring the whole panel.
    """

    metrics = resolve_metrics(config, selected_metrics)
    summary_input = summaries_path or config["outputs"]["summaries"]
    modification_input = modifications_path or config["outputs"]["modifications"]

    summaries = [
        summary
        for summary in read_jsonl(summary_input, Summary)
        if summary.text is not None
    ]
    parents = {summary.summary_id: summary for summary in summaries}
    modifications = [
        modification
        for modification in read_jsonl(modification_input, Modification)
        if modification.valid
    ]

    if any(
        modification.parent_summary_id not in parents for modification in modifications
    ):
        raise ValueError("A modification cannot be traced to one parent summary.")

    if only_modified_parents:
        referenced = {
            modification.parent_summary_id for modification in modifications
        }
        summaries = [
            summary for summary in summaries if summary.summary_id in referenced
        ]

    state_path = Path(config["state"]) / state_name
    records = read_jsonl(state_path, Score)
    completed_ids = {record.score_id for record in records}
    output_path = Path(output or config["outputs"]["scores"])
    write_score_output(records, output_path)
    batch_session = BatchSession(config=config)
    scorers = configured_scorers(
        config={**config, "metrics": metrics},
        batch_session=batch_session,
        mock=dry_run,
    )
    items = [
        {
            "record_id": summary.summary_id,
            "parent_id": summary.summary_id,
            "parent": summary,
            "text": summary.text or "",
            "condition": "faithful",
            "generator": "none",
            "modifier": "none",
            "judge_modified": False,
        }
        for summary in summaries
    ]
    items.extend(
        {
            "record_id": modification.modification_id,
            "parent_id": modification.parent_summary_id,
            "parent": parents[modification.parent_summary_id],
            "text": modification.text or "",
            "condition": modification.condition,
            "generator": modification.generator,
            "modifier": modification.modifier,
            "judge_modified": modification.modification_is_judge_generated,
        }
        for modification in modifications
    )
    ordered_items = sorted(
        items,
        key=lambda item: (
            item["parent"].source_hash,
            item["record_id"],
        ),
    )
    starting_record_count = len(records)
    current_metric: str | None = None
    last_published_at = time.monotonic()

    check_prompt_drift(
        scorers=scorers,
        records=records,
        state_path=state_path,
    )
    print_scoring_plan(
        scorers=scorers,
        items=items,
        completed_ids=completed_ids,
        state_path=state_path,
        output_path=output_path,
        publish_interval_seconds=SCORE_PUBLISH_INTERVAL_SECONDS,
    )
    report_scoring_status(
        scorers=scorers,
        items=items,
        records=records,
        moment="before_scoring",
        starting_record_count=starting_record_count,
        json_output=False,
    )

    try:
        batch_session.refresh()

        for scorer in scorers:
            current_metric = scorer.name
            batches_this_run = 0
            scores_this_run = 0
            expected_ids = metric_score_ids(items, scorer)
            metric_completed = len(expected_ids.intersection(completed_ids))

            if metric_completed == len(expected_ids):
                tqdm.write(
                    f"[{scorer.name}] Complete: {metric_completed:,}/"
                    f"{len(expected_ids):,}; skipping remote preflight."
                )
                continue

            preflight_scorer(scorer)
            progress = stage_progress_bar(
                total=len(expected_ids),
                initial=metric_completed,
                description=scorer.name,
                unit="score",
            )
            score_count = 0
            score_total = 0.0
            lowest_score = float("inf")
            highest_score = float("-inf")
            score_summary: dict[str, str] = {}
            # Modifications completed so far this run, per modification
            # condition (e.g. "quantifier_change_advantageous", "faithful"),
            # so the bar can show progress within the type being scored.
            type_counts: dict[str, int] = {}
            source_groups = groupby(
                ordered_items,
                key=lambda item: item["parent"].source_hash,
            )

            try:
                for _, source_group in source_groups:
                    source_items = list(source_group)

                    for candidate_items in scoring_batches(source_items, scorer):
                        pending_items = []

                        for item in candidate_items:
                            score_id = score_id_for(item, scorer)

                            if score_id not in completed_ids:
                                pending_items.append((score_id, item))

                        if not pending_items:
                            continue

                        pairs = [
                            (
                                item["parent"].source_text,
                                item["text"],
                            )
                            for _, item in pending_items
                        ]

                        source_sentences = len(
                            split_sentences(pending_items[0][1]["parent"].source_text)
                        )
                        summary_sentences = sum(
                            max(1, len(split_sentences(item["text"])))
                            for _, item in pending_items
                        )
                        estimated_sentence_pairs = source_sentences * summary_sentences
                        product_id = str(pending_items[0][1]["parent"].product_id)
                        batch_conditions = sorted(
                            {item["condition"] for _, item in pending_items}
                        )
                        modification_type = (
                            batch_conditions[0]
                            if len(batch_conditions) == 1
                            else f"{len(batch_conditions)} types"
                        )
                        # The same keys before and after a batch, so the bar
                        # reads as one line that changes rather than two
                        # layouts alternating.
                        batch_postfix = {
                            "product": product_id[:24],
                            "type": modification_type,
                            "items": len(pending_items),
                            "pairs": estimated_sentence_pairs,
                        }
                        progress.set_postfix(
                            status="scoring",
                            **batch_postfix,
                            modifications=sum(
                                type_counts.get(condition, 0)
                                for condition in batch_conditions
                            ),
                            **score_summary,
                            saved=scores_this_run,
                            refresh=True,
                        )
                        batch_started = time.monotonic()

                        try:
                            results = scorer.score_many(pairs)
                        except DeferredRequest:
                            continue

                        if len(results) != len(pending_items):
                            raise RuntimeError(
                                f"{scorer.name} returned {len(results)} scores for "
                                f"{len(pending_items)} inputs."
                            )

                        for (score_id, item), result in zip(
                            pending_items,
                            results,
                            strict=True,
                        ):
                            parent = item["parent"]
                            record = Score(
                                score_id=score_id,
                                record_id=item["record_id"],
                                parent_summary_id=item["parent_id"],
                                product_id=parent.product_id,
                                dataset_role=parent.dataset_role,
                                summary_condition=parent.condition,
                                modification_condition=item["condition"],
                                summarizer_model=parent.summarizer_model,
                                generator=item["generator"],
                                modifier=item["modifier"],
                                summary_is_judge_generated=(
                                    parent.condition == "qwen_faithful"
                                ),
                                modification_is_judge_generated=item["judge_modified"],
                                source_hash=parent.source_hash,
                                summary_hash=digest(item["text"]),
                                seed=config["seed"],
                                result=result,
                            )

                            append_jsonl(state_path, record)
                            records.append(record)
                            completed_ids.add(score_id)

                        for result in results:
                            score_count += 1
                            score_total += result.score
                            lowest_score = min(lowest_score, result.score)
                            highest_score = max(highest_score, result.score)

                        for _, item in pending_items:
                            type_counts[item["condition"]] = (
                                type_counts.get(item["condition"], 0) + 1
                            )

                        score_summary = format_score_summary(
                            count=score_count,
                            total=score_total,
                            last=results[-1].score,
                            lowest=lowest_score,
                            highest=highest_score,
                        )

                        batches_this_run += 1
                        scores_this_run += len(pending_items)

                        if (
                            time.monotonic() - last_published_at
                            >= SCORE_PUBLISH_INTERVAL_SECONDS
                        ):
                            write_score_output(records, output_path)
                            last_published_at = time.monotonic()

                        progress.update(len(pending_items))
                        progress.set_postfix(
                            status=f"saved in {time.monotonic() - batch_started:.1f}s",
                            **batch_postfix,
                            modifications=sum(
                                type_counts.get(condition, 0)
                                for condition in batch_conditions
                            ),
                            **score_summary,
                            saved=scores_this_run,
                            refresh=False,
                        )
            finally:
                progress.close()
                write_score_output(records, output_path)
                last_published_at = time.monotonic()

            metric_completed = len(expected_ids.intersection(completed_ids))
            distribution = (
                f" mean {score_total / score_count:.3f}, range "
                f"{lowest_score:.3f}-{highest_score:.3f};"
                if score_count
                else ""
            )
            tqdm.write(
                f"[{scorer.name}] Saved {scores_this_run:,} new scores;"
                f"{distribution} "
                f"{metric_completed:,}/{len(expected_ids):,} complete."
            )

        report_scoring_status(
            scorers=scorers,
            items=items,
            records=records,
            moment="after_scoring_pass",
            starting_record_count=starting_record_count,
            current_metric=current_metric,
            json_output=False,
        )

        batch_session.finish()
        write_score_output(records, output_path)
        report_scoring_status(
            scorers=scorers,
            items=items,
            records=records,
            moment="complete",
            starting_record_count=starting_record_count,
            json_output=False,
        )
    except BatchPending as pending:
        write_score_output(records, output_path)
        report_scoring_status(
            scorers=scorers,
            items=items,
            records=records,
            moment="pending",
            starting_record_count=starting_record_count,
            current_metric=current_metric,
            error=pending,
            json_output=False,
        )
        raise
    except Exception as exception:
        write_score_output(records, output_path)
        report_scoring_status(
            scorers=scorers,
            items=items,
            records=records,
            moment="failed",
            starting_record_count=starting_record_count,
            current_metric=current_metric,
            error=exception,
            json_output=False,
        )
        raise
    finally:
        for scorer in scorers:
            scorer.close()

    return records


def worker_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class ExternalMetricBackend(ABC):
    """Superclass for the isolated metric implementations."""

    revision: str

    @abstractmethod
    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one JSON-serializable score."""

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Score a GPU batch, with a correct sequential fallback."""

        return [
            self.score(
                source=source,
                summary=summary,
                parameters=parameters,
            )
            for source, summary in pairs
        ]


def summac_premise_settings(settings: dict[str, Any]) -> tuple[str, int]:
    premises = str(settings["premises"])
    premise_tokens = int(settings["premise_tokens"])

    if premises not in {"reviews", "library"}:
        raise ValueError(f"SummaC premises must be 'reviews' or 'library', not {premises!r}.")

    if premises == "reviews" and not str(settings["granularity"]).startswith("paragraph"):
        raise ValueError(
            "premises: reviews hands SummaC one paragraph per review, so it needs "
            f"a paragraph premise granularity, not {settings['granularity']!r}."
        )

    # SummaC truncates a pair at 500 tokens; the rest is left for the sentence.
    if not 0 < premise_tokens < 500:
        raise ValueError(f"premise_tokens must leave room for a summary sentence: {premise_tokens}.")

    return premises, premise_tokens


def summac_premise_revision(premises: str, premise_tokens: int) -> str:
    return (
        f"premises=reviews;premise_tokens={premise_tokens}"
        if premises == "reviews"
        else "premises=library"
    )


def summac_source(backend: Any, source: str) -> tuple[str, int]:
    """The source as SummaC should see it, and how many reviews were split.

    Under `premises: reviews` each review - or, when one alone exceeds the
    premise budget, each piece of it - becomes one blank-line paragraph, which
    is exactly what SummaC's paragraph splitter separates on.
    """

    if backend.premises != "reviews":
        return source, 0

    tokenizer = backend.imager.tokenizer
    premises, split_reviews = review_premises(
        source,
        lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]),
        backend.premise_tokens,
    )

    return "\n\n".join(premises), split_reviews


class SummaCBackend(ExternalMetricBackend):
    """SummaC-Conv, configured from `metrics.summac_conv.options`.

    Two of SummaC's own defaults are wrong for a 100-review source and are
    settable from `experiment.yaml` rather than patched here:

    `max_doc_sents` caps how many premise units the imager keeps
    (`original_chunks[:max_doc_sents]`). SummaC defaults it to 100, sized for a
    news article; 78% of these sources are longer than that at sentence
    granularity, and the median product had under half its sentences reach the
    model. `null` keeps all of them.

    `half_precision` works around SummaC gating `.half()` on the device string
    being exactly "cuda". `worker_device` returns "cuda:0", so the NLI model
    silently ran in fp32 on a card whose fp16 throughput is roughly four times
    its fp32.

    `granularity` selects the premise/hypothesis units. "sentence" is the
    paper's configuration for SummaC-Conv and what the bundled convolution
    weights were trained on. "paragraph-sentence" makes each premise one whole
    review, which is the natural evidence unit here and cuts the grid by about
    half, but shifts the histogram counts the trained head expects - treat it
    as a reported variant, not a free speedup.
    """

    BASE_REVISION = "summac==0.0.4;vitc@3082ba54;conv@525cb081"
    DEFAULT_OPTIONS: dict[str, Any] = {
        "granularity": "sentence",
        "max_doc_sents": None,
        "half_precision": True,
        # `reviews`: the source is handed over as one paragraph per review (see
        # `summac_source`), so SummaC's paragraph splitter yields exactly one
        # premise per review. `library`: the source as formatted, split by
        # SummaC itself - the authors' setting, kept as an ablation.
        "premises": "library",
        "premise_tokens": 400,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        from summac.model_summac import SummaCConv

        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown SummaC options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        device_name = worker_device(device)
        # SummaC calls .half() only when this string is exactly "cuda".
        half = bool(settings["half_precision"]) and device_name.startswith("cuda")
        nli_device = "cuda" if half else device_name
        maximum_premises = settings["max_doc_sents"]
        self.premises, self.premise_tokens = summac_premise_settings(settings)

        # The configuration is part of what a score means, so it travels in the
        # revision and `preflight` refuses a worker whose options have drifted
        # from the ones experiment.yaml records.
        self.revision = ";".join(
            (
                self.BASE_REVISION,
                f"granularity={settings['granularity']}",
                f"max_doc_sents={'all' if maximum_premises is None else maximum_premises}",
                f"precision={'fp16' if half else 'fp32'}",
                summac_premise_revision(self.premises, self.premise_tokens),
            )
        )

        model_path = os.getenv("SUMMAC_MODEL_PATH")
        self.model = SummaCConv(
            models=["vitc"],
            bins="percentile",
            granularity=str(settings["granularity"]),
            nli_labels="e",
            device=nli_device,
            start_file="default",
            agg="mean",
            # Reaches SummaCImager through SummaCConv's **kwargs.
            max_doc_sents=maximum_premises,
        )

        if model_path:
            self.model.imagers[0].model_card = model_path

        self.imager = self.model.imagers[0]
        # Loaded now rather than on first use: the review split below and the
        # statistics need its tokenizer before anything is scored.
        self.imager.load_nli()

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        prepared = [summac_source(self, source) for source, _ in pairs]
        summaries = [summary for _, summary in pairs]
        result = self.model.score([source for source, _ in prepared], summaries)

        return [
            {
                "score": float(score),
                "details": {
                    "family": "factual_consistency",
                    "variant": "SummaC-Conv",
                    "split_reviews": split_reviews,
                    **summac_input_statistics(self.imager, source, summary),
                },
            }
            for score, (source, split_reviews), summary in zip(
                result["scores"], prepared, summaries
            )
        ]


class SummaCZeroShotBackend(ExternalMetricBackend):
    """SummaC-ZS, the variant that can be read at sentence granularity.

    Both variants build the same NLI grid over (premise, hypothesis) pairs;
    they differ in what they do with it, and only one of the two survives a
    source this long.

    SummaC-Conv turns each summary sentence's column into a histogram and feeds
    that to a convolution head trained alongside the paper. `compute_histogram`
    bins with `density=False`, so those features are raw *counts* and scale with
    the number of premises. The head was fitted on news at up to 100 premises;
    at sentence granularity these sources run a median of 212 and up to 1364,
    which is why `summac_conv` is configured at paragraph granularity instead -
    one premise per review keeps the count inside the range the head saw.

    SummaC-ZS has no trained head. `image2score` takes the max over premises and
    the mean over summary sentences, so it is invariant to how many premises
    there are and sentence granularity costs it nothing. That makes it the
    variant that can run at the paper's own granularity on this data, and the
    two together separate the granularity question from the aggregation one.
    """

    BASE_REVISION = "summac==0.0.4;vitc@3082ba54;zs"
    DEFAULT_OPTIONS: dict[str, Any] = {
        "granularity": "sentence",
        "max_doc_sents": None,
        "half_precision": True,
        # The paper's SummaC-ZS: max over premises, mean over summary
        # sentences, entailment minus contradiction.
        "premise_aggregation": "max",
        "summary_aggregation": "mean",
        "use_entailment": True,
        "use_contradiction": True,
        # As for SummaC-Conv; see SummaCBackend.
        "premises": "library",
        "premise_tokens": 400,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        from summac.model_summac import SummaCZS

        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown SummaC-ZS options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        device_name = worker_device(device)
        half = bool(settings["half_precision"]) and device_name.startswith("cuda")
        nli_device = "cuda" if half else device_name
        maximum_premises = settings["max_doc_sents"]
        premise_aggregation = str(settings["premise_aggregation"])
        summary_aggregation = str(settings["summary_aggregation"])
        use_entailment = bool(settings["use_entailment"])
        use_contradiction = bool(settings["use_contradiction"])
        self.premises, self.premise_tokens = summac_premise_settings(settings)

        if not (use_entailment or use_contradiction):
            raise ValueError(
                "SummaC-ZS needs at least one of use_entailment and "
                "use_contradiction; with neither there is no score to compute."
            )

        self.revision = ";".join(
            (
                self.BASE_REVISION,
                f"granularity={settings['granularity']}",
                f"max_doc_sents={'all' if maximum_premises is None else maximum_premises}",
                f"precision={'fp16' if half else 'fp32'}",
                f"op1={premise_aggregation}",
                f"op2={summary_aggregation}",
                f"labels={'ec' if use_entailment and use_contradiction else ('e' if use_entailment else 'c')}",
                summac_premise_revision(self.premises, self.premise_tokens),
            )
        )

        model_path = os.getenv("SUMMAC_MODEL_PATH")
        self.model = SummaCZS(
            model_name="vitc",
            granularity=str(settings["granularity"]),
            op1=premise_aggregation,
            op2=summary_aggregation,
            use_ent=use_entailment,
            use_con=use_contradiction,
            device=nli_device,
            # Reaches SummaCImager through SummaCZS's **kwargs, as it does for
            # SummaCConv.
            max_doc_sents=maximum_premises,
        )

        if model_path:
            self.model.imager.model_card = model_path

        self.imager = self.model.imager
        self.imager.load_nli()

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        prepared = [summac_source(self, source) for source, _ in pairs]
        summaries = [summary for _, summary in pairs]
        result = self.model.score([source for source, _ in prepared], summaries)

        return [
            {
                "score": float(score),
                "details": {
                    "family": "factual_consistency",
                    "variant": "SummaC-ZS",
                    "split_reviews": split_reviews,
                    **summac_input_statistics(self.imager, source, summary),
                },
            }
            for score, (source, split_reviews), summary in zip(
                result["scores"], prepared, summaries
            )
        ]


class BARTScoreBackend(ExternalMetricBackend):
    revision = "facebook/bart-large-cnn@37f520fa;neulab-algorithm@248f511c"

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        reject_unsupported_options("bartscore", options)
        import torch
        from torch import nn
        from transformers import BartForConditionalGeneration, BartTokenizer

        self.torch = torch
        self.device = worker_device(device)
        model_revision = "37f520fa929c961707657b28798b30c003dd100b"
        model_path = os.getenv(
            "BARTSCORE_MODEL_PATH",
            "facebook/bart-large-cnn",
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            model_path,
            revision=None
            if model_path != "facebook/bart-large-cnn"
            else model_revision,
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            model_path,
            revision=None
            if model_path != "facebook/bart-large-cnn"
            else model_revision,
        )
        self.model.eval()
        self.model.to(self.device)
        self.loss = nn.NLLLoss(
            reduction="none",
            ignore_index=self.model.config.pad_token_id,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def generation_scores(
        self,
        sources: list[str],
        targets: list[str],
        batch_size: int,
    ) -> list[float]:
        scores: list[float] = []

        for start in range(0, len(sources), batch_size):
            source_batch = sources[start : start + batch_size]
            target_batch = targets[start : start + batch_size]

            with self.torch.no_grad():
                encoded_source = self.tokenizer(
                    source_batch,
                    max_length=1024,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded_target = self.tokenizer(
                    target_batch,
                    max_length=1024,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                source_tokens = encoded_source["input_ids"].to(self.device)
                source_mask = encoded_source["attention_mask"].to(self.device)
                target_tokens = encoded_target["input_ids"].to(self.device)
                target_lengths = encoded_target["attention_mask"].sum(dim=1)
                target_lengths = target_lengths.to(self.device)
                output = self.model(
                    input_ids=source_tokens,
                    attention_mask=source_mask,
                    labels=target_tokens,
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)
                losses = self.loss(
                    self.log_softmax(logits),
                    target_tokens.view(-1),
                )
                losses = losses.view(target_tokens.shape[0], -1)
                losses = losses.sum(dim=1) / target_lengths
                scores.extend(-loss.item() for loss in losses)

        return scores

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        direction = parameters["direction"]
        batch_size = int(parameters.get("batch_size", 8))
        generation_sources: list[str] = []
        generation_targets: list[str] = []
        aggregations: list[dict[str, Any]] = []

        for source, summary in pairs:
            source_sentences = split_sentences(source)
            summary_sentences = split_sentences(summary)

            if not source_sentences or not summary_sentences:
                raise ValueError(
                    "BARTScore requires non-empty source and summary text."
                )

            start = len(generation_sources)

            if direction == "faithfulness":
                maximum_characters = int(
                    parameters.get(
                        "maximum_source_chunk_characters",
                        3_000,
                    )
                )
                source_chunks = chunk_sentences(
                    source_sentences,
                    maximum_characters=maximum_characters,
                )
                generation_sources.extend(
                    chunk for sentence in summary_sentences for chunk in source_chunks
                )
                generation_targets.extend(
                    sentence for sentence in summary_sentences for _ in source_chunks
                )
                aggregations.append(
                    {
                        "start": start,
                        "count": len(source_chunks) * len(summary_sentences),
                        "chunk_count": len(source_chunks),
                        "summary_sentence_count": len(summary_sentences),
                    }
                )
            elif direction == "coverage":
                generation_sources.extend([summary] * len(source_sentences))
                generation_targets.extend(source_sentences)
                aggregations.append(
                    {
                        "start": start,
                        "count": len(source_sentences),
                        "source_sentence_count": len(source_sentences),
                    }
                )
            else:
                raise ValueError(f"Unknown BARTScore direction: {direction}")

        raw_scores = self.generation_scores(
            generation_sources,
            generation_targets,
            batch_size,
        )
        results: list[dict[str, Any]] = []

        for aggregation in aggregations:
            start = aggregation["start"]
            end = start + aggregation["count"]
            item_scores = raw_scores[start:end]

            if direction == "faithfulness":
                chunk_count = aggregation["chunk_count"]
                sentence_scores = [
                    max(item_scores[index : index + chunk_count])
                    for index in range(0, len(item_scores), chunk_count)
                ]
                score = sum(sentence_scores) / len(sentence_scores)
                details = {
                    "family": "generation_probability",
                    "direction": "source_chunk_to_summary_sentence",
                    "aggregation": "mean(max(source_chunks))",
                    "source_chunks": chunk_count,
                    "summary_sentences": aggregation["summary_sentence_count"],
                }
            else:
                score = sum(item_scores) / len(item_scores)
                details = {
                    "family": "generation_probability",
                    "direction": "summary_to_source_sentence",
                    "aggregation": "mean(source_sentences)",
                    "source_sentences": aggregation["source_sentence_count"],
                }

            results.append(
                {
                    "score": float(score),
                    "details": details,
                }
            )

        return results


class SupertSentenceEncoder:
    """Expose SUPERT's 2020 token-vector API over a pinned modern checkpoint."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def encode(
        self,
        sentences: list[str],
        token_vecs: bool = False,
    ) -> Any:
        if not token_vecs:
            return self.model.encode(
                sentences,
                convert_to_numpy=True,
            )

        encoded_tokens = self.model.tokenize(sentences)
        input_ids = encoded_tokens["input_ids"].tolist()
        attention_masks = encoded_tokens["attention_mask"].tolist()
        token_embeddings = self.model.encode(
            sentences,
            output_value="token_embeddings",
            convert_to_numpy=False,
        )
        vectors = []
        tokens = []

        for ids, mask, embeddings in zip(
            input_ids,
            attention_masks,
            token_embeddings,
        ):
            token_count = int(sum(mask))
            vectors.append(embeddings[:token_count].detach().cpu().numpy())
            tokens.append(self.model.tokenizer.convert_ids_to_tokens(ids[:token_count]))

        return vectors, tokens


class SupertBackend(ExternalMetricBackend):
    BASE_REVISION = (
        "acl20-ref-free-eval@f283ce2c;"
        "sentence-transformers==3.4.1;"
        "model@4088dbef"
    )
    revision = f"{BASE_REVISION};docs=per_review;top15;f1"

    # SUPERT is a *multi-document* metric, and how the source is divided into
    # documents decides what it measures. `parse_docs` records each sentence's
    # index within its own document, `get_top_weights` keeps those below the
    # top-N cut, and `build_pseudo_ref` then groups the survivors into one
    # pseudo-reference per document, which `get_sbert_score` averages over.
    #
    # Handing it the whole source as a single document collapses all of that:
    # `top15` becomes the first fifteen sentences of the concatenation - a
    # median of 7.1% of the source and as little as 1.1%, or roughly the first
    # three of a median 48 reviews - and the average over documents degenerates
    # to one pseudo-reference. The position heuristic only earns its keep on
    # newswire, where a document's opening sentences are its summary; reviews
    # have no such lead bias, so first-fifteen-of-the-concatenation selects on
    # nothing but presentation order.
    #
    # One review is one document, which is what the source actually is. Reviews
    # run a median 4.4 sentences, so `top15` now keeps essentially all of them
    # and every review contributes its own pseudo-reference to the mean.
    DEFAULT_OPTIONS = {
        "document_segmentation": "per_review",
        "pseudo_reference": "top15",
        "similarity": "f1",
    }
    SEGMENTATIONS = ("per_review", "single_document")

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown SUPERT options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        self.segmentation = str(settings["document_segmentation"])

        if self.segmentation not in self.SEGMENTATIONS:
            raise ValueError(
                f"Unknown SUPERT document_segmentation: {self.segmentation!r}. "
                f"Known: {sorted(self.SEGMENTATIONS)}."
            )

        self.pseudo_reference = str(settings["pseudo_reference"])
        self.similarity = str(settings["similarity"])

        # Unlike QAFactEval's batch sizes these change what a score means, so
        # they travel in the revision and preflight refuses a worker whose
        # configuration has drifted from the one experiment.yaml records.
        self.revision = ";".join(
            (
                self.BASE_REVISION,
                f"docs={self.segmentation}",
                self.pseudo_reference,
                self.similarity,
            )
        )

        repository = os.getenv("SUPERT_HOME")

        if not repository or not Path(repository).is_dir():
            raise ValueError(
                "SUPERT_HOME must point to the pinned official "
                "acl20-ref-free-eval checkout."
            )

        repository = str(Path(repository).resolve())
        while repository in sys.path:
            sys.path.remove(repository)
        sentence_transformers = importlib.import_module("sentence_transformers")
        observed_version = getattr(sentence_transformers, "__version__", None)
        if observed_version != SUPERT_SENTENCE_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "SUPERT requires the pinned modern sentence-transformers runtime "
                f"{SUPERT_SENTENCE_TRANSFORMERS_VERSION}; observed "
                f"{observed_version!r}."
            )

        # The official checkout contains a legacy package with the same module
        # name. Preloading the modern package before adding the checkout keeps
        # SUPERT's algorithm but prevents that legacy encoder from shadowing it.
        sys.path.insert(0, repository)
        supert_module = importlib.import_module("ref_free_metrics.supert")
        similarity_module = importlib.import_module(
            "ref_free_metrics.similarity_scorer"
        )
        if (
            supert_module.SentenceTransformer
            is not sentence_transformers.SentenceTransformer
        ):
            raise RuntimeError(
                "SUPERT imported its bundled legacy sentence_transformers package "
                "instead of the pinned modern runtime."
            )
        self.supert_class = supert_module.Supert
        self.parse_documents = similarity_module.parse_documents
        sentence_model = sentence_transformers.SentenceTransformer(
            os.getenv(
                "SUPERT_MODEL_PATH",
                "sentence-transformers/bert-large-nli-stsb-mean-tokens",
            ),
            device=worker_device(device),
        )
        self.bert_model = SupertSentenceEncoder(sentence_model)
        self.cache: dict[str, Any] = {}

    def source_metric(self, source: str) -> Any:
        source_key = source

        if source_key in self.cache:
            return self.cache[source_key]

        metric = self.supert_class.__new__(self.supert_class)
        metric.bert_model = self.bert_model
        metric.sim_metric = self.similarity

        if self.segmentation == "per_review":
            documents = [
                (f"review-{index}", sentences)
                for index, sentences in enumerate(split_source_documents(source))
            ]
        else:
            documents = []

        # A source that carries no blank-line separator - or the explicit
        # single-document setting - still has to score rather than fail, so it
        # falls back to the whole source as one document.
        if not documents:
            documents = [("reviews", split_sentences(source))]

        (
            metric.sent_info_dic,
            _,
            metric.sents_weights,
        ) = self.parse_documents(documents, None, self.pseudo_reference)
        (
            metric.all_token_vecs,
            metric.all_tokens,
        ) = metric.get_all_token_vecs(
            metric.bert_model,
            metric.sent_info_dic,
        )
        metric.ref_vecs, metric.ref_tokens = metric.build_pseudo_ref(
            self.pseudo_reference
        )
        # What the pseudo-reference leaves out, and what the encoder truncates:
        # SUPERT keeps a sentence when its weight exceeds 0.1 (as its own debug
        # output does), and SBERT cuts each sentence at `max_seq_length`.
        texts = [info["text"] for info in metric.sent_info_dic.values()]
        sentence_model = self.bert_model.model
        tokenizer = sentence_model.tokenizer
        metric.input_statistics = {
            "documents": len(documents),
            "source_sentences": len(texts),
            "pseudo_reference_sentences": sum(
                weight > 0.1 for weight in metric.sents_weights
            ),
            "truncated_sentences": sum(
                len(tokenizer(text)["input_ids"]) > sentence_model.max_seq_length
                for text in texts
            ),
        }
        self.cache[source_key] = metric

        return metric

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        results: list[dict[str, Any] | None] = [None] * len(pairs)
        indexed_pairs = list(enumerate(pairs))
        source_groups = groupby(
            sorted(indexed_pairs, key=lambda item: item[1][0]),
            key=lambda item: item[1][0],
        )

        for source, source_group in source_groups:
            grouped_pairs = list(source_group)
            summaries = [pair[1][1] for pair in grouped_pairs]
            metric = self.source_metric(source)
            scores = metric(summaries)

            for (index, _), score in zip(
                grouped_pairs,
                scores,
            ):
                results[index] = {
                    "score": float(score),
                    "details": {
                        "family": "pseudo_reference_coverage",
                        "document_segmentation": self.segmentation,
                        "pseudo_reference": self.pseudo_reference,
                        "pseudo_reference_count": len(metric.ref_vecs),
                        "similarity": f"soft_token_alignment_{self.similarity}",
                        **metric.input_statistics,
                    },
                }

        return [result for result in results if result is not None]


def redirect_pinned_hub_models(mapping: dict[str, str]) -> None:
    """Serve the named Hugging Face repositories from disk, not the network.

    transformers 3.0.2 resolves a model id against two different hosts:
    `PretrainedConfig` and the tokenizers ask the legacy
    `s3.amazonaws.com/models.huggingface.co` bucket, which still answers, while
    `PreTrainedModel` asks `cdn.huggingface.co`, which was decommissioned long
    after this stack was pinned and no longer resolves at all. A checkpoint the
    QAFactEval archives name therefore half-loads - config and vocabulary
    arrive, `pytorch_model.bin` does not - and the container dies in
    `@modal.enter()` with "Can't load weights for 'facebook/bart-large'".

    Rewriting the id to a directory `prepare_model_weights` filled avoids both
    endpoints and keeps the checkpoint pinned by revision like every other one
    in the volume. `from_pretrained` tests `os.path.isdir` before it builds any
    URL, so a local path simply takes the branch a cached download would.

    The three base classes are patched rather than the `Auto*` façades: those
    delegate to the concrete config, model and tokenizer classes, which inherit
    `from_pretrained` from exactly these three.
    """

    import transformers

    for target in (
        transformers.PretrainedConfig,
        transformers.PreTrainedModel,
        transformers.PreTrainedTokenizerBase,
    ):
        original = target.from_pretrained

        if getattr(original, "_redirects_pinned_hub_models", False):
            continue

        # Accessing a classmethod on the class yields a method already bound to
        # that class, so the underlying function is called with `cls` passed
        # explicitly: calling the bound method would re-bind every subclass to
        # the base, and `PreTrainedModel.from_pretrained` builds `cls(config)`.
        def from_pretrained(cls, name_or_path, *args, _original=original, **kwargs):
            return _original.__func__(
                cls,
                mapping.get(str(name_or_path), name_or_path),
                *args,
                **kwargs,
            )

        from_pretrained._redirects_pinned_hub_models = True
        target.from_pretrained = classmethod(from_pretrained)


class QAFactEvalBackend(ExternalMetricBackend):
    revision = "QAFactEval@01177f11;QAEval==0.1.0;lerc_quip;bart-large@cb48c136"

    # Upstream's `run.py` defaults these to 32/32/8, and this backend inherited
    # those numbers. They do not fit here. Three models sit on the one T4 the
    # metric is pinned to (the README explains why that GPU cannot move), and
    # the answering stage is the expensive one: `qaeval` windows each source
    # into 384-token spans at a stride of 128, and a source here is a whole
    # review set rather than the news article upstream sized for, so a single
    # question becomes twenty-odd windows. At 32 the run died partway into its
    # first batch with 13.70 GiB of the T4's 14.56 GiB already allocated.
    #
    # These are `qaeval`'s own constructor defaults instead - a fourfold cut on
    # the two stages that overflowed, and a number the library ships as
    # reasonable rather than one picked to make a single trace fit.
    #
    # They are deliberately absent from `revision`, unlike SummaC's options.
    # All three only decide how many sequences reach the GPU at once: the
    # answering stage pads every window to 384 and LERC-QuIP pads to 512
    # regardless, so a batch size is a memory budget rather than part of what a
    # score means. Folding it in would force a full re-score of the corpus
    # every time the metric was fitted to a different GPU.
    DEFAULT_OPTIONS = {
        "generation_batch_size": 8,
        "answering_batch_size": 8,
        "lerc_batch_size": 8,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown QAFactEval options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        batch_sizes = {}

        for name in self.DEFAULT_OPTIONS:
            size = int(settings[name])

            if size < 1:
                raise ValueError(f"QAFactEval {name} must be positive, got {size}.")

            batch_sizes[name] = size

        repository = os.getenv("QAFACTEVAL_HOME")
        model_directory = os.getenv("QAFACTEVAL_MODEL_DIR")

        if not repository or not Path(repository).is_dir():
            raise ValueError(
                "QAFACTEVAL_HOME must point to the pinned official QAFactEval checkout."
            )

        if not model_directory or not Path(model_directory).is_dir():
            raise ValueError(
                "QAFACTEVAL_MODEL_DIR must point to the downloaded official models."
            )

        model_root = Path(model_directory)
        base_model_path = model_root / "bart-large"

        if not base_model_path.is_dir():
            raise ValueError(
                f"{base_model_path} is missing. The generation archive rebuilds "
                "its base checkpoint by name before loading its own weights, so "
                "`prepare_model_weights` vendors it beside the other QAFactEval "
                "downloads; re-run that function before scoring."
            )

        # Must precede construction, not just the import: the archive resolves
        # the base checkpoint while QAFactEval's models are being built.
        redirect_pinned_hub_models({"facebook/bart-large": str(base_model_path)})

        sys.path.insert(0, repository)
        from qafacteval import QAFactEval

        cuda_device = 0 if worker_device(device).startswith("cuda") else -1
        self.model = QAFactEval(
            lerc_quip_path=str(model_root / "quip-512-mocha"),
            generation_model_path=str(model_root / "generation" / "model.tar.gz"),
            answering_model_dir=str(model_root / "answering"),
            lerc_model_path=str(model_root / "lerc" / "model.tar.gz"),
            lerc_pretrained_model_path=str(model_root / "lerc" / "pretraining.tar.gz"),
            cuda_device=cuda_device,
            use_lerc_quip=True,
            verbose=False,
            **batch_sizes,
        )

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        sources = [source for source, _ in pairs]
        summaries = [[summary] for _, summary in pairs]
        batch_results = self.model.score_batch_qafacteval(
            sources,
            summaries,
            return_qa_pairs=True,
        )

        return [
            {
                "score": float(result[0]["qa-eval"]["lerc_quip"]),
                "details": {
                    "family": "qa_factual_consistency",
                    "answer_overlap": "lerc_quip",
                },
            }
            for result in batch_results
        ]


def review_premises(
    source: str,
    length: Any,
    budget: int,
) -> tuple[list[str], int]:
    """One premise per review, the rule every premise-based metric here follows.

    A review becomes its own premise unless it alone exceeds `budget`; only then
    is it split, at sentence boundaries, into the fewest pieces that fit. How
    many reviews were split is returned beside the premises so it can be
    reported. `length` measures text in whatever unit `budget` is stated in.
    The "Review N:" headers are dropped, as `split_source_documents` does.

    One review per premise, rather than several packed together, keeps a
    premise independent of the order reviews were presented in, and gives each
    single-document checker the kind of input it was built for.
    """

    premises: list[str] = []
    split_reviews = 0

    for sentences in split_source_documents(source):
        review = " ".join(sentences)

        if length(review) <= budget:
            premises.append(review)
            continue

        split_reviews += 1
        piece: list[str] = []

        for sentence in sentences:
            if piece and length(" ".join([*piece, sentence])) > budget:
                premises.append(" ".join(piece))
                piece = []
            piece.append(sentence)

        if piece:
            premises.append(" ".join(piece))

    return premises or [""], split_reviews


def summac_input_statistics(imager: Any, source: str, summary: str) -> dict[str, int]:
    """What SummaC actually compares, and what it silently leaves out.

    SummaC drops every premise and summary unit of ten characters or fewer,
    keeps only the first `max_doc_sents` premises, and truncates the premise of
    any pair past `max_input_length` tokens. None of this is reported by the
    library, so it is counted here against its own splitter and tokenizer.
    """

    import nltk

    granularities = imager.grans if len(imager.grans) == 2 else imager.grans * 2

    def unfiltered(text: str, granularity: str) -> int:
        if granularity == "paragraph":
            separator = "\n\n" if "\n\n" in text else "\n"
            return sum(bool(part.strip()) for part in text.split(separator))
        return len(nltk.tokenize.sent_tokenize(text))

    premises = imager.split_text(source, granularity=granularities[0])
    used = premises[: imager.max_doc_sents] if imager.max_doc_sents else premises
    hypotheses = imager.split_text(summary, granularity=granularities[1])
    truncated = sum(
        len(imager.tokenizer(premise, hypothesis)["input_ids"]) > imager.max_input_length
        for premise in used
        for hypothesis in hypotheses
    )

    return {
        "premises": len(used),
        "premises_dropped_short": unfiltered(source, granularities[0]) - len(premises),
        "premises_dropped_cap": len(premises) - len(used),
        "summary_units": len(hypotheses),
        "summary_units_dropped_short": unfiltered(summary, granularities[1]) - len(hypotheses),
        "truncated_pairs": int(truncated),
    }


def library_versions(*packages: str) -> str:
    """Installed versions, for a revision that names the runtime it ran on."""

    from importlib.metadata import version

    return ";".join(f"{package}=={version(package)}" for package in packages)


class AlignScoreBackend(ExternalMetricBackend):
    """AlignScore-large, alignment probability from its 3-way NLI head.

    Each summary sentence is scored against every premise, takes its best
    premise, and the score is the mean over sentences - the library's `nli_sp`
    aggregation, and the same max-then-mean shape as SummaC-ZS.

    `chunking` decides what a premise is:

    `reviews` (the setting reported): one review per premise, split at sentence
    boundaries only when it alone exceeds `premise_tokens` RoBERTa tokens, so
    every premise fits the 512-token input beside a summary sentence.

    `library`: AlignScore's own chunking, kept to report as an ablation. It cuts
    the source into groups of an equal number of sentences sized for about 350
    words. On these sources the groups run up to 653 words, and 14.6% of
    (premise, sentence) pairs on the manual sample exceed 512 tokens and are
    truncated from the premise without warning - up to 38% of it.

    The checkpoint stores `roberta-large` by Hub name and rebuilds the encoder
    from it on load, so the image resolves that name offline from a pinned cache
    (see `prepare_model_weights`). It is also a Lightning 1.x checkpoint, which
    torch's weights-only loading refuses; the image allows the full load for it.
    """

    BASE_REVISION = "alignscore@a0936d5a;AlignScore-large@8509e78d;roberta-large@722cf37b"
    # RoBERTa's limit, of which a sentence pair spends four on special tokens.
    MAXIMUM_TOKENS = 512
    DEFAULT_OPTIONS: dict[str, Any] = {
        "chunking": "reviews",
        # Leaves 108 tokens for the summary sentence; the longest on the manual
        # sample is well under that, and any pair that still overflows is
        # counted in the score's details rather than truncated silently.
        "premise_tokens": 400,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown AlignScore options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        self.chunking = str(settings["chunking"])
        self.premise_tokens = int(settings["premise_tokens"])

        if self.chunking not in {"reviews", "library"}:
            raise ValueError(f"AlignScore chunking must be 'reviews' or 'library', not {self.chunking!r}.")

        if not 0 < self.premise_tokens < self.MAXIMUM_TOKENS - 4:
            raise ValueError(f"premise_tokens must leave room for a summary sentence: {self.premise_tokens}.")

        self.revision = ";".join(
            (
                self.BASE_REVISION,
                library_versions("torch", "transformers", "pytorch-lightning"),
                (
                    f"premises=reviews;premise_tokens={self.premise_tokens}"
                    if self.chunking == "reviews"
                    else "premises=library"
                ),
                "claims=sentence;op1=max;op2=mean;precision=fp32",
            )
        )

        from alignscore import AlignScore

        self.model = AlignScore(
            model="roberta-large",
            batch_size=32,
            device=worker_device(device),
            ckpt_path=os.environ["ALIGNSCORE_CHECKPOINT_PATH"],
            evaluation_mode="nli_sp",
            verbose=False,
        )
        self.inferencer = self.model.model
        self.inferencer.disable_progress_bar_in_inference = True
        # AlignScore pads every pair to `model_max_length`. roberta-large ships
        # no tokenizer_config.json, so loaded from the pinned cache that is
        # transformers' 1e30 placeholder and padding overflows; the releases
        # AlignScore was written against hard-coded RoBERTa's 512.
        self.inferencer.tokenizer.model_max_length = self.MAXIMUM_TOKENS

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def _tokens(self, text: str) -> int:
        return len(self.inferencer.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _score_reviews(self, source: str, summary: str) -> dict[str, Any]:
        from nltk.tokenize import sent_tokenize

        premises, split_reviews = review_premises(source, self._tokens, self.premise_tokens)
        claims = sent_tokenize(summary) or [summary]
        # The library's `nli_sp` grid, built over these premises instead of its own.
        probabilities = self.inferencer.inference(
            [premise for premise in premises for _ in claims],
            [claim for _ in premises for claim in claims],
        )[2][:, 0]
        score = probabilities.view(len(premises), len(claims)).max(dim=0).values.mean().item()
        claim_tokens = [self._tokens(claim) for claim in claims]
        truncated = sum(
            self._tokens(premise) + tokens + 4 > self.MAXIMUM_TOKENS
            for premise in premises
            for tokens in claim_tokens
        )

        return {
            "score": float(score),
            "details": {
                "family": "factual_consistency",
                "variant": "AlignScore-large",
                "premises": len(premises),
                "split_reviews": split_reviews,
                "truncated_pairs": int(truncated),
            },
        }

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters

        if self.chunking == "reviews":
            return [self._score_reviews(source, summary) for source, summary in pairs]

        scores = self.model.score(
            contexts=[source for source, _ in pairs],
            claims=[summary for _, summary in pairs],
        )

        return [
            {
                "score": float(score),
                "details": {"family": "factual_consistency", "variant": "AlignScore-large"},
            }
            for score in scores
        ]


class MiniCheckBackend(ExternalMetricBackend):
    """MiniCheck-Flan-T5-Large, scored per summary sentence and averaged.

    MiniCheck checks one claim against a document, returning the support
    probability of the document's best premise. A summary is several claims, so
    each sentence is checked on its own - the authors recommend this, and it is
    how LLM-AggreFact, which they evaluate on, is scored - and the summary's
    score is their mean, the same aggregation as AlignScore and SummaC-ZS.

    `chunking` decides what a premise is:

    `reviews` (the setting reported): one review per premise, split at sentence
    boundaries only when it alone exceeds `premise_words` words, the unit and
    size of MiniCheck's own chunks.

    `library`: MiniCheck's own chunking, kept to report as an ablation - the
    same 500-word budget filled sentence by sentence, regardless of where one
    review ends and the next begins. Neither setting truncates here: the
    longest input on the manual sample is 902 of the 2048 tokens allowed.
    """

    BASE_REVISION = "minicheck@b58b9fa6;MiniCheck-Flan-T5-Large@96eafd01"
    DEFAULT_OPTIONS: dict[str, Any] = {
        "chunking": "reviews",
        "premise_words": 500,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown MiniCheck options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        self.chunking = str(settings["chunking"])
        self.premise_words = int(settings["premise_words"])

        if self.chunking not in {"reviews", "library"}:
            raise ValueError(f"MiniCheck chunking must be 'reviews' or 'library', not {self.chunking!r}.")

        if self.premise_words <= 0:
            raise ValueError(f"premise_words must be positive: {self.premise_words}.")

        self.revision = ";".join(
            (
                self.BASE_REVISION,
                library_versions("torch", "transformers"),
                (
                    f"premises=reviews;premise_words={self.premise_words}"
                    if self.chunking == "reviews"
                    else f"premises=library;premise_words={self.premise_words}"
                ),
                "claims=sentence;op1=max;op2=mean;precision=fp32",
            )
        )
        # MiniCheck places the model itself (`device_map="auto"`).
        del device

        from minicheck.minicheck import MiniCheck

        self.model = MiniCheck(
            model_name="flan-t5-large",
            batch_size=16,
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        self.inferencer = self.model.model

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def _input_tokens(self, premise: str, claim: str) -> int:
        tokenizer = self.inferencer.tokenizer
        return len(tokenizer("predict: " + premise + tokenizer.eos_token + claim)["input_ids"])

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        from nltk.tokenize import sent_tokenize

        results = []

        for source, summary in pairs:
            claims = sent_tokenize(summary) or [summary]
            details: dict[str, Any] = {
                "family": "factual_consistency",
                "variant": "MiniCheck-Flan-T5-Large",
            }

            if self.chunking == "reviews":
                premises, split_reviews = review_premises(
                    source, lambda text: len(text.split()), self.premise_words
                )
                per_sentence = [
                    float(
                        self.inferencer.inference(premises, [claim] * len(premises))[
                            "max_support_prob"
                        ]
                    )
                    for claim in claims
                ]
                details.update(
                    premises=len(premises),
                    split_reviews=split_reviews,
                    truncated_pairs=sum(
                        self._input_tokens(premise, claim) > self.inferencer.max_model_len
                        for premise in premises
                        for claim in claims
                    ),
                )
            else:
                _, probabilities, _, _ = self.model.score(
                    docs=[source] * len(claims),
                    claims=claims,
                    chunk_size=self.premise_words,
                )
                per_sentence = [float(p) for p in probabilities]

            details["sentence_scores"] = per_sentence
            results.append(
                {"score": sum(per_sentence) / len(per_sentence), "details": details}
            )

        return results


class BespokeMiniCheckBackend(ExternalMetricBackend):
    """Bespoke-MiniCheck-7B, the strongest checker of the MiniCheck series.

    An InternLM2-based 7B model that answers "Yes"/"No" to whether a document
    supports a claim. The score is MiniCheck's own: the probability mass of
    first tokens reading "yes" among the top five, from the prompt its library
    builds. It sits beside the 27-32B G-Eval judges as a trained checker at
    LLM scale.

    It runs on the model's own reference code (transformers, trust_remote_code)
    rather than through vLLM as the library does. vLLM 0.21 loads this
    architecture incorrectly: on "The hotel room was clean..." it returned the
    same top tokens (" ", " clear", " conj") for "The room was clean." and "The
    room was dirty.", whatever the start-token or rope-scaling settings, while
    the reference code answers Yes (p=0.98) and No (p=0.99). The model's input
    is token for token the library's: its prompt text, encoded as vLLM encodes
    it - special tokens added, so the template's own `<s>` is preceded by a
    second, as in every score the authors published. Attention uses PyTorch's
    SDPA kernel, the same computation as the config's eager one without its
    quadratic memory.

    One prompt per forward pass, as G-Eval is scored. Batching with padding on
    the right keeps pads out of every attended position, yet it still moved
    P(yes) by up to 0.03 on real premises: a batch runs through differently
    shaped bf16 kernels than a single prompt. Unbatched, a score depends on its
    own prompt alone.

    `chunking`:

    `reviews` (the setting reported): one review per premise, each summary
    sentence checked against each, best premise per sentence, mean over
    sentences - the rule every premise-based metric here follows.

    `library`: the authors' setting, kept as an ablation. The library's own
    chunking at its budget of `max_model_len` less 300 tokens - which every
    source here fits whole - and the summary split into sentences, scored by
    the *minimum* over sentences of the best chunk ("SentenceFusion") rather
    than the mean used everywhere else here.

    Licensed CC BY-NC 4.0 by Bespoke Labs.
    """

    MODEL = "bespokelabs/Bespoke-MiniCheck-7B"
    BASE_REVISION = "minicheck@b58b9fa6;Bespoke-MiniCheck-7B@1ed7786b"
    DEFAULT_OPTIONS: dict[str, Any] = {
        "chunking": "reviews",
        # The library's default context; its chunk budget is this less 300.
        "max_model_len": 32768,
        # Prompts per forward pass. Above one the score depends on which prompts
        # share a batch (see the class docstring), so it is part of the revision.
        "prompts_per_batch": 1,
    }

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        settings = {**self.DEFAULT_OPTIONS, **(options or {})}
        unknown = sorted(set(settings).difference(self.DEFAULT_OPTIONS))

        if unknown:
            raise ValueError(
                f"Unknown Bespoke-MiniCheck options: {unknown}. "
                f"Known options: {sorted(self.DEFAULT_OPTIONS)}."
            )

        self.chunking = str(settings["chunking"])
        self.max_model_len = int(settings["max_model_len"])
        self.prompts_per_batch = int(settings["prompts_per_batch"])

        if self.chunking not in {"reviews", "library"}:
            raise ValueError(f"Bespoke-MiniCheck chunking must be 'reviews' or 'library', not {self.chunking!r}.")

        self.revision = ";".join(
            (
                self.BASE_REVISION,
                library_versions("torch", "transformers"),
                f"backend=reference;tokens=library;attention=sdpa;batch={self.prompts_per_batch}",
                f"max_model_len={self.max_model_len}",
                (
                    "premises=reviews;claims=sentence;op1=max;op2=mean"
                    if self.chunking == "reviews"
                    else "premises=library;claims=sentence;op1=max;op2=min"
                ),
                "precision=bf16",
            )
        )

        import torch
        from minicheck.utils import SYSTEM_PROMPT, USER_PROMPT
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = worker_device(device)
        self.system_prompt, self.user_prompt = SYSTEM_PROMPT, USER_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.MODEL,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        # The library's chunk budget: the context less 300 tokens for the claim.
        self.chunk_tokens = self.max_model_len - 300
        # First tokens the library counts as "yes": it compares each top token's
        # decoded text, lower-cased, with "yes".
        self.yes_ids = {
            index
            for token, index in self.tokenizer.get_vocab().items()
            if self.tokenizer.convert_tokens_to_string([token]).lower() == "yes"
        }

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return self.score_many([(source, summary)], parameters)[0]

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _prompt_ids(self, document: str, claim: str) -> list[int]:
        """MiniCheck's prompt for one (document, claim), tokenized as the library's is.

        The library hands vLLM this text, and vLLM encodes it with the
        tokenizer's special tokens, which prepends `<s>` to the template's own.
        """

        user = self.user_prompt.replace("[DOCUMENT]", document).replace("[CLAIM]", claim)
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(text)["input_ids"]

    def support_probabilities(self, prompts: list[list[int]]) -> list[float]:
        """P(yes) per prompt, as MiniCheck reads it: yes-tokens among the top five."""

        torch = self.torch
        order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        pad = self.tokenizer.pad_token_id
        results: list[float] = [0.0] * len(prompts)
        start = 0

        while start < len(order):
            # Sorting by length keeps any padding small when batching.
            end = min(start + self.prompts_per_batch, len(order))
            batch = [prompts[index] for index in order[start:end]]
            width = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, ids in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1

            with torch.inference_mode():
                hidden = self.model.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention.to(self.device),
                )[0]
                last = hidden[torch.arange(len(batch)), attention.sum(dim=1).to(self.device) - 1]
                top = torch.log_softmax(self.model.output(last).float(), dim=-1).topk(5)

            for row, index in enumerate(order[start:end]):
                results[index] = float(
                    sum(
                        value.exp().item()
                        for value, token in zip(top.values[row], top.indices[row].tolist())
                        if token in self.yes_ids
                    )
                )
            start = end

        return results

    def _library_chunks(self, document: str) -> list[str]:
        """The library's own chunking of a document, at its own budget."""

        from minicheck.inference import LLMCheck

        chunker = LLMCheck.__new__(LLMCheck)
        chunker.tokenizer = self.tokenizer
        chunker.chunk_size = self.chunk_tokens
        chunker.doc_chunk_cache = {}
        return chunker.get_all_chunks_per_doc(document, "")["doc_chunks"]

    def score_many(
        self,
        pairs: list[tuple[str, str]],
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        from nltk.tokenize import sent_tokenize

        layouts, prompts = [], []
        for source, summary in pairs:
            if self.chunking == "reviews":
                premises, split_reviews = review_premises(source, self._tokens, self.chunk_tokens)
            else:
                premises, split_reviews = self._library_chunks(source), 0
            claims = sent_tokenize(summary) or [summary]
            batch = [self._prompt_ids(premise, claim) for premise in premises for claim in claims]
            layouts.append(
                (
                    len(premises),
                    len(claims),
                    split_reviews,
                    sum(len(ids) > self.max_model_len for ids in batch),
                )
            )
            prompts.extend(batch)

        probabilities = self.support_probabilities(prompts)

        results, start = [], 0
        for premises, claims, split_reviews, overlong in layouts:
            grid = probabilities[start:start + premises * claims]
            start += premises * claims
            per_sentence = [
                max(grid[premise * claims + claim] for premise in range(premises))
                for claim in range(claims)
            ]
            score = (
                sum(per_sentence) / len(per_sentence)
                if self.chunking == "reviews"
                else min(per_sentence)
            )
            results.append(
                {
                    "score": score,
                    "details": {
                        "family": "factual_consistency",
                        "variant": "Bespoke-MiniCheck-7B",
                        "premises": premises,
                        "split_reviews": split_reviews,
                        "prompts_over_context": int(overlong),
                        "sentence_scores": per_sentence,
                    },
                }
            )

        return results


class FactGraphBackend(ExternalMetricBackend):
    """FactGraph (Ribeiro et al., NAACL 2022), on the authors' evaluation path.

    FactGraph classifies one summary sentence against a document. The text
    encoder reads the sentence and the document as one 512-token pair, which on
    a review set keeps only its first reviews; the graph encoder reads AMR
    graphs of the sentence and of the `NUMBER_GRAPHS` document sentences closest
    to it (paraphrase-mpnet-base-v2 cosine similarity), which can come from
    anywhere in the source. The score is P(CORRECT) of the trained checkpoint,
    averaged over summary sentences as every other checker here does - the
    authors define FactGraph only for single-sentence summaries.

    Everything between the source and that probability is the authors' own
    code, imported from the pinned checkout: their sentence selection
    (`SelectSentencesScore`), their AMR alignment and graph simplification
    (`align_amr`, `process_amr`) and their input encoding
    (`preprocess_function`), against the released checkpoint. Three things
    differ from running their scripts, none of them in what is computed:

    - Newlines in the source become spaces. Their selection deletes newlines
      outright, which would weld the last word of one review to the first of
      the next.
    - Each sentence is parsed to AMR on its own (amrlib's batch of 12 pads
      sentences together under beam search), and each claim reaches the model
      on its own, so a score depends on its own input alone and not on which
      summaries shared its batch. Parses are cached by sentence, which is what
      makes that affordable: a review set's sentences recur across summaries.
    - A sentence whose parse or alignment fails is dropped as their scripts
      drop it, but counted in `details` rather than only printed.

    The stack is FactGraph's, not this project's: Python 3.7, torch 1.9 with
    PyTorch Geometric, and the repository's own patched copy of transformers
    4.6.1, which the pickled checkpoint needs to unpickle. It therefore runs
    behind `LegacyMetricProcess`, like QAFactEval.
    """

    NUMBER_GRAPHS = 5   # evaluate.sh's NUMBER_GRAPHS, the checkpoint's setting
    revision = (
        "fact-graph@165285a4;factgraph@sha256:e07e209a;"
        "electra-base-discriminator@1ae76a97;paraphrase-mpnet-base-v2@sha256:a52c823f;"
        "amrlib==0.7.1;parse_xfm_bart_large-v0_1_0@sha256:2fcfe211;bart-large@cb48c136;"
        "amr-utils@be5534db;fast_align@cab1e9aa;torch==1.9.0;"
        "article=source;graphs=5;claims=sentence;op2=mean;batch=1;precision=fp32"
    )

    def __init__(self, device: str, options: dict[str, Any] | None = None) -> None:
        reject_unsupported_options("factgraph", options)

        home = os.getenv("FACTGRAPH_HOME")
        model_root = os.getenv("FACTGRAPH_MODEL_DIR")
        bart_tokenizer = os.getenv("FACTGRAPH_BART_TOKENIZER")

        if not home or not Path(home).is_dir():
            raise ValueError("FACTGRAPH_HOME must point to the pinned FactGraph checkout.")

        for name, path in (("FACTGRAPH_MODEL_DIR", model_root),
                           ("FACTGRAPH_BART_TOKENIZER", bart_tokenizer)):
            if not path or not Path(path).is_dir():
                raise ValueError(
                    f"{name} must point to a directory `prepare_model_weights` fills."
                )

        import torch

        if not worker_device(device).startswith("cuda"):
            # Their graph batching and input collation call .cuda() directly.
            raise ValueError("FactGraph runs on CUDA only.")

        preprocess_directory = str(Path(home) / "data" / "preprocess")
        source_directory = str(Path(home) / "src")

        # Both directories hold a module named `utils`. The preprocessing
        # modules are imported first, binding the preprocessing one; it is then
        # dropped so the model modules bind their own. `transformers` resolves
        # to the repository's patched copy throughout, so the parser, the
        # sentence encoder and the checkpoint all run on the one version.
        sys.path[:0] = [preprocess_directory, source_directory]
        import amrlib
        import augmentation_ops
        import preprocess_evaluate

        sys.modules.pop("utils", None)
        sys.path.remove(preprocess_directory)
        import models
        import utils_evaluate

        self.torch = torch
        self.preprocess = preprocess_evaluate
        self.encode = utils_evaluate.preprocess_function

        root = Path(model_root)
        self.selector = augmentation_ops.SelectSentencesScore(
            model_type=str(root / "paraphrase-mpnet-base-v2")
        )
        # amrlib reads the tokenizer by the Hub name the model was trained from;
        # the pinned copy stands in for it offline. One sentence per pass.
        self.parser = amrlib.load_stog_model(
            model_dir=str(root / "model_parse_xfm_bart_large-v0_1_0"),
            tok_name_or_path=bart_tokenizer,
            batch_size=1,
        )
        # evaluate.py's own tokenizer: ELECTRA's, plus the AMR edge labels.
        _, self.tokenizer = models.load_model_and_tokenizer(
            "factgraph", str(root / "electra-base-discriminator"), 2, 32
        )

        checkpoints = sorted((root / "checkpoints" / "factgraph").glob("*/model.pt"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected one FactGraph checkpoint, found {checkpoints}.")
        self.model = torch.load(str(checkpoints[0]))
        self.model.to("cuda")
        self.model.eval()

        self._graphs: dict[str, str | None] = {}
        self._alignments: dict[str, tuple[str, str] | None] = {}

        # A full pass at startup, so a missing kernel or binary fails here
        # rather than on the first real batch.
        probe = self.score("The battery lasts all day. The screen is bright.",
                           "The battery lasts all day.", {})
        if not 0.0 <= probe["score"] <= 1.0:
            raise RuntimeError(f"FactGraph startup probe returned {probe!r}.")

    def _parse(self, sentence: str) -> str | None:
        if sentence not in self._graphs:
            try:
                self._graphs[sentence] = self.parser.parse_sents([sentence], add_metadata=True)[0]
            except Exception:  # noqa: BLE001 - their extract_amrs drops these too
                self._graphs[sentence] = None
        return self._graphs[sentence]

    def _read(self, graph: str) -> Any:
        """A parse as the AMR object their code reads, by their own route."""

        from tempfile import NamedTemporaryFile

        handle = NamedTemporaryFile(mode="w+", delete=False)
        try:
            handle.write(graph)
            handle.close()
            return self.preprocess.reader.load(handle.name, remove_wiki=True)[0]
        finally:
            os.unlink(handle.name)

    def _document_amr(self, sentence: str) -> Any | None:
        graph = self._parse(sentence)
        if graph is None:
            return None
        try:
            return self._read(graph)
        except Exception:  # noqa: BLE001
            return None

    def _summary_amr(self, sentence: str) -> Any | None:
        """The claim's AMR with its token alignment, which process_amr reads."""

        amr = self._document_amr(sentence)
        if amr is None:
            return None

        if sentence not in self._alignments:
            aligned = self.preprocess.align_amr(amr)
            self._alignments[sentence] = (
                (aligned.metadata["tok"], aligned.metadata["alignments"])
                if "alignments" in aligned.metadata else None
            )

        if self._alignments[sentence] is None:
            return None

        # A fresh object each time: their graph simplification may modify it.
        amr.metadata["tok"], amr.metadata["alignments"] = self._alignments[sentence]
        return amr

    def _probability_correct(self, example: dict[str, Any]) -> float:
        torch = self.torch
        encoded = self.encode(
            {key: [example[key]] for key in ("summary", "article", "graphs", "graph_summary")},
            self.tokenizer,
            self.NUMBER_GRAPHS,
        )
        tensors = {
            key: torch.tensor(encoded[key], dtype=torch.long).cuda()
            for key in ("input_ids", "attention_mask", "graphs", "graphs_attn", "mask_graph")
        }
        graph_structure = {
            "num_graphs": self.NUMBER_GRAPHS + 1,
            "mask_graph": tensors["mask_graph"],
            "edge_index": encoded["edge_index"],
            "edge_type": encoded["edge_type"],
        }
        with torch.no_grad():
            logits = self.model(tensors["input_ids"], tensors["attention_mask"],
                                tensors["graphs"], tensors["graphs_attn"], graph_structure)
        # evaluate.py's labels: 1 is CORRECT.
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def score(
        self,
        source: str,
        summary: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        del parameters
        article = re.sub(r"\s*\n\s*", " ", source).strip()
        claims = [
            sentence.text.strip()
            for sentence in self.selector.spacy(summary.strip()).sents
            if sentence.text.strip()
        ] or [summary.strip()]

        examples = []
        for claim in claims:
            example = self.selector.transform({"summary": claim, "article": article},
                                              number_sents=self.NUMBER_GRAPHS)
            if example is None:
                raise RuntimeError(
                    f"FactGraph's sentence selection returned nothing for {claim!r} "
                    "(it times out after 15 s or finds no sentence)."
                )
            examples.append(example)

        document_sentences = list(dict.fromkeys(
            selected[0] for example in examples
            for selected in json.loads(example["sentences"])
        ))
        document_amrs = [amr for amr in map(self._document_amr, document_sentences)
                         if amr is not None]
        summary_amrs = [amr for amr in map(self._summary_amr,
                                           dict.fromkeys(e["summary"] for e in examples))
                        if amr is not None]

        with contextlib.redirect_stdout(sys.stderr):
            self.preprocess.process_amr(examples, document_amrs, summary_amrs,
                                        self.NUMBER_GRAPHS)

        sentence_scores = [self._probability_correct(example) for example in examples]

        return {
            "score": sum(sentence_scores) / len(sentence_scores),
            "details": {
                "family": "graph_factual_consistency",
                "variant": "FactGraph",
                "sentence_scores": sentence_scores,
                "document_graphs": [len(example["graphs"]) for example in examples],
                "claims_without_graph": sum(
                    not example["graph_summary"]["amr_simple"] for example in examples
                ),
                "document_sentences_unparsed": len(document_sentences) - len(document_amrs),
            },
        }


def reject_unsupported_options(
    metric: str,
    options: dict[str, Any] | None,
) -> None:
    """Fail loudly rather than ignore an option this backend cannot honour."""

    if options:
        raise ValueError(
            f"The {metric} backend takes no constructor options, but "
            f"experiment.yaml supplied {sorted(options)}. Per-call settings "
            "belong under `parameters`, not `options`."
        )


def build_external_backend(
    metric: str,
    device: str,
    options: dict[str, Any] | None = None,
) -> ExternalMetricBackend:
    backends = {
        "summac": SummaCBackend,
        "summac_zs": SummaCZeroShotBackend,
        "bartscore": BARTScoreBackend,
        "supert": SupertBackend,
        "qafacteval": QAFactEvalBackend,
        "alignscore": AlignScoreBackend,
        "minicheck": MiniCheckBackend,
        "minicheck_7b": BespokeMiniCheckBackend,
        "factgraph": FactGraphBackend,
    }

    return backends[metric](device, options)


def external_metric_main(argv: list[str] | None = None) -> None:
    """Persistent JSONL boundary for the four isolated local metrics."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        required=True,
        choices=["summac", "summac_zs", "bartscore", "supert", "qafacteval", "factgraph"],
    )
    parser.add_argument("--device", default="auto")
    # Constructor options reach an in-process backend as a dict. A metric run
    # behind `LegacyMetricProcess` has no such call to receive them, so they
    # cross the boundary here, as the same JSON the protocol already speaks.
    parser.add_argument("--options", default="{}")
    arguments = parser.parse_args(argv)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend = build_external_backend(
                metric=arguments.metric,
                device=arguments.device,
                options=json.loads(arguments.options),
            )
    except Exception as exception:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exception).__name__}: {exception}",
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from exception

    print(
        json.dumps(
            {
                "status": "ready",
                "metric": arguments.metric,
                "revision": backend.revision,
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        if not line.strip():
            continue

        try:
            request = json.loads(line)

            # A `batch` message hands the whole group to `score_many`, so a
            # metric that batches internally keeps doing so across the process
            # boundary. The single-request form below is unchanged.
            if "batch" in request:
                items = request["batch"]
                parameters = items[0].get("parameters", {}) if items else {}

                if any(item.get("parameters", {}) != parameters for item in items):
                    raise ValueError(
                        "one external metric batch contained different "
                        "parameter sets."
                    )

                with contextlib.redirect_stdout(sys.stderr):
                    results = backend.score_many(
                        pairs=[
                            (item["source_text"], item["summary_text"])
                            for item in items
                        ],
                        parameters=parameters,
                    )

                print(
                    json.dumps({"results": results}, ensure_ascii=False),
                    flush=True,
                )
                continue

            with contextlib.redirect_stdout(sys.stderr):
                result = backend.score(
                    source=request["source_text"],
                    summary=request["summary_text"],
                    parameters=request.get("parameters", {}),
                )

            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exception:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "error": f"{type(exception).__name__}: {exception}",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    external_metric_main()
