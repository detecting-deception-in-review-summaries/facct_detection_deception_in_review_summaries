from __future__ import annotations

import json
import math
import random
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError
from tqdm.auto import tqdm

from .common import (
    append_jsonl,
    check_context_budget,
    check_prompt_budget,
    digest,
    read_jsonl,
    stable_id,
    stable_json,
    write_jsonl,
)
from .data import format_reviews
from .generation import (
    Summarizer,
    generate_text,
    load_style_prompt,
    preflight,
)
from .llms import (
    BatchSession,
    DeferredRequest,
    ResponseCache,
    build_provider_request,
    client,
    is_retryable_error,
    model_identity,
    provider_supports_json,
)
from .models import Edit, Model, Modification, Product, Summary
from .reporting import (
    report_modification_progress,
    report_modification_status,
    stage_progress_bar,
)

# What a modification prompt returns for a slot the summary cannot fill - no
# aspect of that kind to alter, no quantifier to swap.
INAPPLICABLE_MARKER = "INVALID"

# One self-contained, one-shot prompt per family: each finds its own material
# (COST/PURPOSE/EFFICACY/SAFETY) aspects and returns all five count/direction
# outputs from a single call - no aspect-inventory call, no shared preamble.
# `experiment.yaml`'s `prompts.tasks` can override this mapping. See
# LLMModifier.
PROMPTS_BY_FAMILY = {
    "factual_misrepresentation": "misrepresentation.txt",
    "omission": "omission.txt",
    "quantifier_change": "quantifier.txt",
}


@lru_cache(maxsize=None)
def load_modification_prompt(name: str) -> tuple[str, str]:
    """Read and hash one modification prompt file. Cached: one read per file."""

    text = (Path("prompts/modifications") / name).read_text(encoding="utf-8")

    return text, digest(text)


def prompts_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the `prompts` block of experiment.yaml, with defaults filled in.

    Every default here matches the filenames this module falls back to on its
    own, so a config with no `prompts` section at all still resolves.
    """

    configured = config.get("prompts") or {}

    return {"tasks": {**PROMPTS_BY_FAMILY, **configured.get("tasks", {})}}


def clear_prompt_caches() -> None:
    """Drop every cached prompt read.

    `load_modification_prompt` caches once per process, which is right for a
    pipeline run - the prompts do not change mid-run - and wrong for a tool
    that edits a prompt file between cells, such as
    tune_modification_prompt.ipynb, which must clear it to see the edit.
    """

    load_modification_prompt.cache_clear()


MODIFICATION_FAMILIES = (
    "factual_misrepresentation",
    "omission",
    "quantifier_change",
)
# Names match Modifier.name on PrimaryLLMModifier and JudgeModelModifier.
# run()'s llm_modifiers argument selects among these; both run by default.
LLM_MODIFIER_NAMES = ("primary_llm", "judge_model")
REVIEW_FILTER_DETAILS = {
    "remove_50pct_one_star": (1, 50, "advantageous"),
    "remove_100pct_one_star": (1, 100, "advantageous"),
    "remove_50pct_five_star": (5, 50, "disadvantageous"),
    "remove_100pct_five_star": (5, 100, "disadvantageous"),
}
MODIFICATION_CHOICES = (
    "all",
    *MODIFICATION_FAMILIES,
    *REVIEW_FILTER_DETAILS,
)


class EditResult(Model):
    text: str
    edits: list[Edit]
    explanation: str
    details: dict[str, Any] | None = None


class ModificationResponse(Model):
    """The JSON object requested by each family's one-shot prompt."""

    model_config = ConfigDict(extra="ignore")

    summary_count1_advantageous: str
    summary_count1_disadvantageous: str
    summary_count2: str
    summary_count3: str
    summary_count4: str


class Modifier(ABC):
    """Shared interface for all summary-modification generators."""

    name: str
    modifier_id: str
    # Recorded when `apply` returns None: this modifier had nothing to change.
    inapplicable_reason = "The requested condition is not applicable."

    @abstractmethod
    def apply(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> EditResult | None:
        """Apply one requested output from a modification prompt."""

    @abstractmethod
    def fingerprint(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> tuple[str, str]:
        """Return the (modifier, prompt_hash) this attempt would be produced by.

        `attempt_id` names a cell of the experimental design - one parent, one
        modifier role, one condition - and deliberately stays the same when the
        model or the prompt behind that cell changes, so the file keeps exactly
        one row per cell. This is what tells a resumed run whether the stored
        row is still the row the current configuration produces; when it is
        not, the attempt is re-run and the stale row replaced.
        """

class LLMModifier(Modifier):
    """Runs the one-shot, categorized-only, multi-count modification prompts.

    One call per family per summary: the family's own prompt finds its
    material aspects and returns all five count/direction outputs (see
    ModificationResponse) directly. There is nothing to pre-resolve before a
    family call - a request either answers immediately or comes back as a
    DeferredRequest under batch execution, the same way
    ReviewFilteringModifier's requests do.
    """

    modifier_id: str

    def __init__(
        self,
        name: str,
        model: dict[str, Any],
        config: dict[str, Any],
        batch_session: BatchSession,
        mock: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.modifier_id = model_identity(model)
        self.mock = mock
        self.task_files = prompts_config(config)["tasks"]
        self.request_ids: set[str] = set()
        self.request_errors: dict[str, str] = {}
        self.client = client(
            provider=model["provider"],
            mock=mock,
            batch_session=batch_session,
        )
        self.cache = ResponseCache(config["cache"])
        self.decoding = config["modification_decoding"]
        check_context_budget(
            model_key=self.name,
            model=model,
            context_budget=self.decoding["context_tokens"],
            stage="modification",
        )

        if not provider_supports_json(model["provider"]):
            raise ValueError(
                f"The modifier {self.name} uses provider {model['provider']}, "
                "which has no JSON-object response mode and cannot serve the "
                "modification stage."
            )

    def _build_request(self, prompt: str) -> dict[str, Any]:
        return build_provider_request(
            provider=self.model["provider"],
            model=self.model,
            prompt=prompt,
            decoding=self.decoding,
            json_object=True,
        )

    def _prompt_template(self, family: str) -> tuple[str, str]:
        return load_modification_prompt(self.task_files[family])

    def _response_request(
        self,
        parent_text: str,
        family: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_name = self.task_files[family]
        template, prompt_hash = self._prompt_template(family)
        prompt = template.replace("{input_text}", parent_text)
        prompt = prompt.replace("{{", "{").replace("}}", "}")

        check_prompt_budget(
            prompt_characters=len(prompt),
            decoding=self.decoding,
            subject=f"The {prompt_name} prompt",
            stage="modification",
        )

        request = self._build_request(prompt)
        cache_key = {
            "provider": self.model["provider"],
            "revision": model_identity(self.model),
            "prompt_file": prompt_name,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "request": request,
        }

        return request, cache_key

    def _load_response(
        self,
        parent_text: str,
        family: str,
    ) -> ModificationResponse:
        request, cache_key = self._response_request(parent_text, family)
        cache_id = self.cache.cache_id(cache_key)
        self.request_ids.add(cache_id)

        if cache_id in self.request_errors:
            raise RuntimeError(self.request_errors[cache_id])

        if self.mock:
            raw_response = mock_modification_response(parent_text)
        else:
            try:
                response, _ = self.cache.call(
                    key=cache_key,
                    llm_client=self.client,
                    payload=request,
                )
            except DeferredRequest:
                raise
            except Exception as exception:
                self.request_errors[cache_id] = str(exception)
                raise

            raw_response = response.text

        cleaned_response = strip_json_fence(raw_response)

        try:
            return ModificationResponse.model_validate_json(cleaned_response)
        except ValidationError as exception:
            raise RuntimeError(
                f"The {self.name} response could not be parsed as the "
                f"expected JSON object: {exception}. Response begins: "
                f"{cleaned_response[:200]!r}"
            ) from exception

    def fingerprint(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> tuple[str, str]:
        del parent
        _, prompt_hash = self._prompt_template(request["modification_family"])

        return self.modifier_id, prompt_hash

    def apply(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> EditResult | None:
        family = request["modification_family"]
        parent_text = parent.text or ""
        response = self._load_response(parent_text, family)
        modified_summary = getattr(response, request["response_field"])
        edits = derive_edits(parent_text, modified_summary)

        return EditResult(
            text=modified_summary,
            edits=edits,
            explanation="one-shot material prompt; no per-output explanation",
            details=None,
        )


class PrimaryLLMModifier(LLMModifier):
    def __init__(
        self,
        model: dict[str, Any],
        config: dict[str, Any],
        batch_session: BatchSession,
        mock: bool = False,
    ) -> None:
        super().__init__(
            name="primary_llm",
            model=model,
            config=config,
            batch_session=batch_session,
            mock=mock,
        )


class JudgeModelModifier(LLMModifier):
    def __init__(
        self,
        model: dict[str, Any],
        config: dict[str, Any],
        batch_session: BatchSession,
        mock: bool = False,
    ) -> None:
        super().__init__(
            name="judge_model",
            model=model,
            config=config,
            batch_session=batch_session,
            mock=mock,
        )


class ReviewFilteringModifier(Modifier):
    """Filters ratings deterministically and regenerates with the parent model."""

    name = "primary_llm"
    modifier_id = "deterministic-review-filter-v1"
    inapplicable_reason = "No reviews matched the requested filter condition."

    def __init__(
        self,
        config: dict[str, Any],
        summarizers: dict[str, Summarizer],
    ) -> None:
        self.config = config
        self.summarizers = summarizers

    def modifier_id_for(self, parent: Summary) -> str:
        """Return the identifier stamped on one regenerated attempt.

        Regeneration is performed by whichever model wrote the parent summary,
        so the parent's revision is part of what identifies this modifier.
        """

        return f"{self.modifier_id}:{parent.generation.revision}"

    def fingerprint(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> tuple[str, str]:
        del request
        specification = self.config["summary_conditions"][parent.condition]
        _, prompt_hash = load_style_prompt(specification["prompt"])

        return self.modifier_id_for(parent), prompt_hash

    def apply(
        self,
        parent: Summary,
        request: dict[str, Any],
    ) -> EditResult | None:
        target_rating = float(request["target_rating"])
        percentage = int(request["percentage"])
        eligible_indexes = [
            index
            for index, review in enumerate(parent.source_reviews)
            if review.rating == target_rating
        ]

        if not eligible_indexes:
            return None

        removal_count = math.ceil(len(eligible_indexes) * percentage / 100)
        # Derived from the product rather than the summary, so every summary
        # style of one product loses exactly the same reviews under the same
        # filter condition. Seeding per summary instead would let the style
        # effect and the removal draw vary together, which is not separable
        # afterwards.
        selection_seed = int(
            digest(
                stable_json(
                    [
                        self.config["seed"],
                        parent.product_id,
                        request["modification_condition"],
                    ]
                ),
                16,
            ),
            16,
        )
        random_generator = random.Random(selection_seed)
        removed_indexes = set(random_generator.sample(eligible_indexes, removal_count))
        retained_reviews = [
            review
            for index, review in enumerate(parent.source_reviews)
            if index not in removed_indexes
        ]

        if not retained_reviews:
            return None

        filtered_source_text = format_reviews(retained_reviews)
        filtered_product = Product(
            product_id=parent.product_id,
            source=parent.source,
            category=parent.category,
            review_count_bin=parent.review_count_bin,
            dataset_role=parent.dataset_role,
            reviews=retained_reviews,
            source_text=filtered_source_text,
            source_hash=digest(filtered_source_text),
        )
        specification = self.config["summary_conditions"][parent.condition]
        model = self.config["models"][specification["model"]]
        configured_identity = (
            model["provider"],
            model["model"],
            model_identity(model),
        )
        parent_identity = (
            parent.generation.provider,
            parent.generation.model,
            parent.generation.revision,
        )
        if configured_identity != parent_identity:
            raise ValueError(
                "Cannot regenerate with the parent generator because its "
                "provider/model/revision no longer matches experiment.yaml: "
                f"parent={parent_identity!r}, configured={configured_identity!r}."
            )
        regenerated = generate_text(
            product=filtered_product,
            condition=parent.condition,
            config=self.config,
            summarizers=self.summarizers,
        )
        if regenerated.prompt_hash != parent.generation.prompt_hash:
            raise ValueError(
                "Cannot regenerate with the parent style because its prompt has "
                "changed since the original summary was generated."
            )

        return EditResult(
            text=regenerated.text,
            edits=derive_edits(parent.text or "", regenerated.text),
            explanation=(
                f"Removed {removal_count} of {len(eligible_indexes)} "
                f"{target_rating:g}-star reviews, then regenerated with "
                f"{parent.generation.revision}."
            ),
            details={
                "review_filtering": {
                    "target_rating": target_rating,
                    "percentage": percentage,
                    "eligible_reviews": len(eligible_indexes),
                    "removed_reviews": removal_count,
                    "retained_reviews": len(retained_reviews),
                    "summary_condition": parent.condition,
                    "generator": {
                        "provider": model["provider"],
                        "model": model["model"],
                        "revision": model_identity(model),
                    },
                    "prompt_hash": regenerated.prompt_hash,
                    "request_hash": regenerated.request_hash,
                    "cached": regenerated.was_cached,
                }
            },
        )


def strip_json_fence(text: str) -> str:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    return stripped


def derive_edits(parent: str, modified: str) -> list[Edit]:
    parent_words = parent.split()
    modified_words = modified.split()
    matcher = SequenceMatcher(None, parent_words, modified_words)
    edits: list[Edit] = []

    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            continue

        edits.append(
            Edit(
                before=" ".join(parent_words[old_start:old_end]),
                after=" ".join(modified_words[new_start:new_end]),
            )
        )

    return edits


CONDITION_PREFIXES = {
    "factual_misrepresentation": "misrepresent",
    "omission": "omit",
    "quantifier_change": "quantifier",
}

# (response field, condition suffix, count, direction) for each of the five
# outputs a family's one-shot prompt returns. Direction is "mixed" from
# count2 on, since those slots combine an advantageous and a disadvantageous
# change (and, from count3, whatever direction was available for the extra
# one) rather than being purely one or the other.
MODIFICATION_SLOTS = (
    ("summary_count1_advantageous", "count1_advantageous", 1, "advantageous"),
    ("summary_count1_disadvantageous", "count1_disadvantageous", 1, "disadvantageous"),
    ("summary_count2", "count2", 2, "mixed"),
    ("summary_count3", "count3", 3, "mixed"),
    ("summary_count4", "count4", 4, "mixed"),
)


def modification_requests(
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected_families = (
        set(MODIFICATION_FAMILIES) if families is None else families
    )
    unsupported = selected_families - set(MODIFICATION_FAMILIES)
    if unsupported:
        raise ValueError(
            "Unsupported modification families: " + ", ".join(sorted(unsupported))
        )

    return [
        {
            "modification_family": family,
            "modification_condition": f"{CONDITION_PREFIXES[family]}_{suffix}",
            "severity": count,
            "direction": direction,
            "aspect_group": "categorized",
            "response_field": field,
        }
        for family in MODIFICATION_FAMILIES
        if family in selected_families
        for field, suffix, count, direction in MODIFICATION_SLOTS
    ]


def review_filter_requests(
    config: dict[str, Any],
    conditions: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    selected_conditions = (
        list(config["modification"]["review_filters"])
        if conditions is None
        else list(conditions)
    )

    for condition in selected_conditions:
        try:
            target_rating, percentage, direction = REVIEW_FILTER_DETAILS[condition]
        except KeyError as exception:
            raise ValueError(
                f"Unsupported review-filter condition: {condition}"
            ) from exception

        requests.append(
            {
                "modification_family": "review_filtering",
                "modification_condition": condition,
                "severity": percentage,
                "direction": direction,
                "aspect_group": "not_applicable",
                "target_rating": target_rating,
                "percentage": percentage,
            }
        )

    return requests


def resolve_modification_selection(
    config: dict[str, Any],
    selected: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Resolve repeatable CLI selectors into the exact execution plan."""

    if not selected:
        return {
            "requested": ["configured_default"],
            "families": list(MODIFICATION_FAMILIES),
            "review_filters": list(config["modification"]["review_filters"]),
        }

    selectors = list(dict.fromkeys(selected))
    unsupported = set(selectors) - set(MODIFICATION_CHOICES)
    if unsupported:
        raise ValueError(
            "Unsupported modification selectors: " + ", ".join(sorted(unsupported))
        )
    if "all" in selectors and len(selectors) != 1:
        raise ValueError("Use --modification all by itself.")
    if selectors == ["all"]:
        return {
            "requested": selectors,
            "families": list(MODIFICATION_FAMILIES),
            "review_filters": list(REVIEW_FILTER_DETAILS),
        }

    return {
        "requested": selectors,
        "families": [
            family for family in MODIFICATION_FAMILIES if family in selectors
        ],
        "review_filters": [
            condition for condition in REVIEW_FILTER_DETAILS if condition in selectors
        ],
    }


def validate(parent: str, result: EditResult) -> list[str]:
    """Reject only what cannot be an edit at all.

    Three outcomes are not modifications: the sentinel a modifier returns when
    the requested change does not apply to this summary, an empty response, and
    a response identical to the parent. Everything else is kept, including
    edits that touch more text than intended - `similarity`,
    `changed_word_percentage`, and `length_reduction` are recorded on every
    record, so how conservative an edit had to be is a question the analysis
    answers from the data rather than one this gate decides in advance and
    discards the evidence for.
    """

    text = result.text.strip()

    if text == INAPPLICABLE_MARKER:
        return ["The requested condition is not applicable."]

    if not text:
        return ["The modified summary is empty."]

    if text == parent.strip():
        return ["The modified summary is identical to its parent."]

    return []


def attempt_id_for(
    parent: Summary,
    modifier: Modifier,
    request: dict[str, Any],
) -> str:
    """Return the identifier of one cell of the modification design.

    One parent summary, one modifier role, one condition. The model and prompt
    behind that cell are deliberately absent: they are recorded on the row
    instead, so replacing a stale result overwrites the cell rather than
    creating a second row the analysis would count twice.
    """

    return stable_id(
        "att",
        parent.summary_id,
        modifier.name,
        request["modification_condition"],
        request["direction"],
        request["aspect_group"],
        request["severity"],
    )


def build_record(
    parent: Summary,
    modifier: Modifier,
    request: dict[str, Any],
    result: EditResult | None,
    error: str | None,
) -> Modification:
    valid = result is not None and error is None
    modifier_id, prompt_hash = modifier.fingerprint(parent, request)
    attempt_id = attempt_id_for(parent, modifier, request)

    similarity = None
    changed_word_percentage = None
    length_reduction = None

    if result is not None:
        comparison_text = parent.text or ""
        similarity = SequenceMatcher(None, comparison_text, result.text).ratio()
        changed_word_percentage = 100 * (
            1
            - SequenceMatcher(
                None,
                comparison_text.split(),
                result.text.split(),
            ).ratio()
        )
        length_reduction = (
            len(comparison_text.split()) - len(result.text.split())
        ) / max(1, len(comparison_text.split()))

    return Modification(
        attempt_id=attempt_id,
        modification_id=stable_id("mod", attempt_id) if valid else None,
        parent_summary_id=parent.summary_id,
        product_id=parent.product_id,
        dataset_role=parent.dataset_role,
        summary_condition=parent.condition,
        condition=request["modification_condition"],
        family=request["modification_family"],
        severity=request["severity"],
        direction=request["direction"],
        aspect_group=request["aspect_group"],
        generator=modifier.name,
        modifier=modifier_id,
        prompt_hash=prompt_hash,
        parent_text=parent.text or "",
        text=result.text if result else None,
        edits=result.edits if result else [],
        details=result.details if result else None,
        changed_word_percentage=changed_word_percentage,
        length_reduction=length_reduction,
        similarity=similarity,
        valid=valid,
        error=error,
        summary_is_judge_generated=parent.condition == "qwen_faithful",
        modification_is_judge_generated=modifier.name == "judge_model",
    )


def mock_modification_response(parent: str) -> str:
    """Recorded deterministic response for the no-cost integration test."""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", parent)
        if sentence.strip()
    ]
    count1_advantageous = parent.replace("bad", "good", 1)
    count1_disadvantageous = parent.replace("good", "bad", 1)
    count2 = count1_disadvantageous.replace("bad", "good", 1)
    count3 = " ".join(sentences[:-1]) or parent
    count4 = "INVALID"

    response = {
        "summary_count1_advantageous": count1_advantageous,
        "summary_count1_disadvantageous": count1_disadvantageous,
        "summary_count2": count2,
        "summary_count3": count3,
        "summary_count4": count4,
    }

    return json.dumps(response)


def stale_attempt_ids(
    summaries: list[Summary],
    records: list[Modification],
    modifier_requests: list[tuple[Modifier, dict[str, Any]]],
) -> set[str]:
    """Return stored attempts the current configuration would no longer produce.

    A stored row is stale when the model or the prompt named on it is not the
    one this configuration now uses - an edited modification prompt, a swapped
    modifier, a restyled parent summary. Without this check a resumed run
    skips exactly those rows as already done, and the file quietly mixes two
    generations of results.
    """

    stored = {record.attempt_id: record for record in records}
    stale: set[str] = set()

    for parent in summaries:
        for modifier, request in modifier_requests:
            attempt_id = attempt_id_for(parent, modifier, request)
            record = stored.get(attempt_id)

            if record is None:
                continue

            if (record.modifier, record.prompt_hash) != modifier.fingerprint(
                parent,
                request,
            ):
                stale.add(attempt_id)

    return stale


def is_retryable_modification_error(error: str | None) -> bool:
    """Return whether a saved failure should be attempted again."""

    return is_retryable_error(error)


def _store_record(
    output_path: str,
    record: Modification,
    records: list[Modification],
    record_positions: dict[str, int],
) -> None:
    """Append one attempt, superseding any earlier row for the same cell.

    Always appending keeps the cost of saving a record independent of how many
    are already stored; rewriting the file per replacement made a long run
    quadratic in its own output. Loading keeps the last row per `attempt_id`
    and rewrites the file once when it finds duplicates, so a superseded row
    never survives a second run.
    """

    append_jsonl(output_path, record)
    existing_position = record_positions.get(record.attempt_id)

    if existing_position is None:
        record_positions[record.attempt_id] = len(records)
        records.append(record)
        return

    records[existing_position] = record


def run(
    config: dict[str, Any],
    summaries_path: str | None = None,
    output: str | None = None,
    dry_run: bool = False,
    selected_modifications: list[str] | tuple[str, ...] | None = None,
    llm_modifiers: list[str] | tuple[str, ...] = LLM_MODIFIER_NAMES,
) -> list[Modification]:
    unsupported_modifiers = set(llm_modifiers) - set(LLM_MODIFIER_NAMES)
    if unsupported_modifiers:
        raise ValueError(
            "Unsupported llm_modifiers: "
            + ", ".join(sorted(unsupported_modifiers))
            + f". Expected a subset of {LLM_MODIFIER_NAMES}."
        )

    input_path = summaries_path or config["outputs"]["summaries"]
    output_path = output or config["outputs"]["modifications"]

    summaries = [
        summary
        for summary in read_jsonl(input_path, Summary)
        if summary.text is not None
    ]

    if not summaries:
        raise ValueError("No successful summaries were found.")

    selection = resolve_modification_selection(config, selected_modifications)
    requests = modification_requests(set(selection["families"]))
    filter_requests = review_filter_requests(
        config,
        selection["review_filters"],
    )
    primary_model = config["models"]["primary_modifier"]
    judge_model = config["models"]["qwen"]
    run_primary = requests and "primary_llm" in llm_modifiers
    run_judge = requests and "judge_model" in llm_modifiers

    if run_judge and not dry_run and "$" in judge_model.get("revision", ""):
        raise ValueError("Missing pinned QWEN_REVISION.")

    batch_session = BatchSession(config=config)
    modifiers: list[Modifier] = []
    if run_primary:
        modifiers.append(
            PrimaryLLMModifier(
                primary_model,
                config,
                batch_session=batch_session,
                mock=dry_run,
            )
        )
    if run_judge:
        modifiers.append(
            JudgeModelModifier(
                judge_model,
                config,
                batch_session=batch_session,
                mock=dry_run,
            )
        )
    review_filter_modifier: ReviewFilteringModifier | None = None
    if filter_requests:
        source_products = [
            Product(
                product_id=summary.product_id,
                source=summary.source,
                category=summary.category,
                review_count_bin=summary.review_count_bin,
                dataset_role=summary.dataset_role,
                reviews=summary.source_reviews,
                source_text=summary.source_text,
                source_hash=summary.source_hash,
            )
            for summary in summaries
        ]
        summarizers = preflight(
            config=config,
            products=source_products,
            batch_session=batch_session,
            mock=dry_run,
        )
        review_filter_modifier = ReviewFilteringModifier(
            config=config,
            summarizers=summarizers,
        )
    batch_session.refresh()

    loaded_records = read_jsonl(output_path, Modification)
    latest_records = {record.attempt_id: record for record in loaded_records}
    records = list(latest_records.values())

    if len(records) != len(loaded_records):
        write_jsonl(output_path, records)

    modifier_requests: list[tuple[Modifier, dict[str, Any]]] = [
        (modifier, request) for modifier in modifiers for request in requests
    ]

    if review_filter_modifier is not None:
        modifier_requests.extend(
            (review_filter_modifier, request) for request in filter_requests
        )

    # Counted over the attempts this run actually covers, so narrowing the run
    # with --modification or --modifier does not leave the bar reporting
    # against work it was never asked to do.
    expected_attempt_ids = {
        attempt_id_for(parent, modifier, request)
        for parent in summaries
        for modifier, request in modifier_requests
    }

    # A stale row is treated as never written: it stays in the file until the
    # replacement is appended after it, and loading keeps only the last row per
    # attempt, so nothing is lost if this run is interrupted first.
    stale_ids = stale_attempt_ids(
        summaries=summaries,
        records=records,
        modifier_requests=modifier_requests,
    )

    # A record can also be orphaned rather than stale: its condition was
    # renamed or dropped from the design entirely, so no request will ever
    # recompute it under its old attempt_id. Left alone it would linger in the
    # file forever and keep resurfacing whenever its parent is resampled by
    # some other, unrelated run. Checked against every condition the codebase
    # can currently produce - not just what this run's family/modifier
    # selection covers - so a record for a family this run didn't select is
    # never mistaken for orphaned, but a dead condition is caught regardless
    # of whose sample happened to include its parent.
    known_conditions = {
        request["modification_condition"] for request in modification_requests()
    } | {
        request["modification_condition"]
        for request in review_filter_requests(config, conditions=list(REVIEW_FILTER_DETAILS))
    }
    orphaned_ids = {
        record.attempt_id
        for record in records
        if record.condition not in known_conditions
    }

    if orphaned_ids:
        print(
            json.dumps(
                {
                    "event": "orphaned_modifications_removed",
                    "attempts": len(orphaned_ids),
                    "reason": (
                        "These attempts' conditions no longer exist in the "
                        "current modification design."
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    drop_ids = stale_ids | orphaned_ids

    if drop_ids:
        if stale_ids:
            print(
                json.dumps(
                    {
                        "event": "stale_modifications_will_be_redone",
                        "attempts": len(stale_ids),
                        "reason": (
                            "The modifier model or the prompt behind these "
                            "attempts changed since they were stored."
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        records = [record for record in records if record.attempt_id not in drop_ids]

        if orphaned_ids:
            # Orphaned rows have no replacement request that will re-append
            # them, unlike stale ones, so the file must be rewritten now or
            # they survive on disk under a dead attempt_id forever.
            write_jsonl(output_path, records)

    record_positions = {
        record.attempt_id: index for index, record in enumerate(records)
    }
    completed_ids = {
        record.attempt_id
        for record in records
        if not is_retryable_modification_error(record.error)
    }
    selected_conditions_by_group = {
        "primary_llm": {request["modification_condition"] for request in requests},
        "judge_model": {request["modification_condition"] for request in requests},
        "review_filtering": {
            request["modification_condition"] for request in filter_requests
        },
    }

    print(
        json.dumps(
            {
                "stage": "modification",
                "status": "collecting_requests",
                "summaries": len(summaries),
                "selected_modifications": selection["requested"],
                "attempts_per_modifier": len(requests),
                "review_filter_attempts_per_summary": len(filter_requests),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report_modification_status(
        config=config,
        summaries=summaries,
        records=records,
        batch_session=batch_session,
        modifiers=modifiers,
        moment="before_modification",
        requests=requests,
        filter_requests=filter_requests,
        selected_conditions_by_group=selected_conditions_by_group,
    )
    starting_record_count = len(records)
    counts = {"valid": 0, "invalid": 0, "retryable": 0, "deferred": 0}
    unexpected_failures = 0
    progress = stage_progress_bar(
        total=len(expected_attempt_ids),
        initial=len(expected_attempt_ids & completed_ids),
        description="modification",
        unit="attempt",
    )

    for parent_index, parent in enumerate(summaries, start=1):
        for modifier, request in modifier_requests:
            attempt_id = attempt_id_for(parent, modifier, request)

            if attempt_id in completed_ids:
                continue

            result = None
            error = None

            try:
                result = modifier.apply(parent, request)

                if result is None:
                    # The modifier had nothing to change. That is an expected
                    # outcome of the design, not a failure of the run.
                    error = modifier.inapplicable_reason
                else:
                    # A rejected edit is kept on the record. It is still
                    # invalid and still carries no `modification_id`, so
                    # nothing downstream scores it - but discarding what the
                    # model actually wrote leaves no way to tell a bad prompt
                    # from a strict validator without paying for the run again.
                    errors = validate(parent=parent.text or "", result=result)

                    if errors:
                        error = "; ".join(errors)

            except DeferredRequest:
                counts["deferred"] += 1
                progress.set_postfix(**counts, refresh=False)
                continue

            except Exception as exception:
                # Everything above is an expected outcome, so anything that
                # reaches here is the run going wrong rather than the model
                # declining: a provider error, an unparsable answer, a
                # regeneration whose parent no longer matches the config.
                error = str(exception)
                result = None
                unexpected_failures += 1

                if unexpected_failures <= 20 or unexpected_failures % 100 == 0:
                    tqdm.write(
                        f"  {request['modification_condition']} / "
                        f"{parent.summary_id}: {type(exception).__name__}: "
                        f"{exception}"
                    )

            record = build_record(
                parent=parent,
                modifier=modifier,
                request=request,
                result=result,
                error=error,
            )

            _store_record(
                output_path=output_path,
                record=record,
                records=records,
                record_positions=record_positions,
            )

            if is_retryable_modification_error(record.error):
                counts["retryable"] += 1
            else:
                completed_ids.add(attempt_id)
                progress.update(1)

                if record.valid:
                    counts["valid"] += 1
                else:
                    counts["invalid"] += 1

            progress.set_postfix(
                condition=request["modification_condition"][:22],
                **counts,
                refresh=False,
            )

        if parent_index % 10 == 0 or parent_index == len(summaries):
            report_modification_progress(
                summaries_processed=parent_index,
                summaries_total=len(summaries),
                records=records,
                starting_record_count=starting_record_count,
                selected_conditions_by_group=selected_conditions_by_group,
            )

    progress.close()
    tqdm.write(
        f"Modification pass finished: {counts['valid']:,} valid, "
        f"{counts['invalid']:,} invalid, {counts['retryable']:,} retryable, "
        f"{counts['deferred']:,} awaiting a provider batch; "
        f"{unexpected_failures:,} unexpected failures."
    )

    report_modification_status(
        config=config,
        summaries=summaries,
        records=records,
        batch_session=batch_session,
        modifiers=modifiers,
        moment="after_modification",
        requests=requests,
        filter_requests=filter_requests,
        selected_conditions_by_group=selected_conditions_by_group,
    )
    batch_session.finish()
    return records
