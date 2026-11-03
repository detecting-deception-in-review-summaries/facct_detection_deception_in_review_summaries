"""How every stage says what it is doing while it runs.

A stage's own module is about deciding what work exists and doing it; this one
is about making that work visible. Keeping them apart is what lets each stage
read as build work, do work, save, and it keeps the run loops free of the
formatting that otherwise dominates them.

Nothing here is imported by the stages' own logic, and nothing here imports a
stage, so the dependency runs one way only: stages call reporting, never the
reverse. Everything stage-specific arrives as an argument.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from .common import metric_score_ids, score_id_for
from .llms import BatchPending, BatchSession, is_retryable_error, model_identity
from .models import Modification, Product, Score, Summary

if TYPE_CHECKING:
    from .modification import Modifier
    from .scoring import Scorer


def stage_progress_bar(
    *,
    total: int,
    initial: int,
    description: str,
    unit: str,
) -> tqdm:
    """Build the progress bar every stage shows, with one set of options.

    The bar counts work that is actually finished, so a failed or deferred item
    leaves it in place rather than reporting progress the next run has to make
    again.
    """

    return tqdm(
        total=total,
        initial=initial,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


def format_score_summary(
    *,
    count: int,
    total: float,
    last: float,
    lowest: float,
    highest: float,
) -> dict[str, str]:
    """Summarize the scores a metric has produced so far, for the live bar.

    A metric that has quietly broken - a scorer returning the same label for
    everything, a rubric the model no longer follows - shows up here as a mean
    pinned to one end of the scale long before the run ends.
    """

    return {
        "last": f"{last:.3f}",
        "mean": f"{total / count:.3f}" if count else "-",
        "range": f"{lowest:.2f}-{highest:.2f}",
    }


def print_generation_plan(
    config: dict[str, Any],
    products: list[Product],
    conditions: dict[str, Any],
    expected_by_condition: dict[str, set[str]],
    completed_ids: set[str],
    output_path: str | Path,
) -> None:
    """Print a resume-aware plan of what this run will redo and what it skips.

    Printed before any provider call, so an interrupted run makes its intent
    plain before it spends anything.
    """

    expected = set().union(*expected_by_condition.values()) if conditions else set()
    saved = len(expected & completed_ids)

    tqdm.write("")
    tqdm.write("Generation plan")
    tqdm.write(
        f"  Products: {len(products):,}   Conditions: {len(conditions):,}   "
        f"Summaries: {len(expected):,}"
    )
    tqdm.write(f"  Output: {output_path} ({saved:,} already saved)")
    tqdm.write("  Conditions:")

    for condition, specification in conditions.items():
        condition_expected = expected_by_condition[condition]
        condition_saved = len(condition_expected & completed_ids)
        remaining = len(condition_expected) - condition_saved
        status = (
            "complete — will skip" if remaining == 0 else f"{remaining:,} remaining"
        )
        model = config["models"][specification["model"]]
        tqdm.write(
            f"    {condition:<26} {condition_saved:>4,}/{len(condition_expected):<4,}  "
            f"{status:<22} [{model['provider']}:{model['model']}]"
        )

    tqdm.write("")


def report_generation_status(
    config: dict[str, Any],
    products: list[Product],
    records: list[Summary],
    batch_session: BatchSession,
    moment: str,
    conditions: dict[str, Any] | None = None,
) -> None:
    """Print saved, failed, remaining, and provider-job counts by model.

    Counts cover the conditions this run generates, so a restricted run is not
    reported as permanently short of the conditions it was never asked for.
    """

    if conditions is None:
        conditions = config["summary_conditions"]

    latest_records = {
        record.summary_id: record
        for record in records
    }
    jobs = list(batch_session.state.get("jobs", {}).values())
    model_status: dict[str, dict[str, Any]] = {}

    model_keys = sorted(
        {specification["model"] for specification in conditions.values()}
    )

    for model_key in model_keys:
        condition_names = {
            condition
            for condition, specification in conditions.items()
            if specification["model"] == model_key
        }
        model_records = [
            record
            for record in latest_records.values()
            if record.condition in condition_names
        ]
        saved = sum(record.error is None for record in model_records)
        errors = sum(record.error is not None for record in model_records)
        expected = len(products) * len(condition_names)
        provider = config["models"][model_key]["provider"]
        requests_by_status: dict[str, int] = {}

        for job in jobs:
            if job["provider"] != provider:
                continue

            status = job["status"]
            requests_by_status[status] = (
                requests_by_status.get(status, 0)
                + len(job["request_ids"])
            )

        model_status[model_key] = {
            "provider": provider,
            "model": config["models"][model_key]["model"],
            "expected": expected,
            "saved": saved,
            "errors": errors,
            "remaining": expected - saved,
            "batch_requests_by_status": requests_by_status,
        }

    tqdm.write(
        json.dumps(
            {
                "event": "generation_status",
                "moment": moment,
                "models": model_status,
            },
            sort_keys=True,
        )
    )


def _record_group(record: Modification) -> str:
    """Return the user-facing execution group for one saved attempt."""

    if record.family == "review_filtering":
        return "review_filtering"

    return record.generator


def _batch_request_status(
    batch_session: BatchSession,
    request_ids: set[str],
) -> dict[str, int]:
    """Count only provider requests made by the selected modifier."""

    requests_by_status: dict[str, int] = {}
    state_request_ids: set[str] = set()

    for job in batch_session.state.get("jobs", {}).values():
        matching_ids = request_ids.intersection(job["request_ids"])

        if not matching_ids:
            continue

        status = job["status"]
        requests_by_status[status] = requests_by_status.get(status, 0) + len(
            matching_ids
        )
        state_request_ids.update(matching_ids)

    queued_request_ids = {
        cache_id
        for queued_group in batch_session.queued.values()
        for cache_id in queued_group
    }
    newly_queued_ids = request_ids.intersection(queued_request_ids) - state_request_ids

    if newly_queued_ids:
        requests_by_status["queued"] = len(newly_queued_ids)

    return requests_by_status


def _review_filter_provider_status(
    summaries: list[Summary],
    records: list[Modification],
    requests_per_summary: int,
) -> dict[str, dict[str, int]]:
    """Show which original summary providers perform review regeneration."""

    parent_providers = {
        summary.summary_id: summary.generation.provider for summary in summaries
    }
    expected_by_provider = Counter(summary.generation.provider for summary in summaries)
    recorded_by_provider: Counter[str] = Counter()
    valid_by_provider: Counter[str] = Counter()
    invalid_by_provider: Counter[str] = Counter()
    retryable_by_provider: Counter[str] = Counter()
    errors_by_provider: Counter[str] = Counter()

    for record in records:
        if record.family != "review_filtering":
            continue

        provider = parent_providers.get(record.parent_summary_id, "unknown")
        recorded_by_provider[provider] += 1

        if record.valid:
            valid_by_provider[provider] += 1
        elif is_retryable_error(record.error):
            retryable_by_provider[provider] += 1
        else:
            invalid_by_provider[provider] += 1

        if record.error is not None:
            errors_by_provider[provider] += 1

    providers = set(expected_by_provider) | set(recorded_by_provider)

    return {
        provider: {
            "expected": expected_by_provider[provider] * requests_per_summary,
            "recorded": recorded_by_provider[provider],
            "completed": (valid_by_provider[provider] + invalid_by_provider[provider]),
            "valid": valid_by_provider[provider],
            "invalid": invalid_by_provider[provider],
            "retryable_errors": retryable_by_provider[provider],
            "errors": errors_by_provider[provider],
            "remaining": (
                expected_by_provider[provider] * requests_per_summary
                - valid_by_provider[provider]
                - invalid_by_provider[provider]
            ),
        }
        for provider in sorted(providers)
    }


def report_modification_status(
    config: dict[str, Any],
    summaries: list[Summary],
    records: list[Modification],
    batch_session: BatchSession,
    modifiers: list["Modifier"],
    moment: str,
    requests: list[dict[str, Any]],
    filter_requests: list[dict[str, Any]],
    selected_conditions_by_group: dict[str, set[str]] | None = None,
) -> None:
    """Print attempt and model-request progress for each modification path."""

    latest_records = {record.attempt_id: record for record in records}
    records_by_group: dict[str, list[Modification]] = {
        "primary_llm": [],
        "judge_model": [],
        "review_filtering": [],
    }

    for record in latest_records.values():
        group = _record_group(record)

        if group in records_by_group:
            if (
                selected_conditions_by_group is not None
                and record.condition
                not in selected_conditions_by_group.get(group, set())
            ):
                continue
            records_by_group[group].append(record)

    parent_count = len(summaries)
    attempts = parent_count * len(requests)
    active_modifier_names = {modifier.name for modifier in modifiers}
    primary_enabled = "primary_llm" in active_modifier_names
    judge_enabled = "judge_model" in active_modifier_names
    expected_attempts = {
        "primary_llm": attempts if primary_enabled else 0,
        "judge_model": attempts if judge_enabled else 0,
        "review_filtering": parent_count * len(filter_requests),
    }
    llm_requests_per_parent = len(
        {request["modification_family"] for request in requests}
    )
    maximum_llm_requests = parent_count * llm_requests_per_parent
    llm_modifiers = {
        modifier.name: modifier
        for modifier in modifiers
        if hasattr(modifier, "request_ids")
    }
    primary_model = config["models"]["primary_modifier"]
    judge_model = config["models"]["qwen"]
    group_details = {
        "primary_llm": {
            "provider": primary_model["provider"],
            "model": primary_model["model"],
            "revision": model_identity(primary_model),
        },
        "judge_model": {
            "provider": judge_model["provider"],
            "model": judge_model["model"],
            "revision": model_identity(judge_model),
        },
        "review_filtering": {
            "provider": "parent_summary_provider",
            "model": "parent summary model",
            "revision": "parent summary revision",
        },
    }
    status: dict[str, dict[str, Any]] = {}

    for group, details in group_details.items():
        group_records = records_by_group[group]
        recorded = len(group_records)
        valid = sum(record.valid for record in group_records)
        retryable_errors = sum(
            is_retryable_error(record.error) for record in group_records
        )
        invalid = sum(
            not record.valid and not is_retryable_error(record.error)
            for record in group_records
        )
        completed = valid + invalid
        group_status = {
            **details,
            "expected_attempts": expected_attempts[group],
            "recorded_attempts": recorded,
            "completed_attempts": completed,
            "valid": valid,
            "invalid": invalid,
            "retryable_errors": retryable_errors,
            "errors": sum(record.error is not None for record in group_records),
            "remaining_attempts": expected_attempts[group] - completed,
        }

        if group in llm_modifiers:
            request_ids = llm_modifiers[group].request_ids
            group_status["maximum_llm_requests"] = maximum_llm_requests
            group_status["expected_llm_requests"] = len(request_ids)
            group_status["batch_requests_by_status"] = _batch_request_status(
                batch_session=batch_session,
                request_ids=request_ids,
            )


        if group == "review_filtering":
            group_status["providers"] = _review_filter_provider_status(
                summaries=summaries,
                records=group_records,
                requests_per_summary=len(filter_requests),
            )

        status[group] = group_status

    print(
        json.dumps(
            {
                "event": "modification_status",
                "moment": moment,
                "summaries": parent_count,
                "modifiers": status,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def report_modification_progress(
    summaries_processed: int,
    summaries_total: int,
    records: list[Modification],
    starting_record_count: int,
    selected_conditions_by_group: dict[str, set[str]] | None = None,
) -> None:
    """Print a compact heartbeat while modification requests are collected."""

    latest_records = {record.attempt_id: record for record in records}
    recorded_by_group: Counter[str] = Counter()
    for record in latest_records.values():
        group = _record_group(record)
        if (
            selected_conditions_by_group is not None
            and record.condition not in selected_conditions_by_group.get(group, set())
        ):
            continue
        recorded_by_group[group] += 1

    print(
        json.dumps(
            {
                "event": "modification_progress",
                "summaries_processed": summaries_processed,
                "summaries_total": summaries_total,
                "new_records_this_run": len(records) - starting_record_count,
                "recorded_attempts": {
                    group: recorded_by_group.get(group, 0)
                    for group in (
                        "primary_llm",
                        "judge_model",
                        "review_filtering",
                    )
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def print_scoring_plan(
    *,
    scorers: list["Scorer"],
    items: list[dict[str, Any]],
    completed_ids: set[str],
    state_path: Path,
    output_path: Path,
    publish_interval_seconds: float,
) -> None:
    """Print a concise resume-aware plan before any remote service starts."""

    faithful = sum(item["condition"] == "faithful" for item in items)
    modified = len(items) - faithful
    tqdm.write("")
    tqdm.write("Scoring plan")
    tqdm.write(
        f"  Inputs: {len(items):,} total ({faithful:,} faithful, {modified:,} modified)"
    )
    tqdm.write(f"  Checkpoint: {state_path} ({len(completed_ids):,} saved scores)")
    tqdm.write(
        f"  Live output: {output_path} (republished at most every "
        f"{publish_interval_seconds:.0f}s and at the end of each metric)"
    )
    tqdm.write("  Metrics:")
    for scorer in scorers:
        expected_ids = metric_score_ids(items, scorer)
        completed = len(expected_ids.intersection(completed_ids))
        remaining = len(expected_ids) - completed
        status = (
            "complete — will skip" if remaining == 0 else f"{remaining:,} remaining"
        )
        tqdm.write(
            f"    {scorer.name:<26} {completed:>5,}/{len(expected_ids):<5,}  "
            f"{status}  [{scorer.backend}]"
        )
    tqdm.write("")


def report_scoring_status(
    *,
    scorers: list["Scorer"],
    items: list[dict[str, Any]],
    records: list[Score],
    moment: str,
    starting_record_count: int = 0,
    current_metric: str | None = None,
    error: Exception | None = None,
    json_output: bool = True,
) -> None:
    """Print exact completion counts for every active metric revision."""

    saved_by_id = {record.score_id: record for record in records}
    faithful_expected = sum(item["condition"] == "faithful" for item in items)
    modified_expected = len(items) - faithful_expected
    metric_status: dict[str, dict[str, Any]] = {}
    all_expected_ids: set[str] = set()

    for scorer in scorers:
        faithful_ids = {
            score_id_for(item, scorer)
            for item in items
            if item["condition"] == "faithful"
        }
        modified_ids = {
            score_id_for(item, scorer)
            for item in items
            if item["condition"] != "faithful"
        }
        expected_ids = faithful_ids | modified_ids
        completed_ids = expected_ids.intersection(saved_by_id)
        all_expected_ids.update(expected_ids)
        metric_status[scorer.name] = {
            "backend": scorer.backend,
            "revision": scorer.revision,
            "batch_size": scorer.batch_size,
            "expected": len(expected_ids),
            "completed": len(completed_ids),
            "remaining": len(expected_ids - completed_ids),
            "faithful_summaries": {
                "expected": faithful_expected,
                "completed": len(faithful_ids.intersection(saved_by_id)),
            },
            "modified_summaries": {
                "expected": modified_expected,
                "completed": len(modified_ids.intersection(saved_by_id)),
            },
        }

    active_completed = len(all_expected_ids.intersection(saved_by_id))
    payload: dict[str, Any] = {
        "event": "scoring_status",
        "moment": moment,
        "current_metric": current_metric,
        "inputs": {
            "faithful_summaries": faithful_expected,
            "valid_modifications": modified_expected,
            "total": len(items),
        },
        "expected_scores": len(all_expected_ids),
        "completed_scores": active_completed,
        "remaining_scores": len(all_expected_ids) - active_completed,
        "new_scores_this_run": max(0, len(records) - starting_record_count),
        "saved_scores": len(saved_by_id),
        "stale_or_inactive_saved_scores": len(set(saved_by_id) - all_expected_ids),
        "metrics": metric_status,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["retryable_error"] = isinstance(
            error,
            BatchPending,
        ) or is_retryable_error(str(error))

    if not json_output:
        labels = {
            "before_scoring": "ready",
            "after_scoring_pass": "pass finished",
            "complete": "complete",
            "pending": "waiting for remote batch",
            "failed": "failed",
        }
        tqdm.write(
            f"Scoring {labels.get(moment, moment)}: "
            f"{active_completed:,}/{len(all_expected_ids):,} complete, "
            f"{len(all_expected_ids) - active_completed:,} remaining, "
            f"{payload['new_scores_this_run']:,} newly saved this run."
        )
        if payload["stale_or_inactive_saved_scores"]:
            tqdm.write(
                "  Saved scores outside the active metric revisions: "
                f"{payload['stale_or_inactive_saved_scores']:,} "
                "(retained, not counted as complete)."
            )
        if error is not None:
            tqdm.write(f"  {type(error).__name__}: {error}")
        return

    print(json.dumps(payload, sort_keys=True), flush=True)
