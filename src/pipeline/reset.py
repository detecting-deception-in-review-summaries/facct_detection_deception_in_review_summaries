"""Remove generated records so a stage can be re-run from a clean slate.

Nothing is unlinked. Everything this tool removes is first moved into a
timestamped backup directory, and a run only touches the working tree once
`--yes` is given, so the default invocation is a preview.

Three details make a naive `rm` unsafe, and this module encodes all three:

* Scores live in `.state/scores.jsonl` (`scores_manual.jsonl` under `--manual`).
  The parquet under `outputs/` is a published view rebuilt from that state, so
  deleting only the parquet resurrects every score on the next run.
* Scoring refuses to start when a valid modification has no parent summary, so
  removing summaries without their dependent modifications breaks the next
  scoring run rather than cleaning it.
* A summary's cached provider response outlives its record. Left in place the
  next run reproduces the identical text from cache, which is usually what you
  want; `--cache` is how you ask for genuinely fresh sampling.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import load_config, read_jsonl, write_jsonl
from .models import Modification, Score, Summary

STAGES = ("summaries", "modifications", "scores")

# Removing a stage invalidates everything derived from it.
DOWNSTREAM_STAGES: dict[str, tuple[str, ...]] = {
    "summaries": ("modifications", "scores"),
    "modifications": ("scores",),
    "scores": (),
}


def scores_state_path(config: dict[str, Any], manual: bool = False) -> Path:
    """Return the durable score store, of which the parquet is only a view."""

    name = "scores_manual.jsonl" if manual else "scores.jsonl"

    return Path(config["state"]) / name


def stage_paths(
    config: dict[str, Any],
    stage: str,
    manual: bool = False,
) -> list[Path]:
    """Return every file that holds this stage's records.

    `manual` selects the hand-annotated counterpart of a stage: the workbook's
    modification records and the scores `score --manual` produces. Summaries
    have no manual counterpart. Only paths named by the configuration are ever
    returned, so files kept alongside them by hand are never candidates.
    """

    if stage == "scores":
        output = config["outputs"]["manual_scores" if manual else "scores"]

        return [scores_state_path(config, manual), Path(output)]

    if stage == "modifications" and manual:
        return [Path(config["manual_modification"]["output"])]

    return [Path(config["outputs"][stage])]


def backup_directory(root: str | Path, moment: datetime | None = None) -> Path:
    """Return a fresh timestamped directory to hold what a run removes."""

    stamp = (moment or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")

    return Path(root) / stamp


def _load(config: dict[str, Any], manual: bool = False) -> dict[str, list[Any]]:
    # Summaries are shared: hand annotations are edits of the same summaries the
    # generated modifications descend from, so there is no manual variant.
    return {
        "summaries": read_jsonl(Path(config["outputs"]["summaries"]), Summary),
        "modifications": read_jsonl(
            stage_paths(config, "modifications", manual)[0],
            Modification,
        ),
        "scores": read_jsonl(scores_state_path(config, manual), Score),
    }


def plan_removal(
    config: dict[str, Any],
    stages: list[str] | tuple[str, ...],
    conditions: list[str] | None = None,
    cascade: bool = False,
    manual: bool = False,
) -> dict[str, Any]:
    """Decide exactly which records each stage loses, and refuse to orphan.

    Selecting no conditions clears the whole stage. Selecting conditions
    matches each stage on its own condition field: `Summary.condition` for
    summaries, `Modification.condition` for modifications, and
    `Score.modification_condition` for scores.
    """

    unsupported = sorted(set(stages) - set(STAGES))

    if unsupported:
        raise ValueError(
            "Unknown stages: "
            + ", ".join(unsupported)
            + ". Choose from: "
            + ", ".join(STAGES)
        )

    if not stages:
        raise ValueError("Select at least one stage to remove.")

    if manual and "summaries" in stages:
        raise ValueError(
            "Summaries have no manual counterpart; drop --manual or drop "
            "--stage summaries."
        )

    selected_conditions = set(conditions or ())
    loaded = _load(config, manual)

    removed: dict[str, list[Any]] = {stage: [] for stage in STAGES}

    if "summaries" in stages:
        removed["summaries"] = [
            summary
            for summary in loaded["summaries"]
            if not selected_conditions or summary.condition in selected_conditions
        ]

    if "modifications" in stages:
        removed["modifications"] = [
            record
            for record in loaded["modifications"]
            if not selected_conditions or record.condition in selected_conditions
        ]

    if "scores" in stages:
        removed["scores"] = [
            score
            for score in loaded["scores"]
            if not selected_conditions
            or score.modification_condition in selected_conditions
        ]

    # Records the selection did not name but that cannot survive it.
    orphaned_summary_ids = {summary.summary_id for summary in removed["summaries"]}
    orphaned_modifications = [
        record
        for record in loaded["modifications"]
        if record.parent_summary_id in orphaned_summary_ids
        and record not in removed["modifications"]
    ]
    doomed_modification_ids = orphaned_summary_ids | {
        record.modification_id
        for record in [*removed["modifications"], *orphaned_modifications]
        if record.modification_id is not None
    }
    orphaned_scores = [
        score
        for score in loaded["scores"]
        if (
            score.parent_summary_id in orphaned_summary_ids
            or score.record_id in doomed_modification_ids
        )
        and score not in removed["scores"]
    ]

    orphans = {
        "modifications": len(orphaned_modifications),
        "scores": len(orphaned_scores),
    }

    if cascade:
        removed["modifications"] = [
            *removed["modifications"],
            *orphaned_modifications,
        ]
        removed["scores"] = [*removed["scores"], *orphaned_scores]
        orphans = {"modifications": 0, "scores": 0}

    elif any(orphans.values()):
        detail = ", ".join(
            f"{count} {stage}" for stage, count in orphans.items() if count
        )
        raise ValueError(
            f"Removing this selection would orphan {detail}. Scoring refuses to "
            "run when a valid modification has no parent summary, so re-run with "
            "--cascade to remove the dependent records too."
        )

    return {
        "stages": list(stages),
        "conditions": sorted(selected_conditions),
        "cascade": cascade,
        "manual": manual,
        "removed": removed,
        "kept": {
            stage: [
                record for record in loaded[stage] if record not in removed[stage]
            ]
            for stage in STAGES
        },
        "counts": {
            stage: {
                "removed": len(removed[stage]),
                "kept": len(loaded[stage]) - len(removed[stage]),
            }
            for stage in STAGES
        },
    }


def cached_response_paths(
    config: dict[str, Any],
    summaries: list[Summary],
) -> list[Path]:
    """Return the cached provider responses belonging to these summaries.

    A summary's `generation.request_hash` is the same digest the response cache
    files itself under, so generation records map back to their cache entries.
    Modification records store no request hash, so their cached responses
    cannot be identified this way and are always left in place.
    """

    directory = Path(config["cache"])

    return [
        path
        for path in (
            directory / f"{summary.generation.request_hash}.json"
            for summary in summaries
        )
        if path.is_file()
    ]


def apply_removal(
    config: dict[str, Any],
    plan: dict[str, Any],
    backup: Path,
    drop_cache: bool = False,
    reset_batch_jobs: bool = False,
) -> dict[str, Any]:
    """Move the planned records into `backup` and rewrite what survives."""

    backup.mkdir(parents=True, exist_ok=True)
    moved: dict[str, Any] = {}

    for stage in STAGES:
        if not plan["removed"][stage]:
            continue

        write_jsonl(backup / f"{stage}.jsonl", plan["removed"][stage])

        if plan["counts"][stage]["kept"]:
            for path in stage_paths(config, stage, plan["manual"]):
                if path.suffix == ".jsonl" and path.is_file():
                    write_jsonl(path, plan["kept"][stage])
        else:
            for path in stage_paths(config, stage, plan["manual"]):
                if path.is_file():
                    path.replace(backup / path.name)

        moved[stage] = len(plan["removed"][stage])

    if plan["counts"]["scores"]["removed"] and plan["counts"]["scores"]["kept"]:
        # The parquet is a view; republish it so it matches the trimmed state.
        from .scoring import write_score_output

        write_score_output(
            plan["kept"]["scores"],
            stage_paths(config, "scores", plan["manual"])[1],
        )

    if drop_cache and plan["removed"]["summaries"]:
        cache_backup = backup / "responses"
        cache_backup.mkdir(parents=True, exist_ok=True)
        cache_paths = cached_response_paths(config, plan["removed"]["summaries"])

        for path in cache_paths:
            path.replace(cache_backup / path.name)

        moved["cached_responses"] = len(cache_paths)

    if reset_batch_jobs:
        jobs_path = Path(config["state"]) / "batch_jobs.json"

        if jobs_path.is_file():
            jobs_path.replace(backup / jobs_path.name)
            moved["batch_jobs"] = True

    return moved


def report(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove generated summaries, modifications, or scores so a stage "
            "can be re-run cleanly. Previews by default; pass --yes to apply."
        )
    )
    parser.add_argument(
        "--config",
        default="experiment.yaml",
        help="Path to the centralized experiment configuration.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        required=True,
        help="Stage to clear. Repeat the argument to clear several.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help=(
            "Remove only records of this condition, matched against each "
            "stage's own condition field. Repeatable. Without it the whole "
            "stage is cleared."
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Act on the hand-annotated artifacts instead: the workbook's "
            "modification records and the scores written by `score --manual`. "
            "Not valid with --stage summaries, which has no manual variant."
        ),
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Also remove records that the selection would otherwise orphan.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Also move the cached provider responses for the removed "
            "summaries, so the next run samples afresh instead of "
            "reproducing the cached text. Summaries only."
        ),
    )
    parser.add_argument(
        "--batch-jobs",
        action="store_true",
        help="Also clear .state/batch_jobs.json, for a run interrupted mid-batch.",
    )
    parser.add_argument(
        "--backup-dir",
        default="outputs/reset_backups",
        help="Root directory for the timestamped backup of what is removed.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the plan. Without it the command only prints what it would do.",
    )
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)

    try:
        plan = plan_removal(
            config=config,
            stages=arguments.stage,
            conditions=arguments.condition,
            cascade=arguments.cascade,
            manual=arguments.manual,
        )
    except ValueError as error:
        report("reset_refused", reason=str(error))
        raise SystemExit(2) from error

    if not arguments.yes:
        report(
            "reset_preview",
            stages=plan["stages"],
            conditions=plan["conditions"],
            cascade=plan["cascade"],
            manual=plan["manual"],
            counts=plan["counts"],
            hint="Re-run with --yes to apply.",
        )
        return

    backup = backup_directory(arguments.backup_dir)
    moved = apply_removal(
        config=config,
        plan=plan,
        backup=backup,
        drop_cache=arguments.cache,
        reset_batch_jobs=arguments.batch_jobs,
    )
    report(
        "reset_applied",
        stages=plan["stages"],
        conditions=plan["conditions"],
        cascade=plan["cascade"],
        counts=plan["counts"],
        moved=moved,
        backup=str(backup),
    )


if __name__ == "__main__":
    main()
