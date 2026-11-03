from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import generation, modification, reset, scoring
from .common import load_config, read_jsonl, record_manifest, write_json, write_jsonl
from .llms import BatchPending
from .models import Modification, Summary


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="experiment.yaml",
        help="Path to the centralized experiment configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic recorded responses instead of provider calls.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["batch", "direct"],
        help=(
            "Override llm_execution from the config. 'batch' uses asynchronous "
            "discounted/native batches; 'direct' calls each normal LLM endpoint "
            "synchronously."
        ),
    )

    return parser


def apply_llm_mode(
    config: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    """Apply the optional command-line execution-mode override."""

    if arguments.llm_mode is not None:
        config["llm_execution"] = arguments.llm_mode


def isolate_stage_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Give standalone --dry-run stages storage separate from live results."""

    isolated = copy.deepcopy(config)
    isolated["cache"] = ".cache/dry_run/responses"
    isolated["state"] = ".state/dry_run"
    isolated["outputs"] = {
        "summaries": "outputs/dry_run/summaries.jsonl",
        "modifications": "outputs/dry_run/modifications.jsonl",
        "scores": "outputs/dry_run/scores.parquet",
        "manual_scores": "outputs/dry_run/scores_manual.parquet",
        "manifest": "outputs/dry_run/run_manifest.json",
    }

    # The hand-annotated edits live outside the outputs block, but a dry run
    # must not read real annotations against mock summaries: their parents
    # would be missing and the stage would abort instead of no-opping.
    if "manual_modification" in isolated:
        isolated["manual_modification"] = {
            **isolated["manual_modification"],
            "workbook": "outputs/dry_run/manual_review.xlsx",
            "output": "outputs/dry_run/modifications_manual_edits.jsonl",
        }

    return isolated


def execute_stage(
    stage: str,
    arguments: argparse.Namespace,
    stage_function: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    config = load_config(arguments.config)
    apply_llm_mode(config, arguments)

    if arguments.dry_run:
        config = isolate_stage_dry_run(config)

    print(
        json.dumps(
            {
                "stage": stage,
                "status": "starting",
                "llm_mode": config.get("llm_execution", "batch"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        result = stage_function(config)
        record_manifest(
            config=config,
            stage=stage,
            status="complete",
            detail=result,
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "status": "complete",
                    **result,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    except BatchPending as pending:
        detail = {
            "message": str(pending),
            "jobs": pending.jobs,
        }
        record_manifest(
            config=config,
            stage=stage,
            status="pending",
            detail=detail,
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "status": "pending",
                    **detail,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    except Exception as exception:
        detail = {
            "error_type": type(exception).__name__,
            "error": str(exception),
        }
        record_manifest(
            config=config,
            stage=stage,
            status="failed",
            detail=detail,
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "status": "failed",
                    **detail,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2) from exception


def generate_main(argv: list[str] | None = None) -> None:
    parser = base_parser("Generate the configured summary conditions from the dataset.")
    parser.add_argument("--output")
    parser.add_argument(
        "--style",
        "--condition",
        dest="style",
        action="append",
        help=(
            "Generate only this summary condition. Repeat the argument to "
            "combine styles. Without it every configured condition is "
            "generated. Summaries already stored for other conditions are "
            "left untouched."
        ),
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="Print the configured summary conditions and exit.",
    )
    arguments = parser.parse_args(argv)

    if arguments.list_styles:
        config = load_config(arguments.config)
        for name, specification in config["summary_conditions"].items():
            print(f"{name}\t{specification['prompt']}")
        return

    def generate_stage(config: dict[str, Any]) -> dict[str, Any]:
        conditions = generation.resolve_summary_conditions(config, arguments.style)
        records = generation.run(
            config=config,
            output=arguments.output,
            dry_run=arguments.dry_run,
            selected_conditions=arguments.style,
        )
        selected_records = [
            record for record in records if record.condition in conditions
        ]
        return {
            "summaries": len(selected_records),
            "stored_summaries": len(records),
            "selected_conditions": sorted(conditions),
        }

    execute_stage("generation", arguments, generate_stage)


def reset_main(argv: list[str] | None = None) -> None:
    reset.main(argv)


def modify_main(argv: list[str] | None = None) -> None:
    parser = base_parser(
        "Run selected controlled modifications and review-filter regenerations."
    )
    parser.add_argument("--summaries")
    parser.add_argument("--output")
    parser.add_argument(
        "--modification",
        action="append",
        choices=modification.MODIFICATION_CHOICES,
        help=(
            "Run only this modification. Repeat the argument to combine choices. "
            "Without it, experiment.yaml controls the default set. Explicit "
            "'all' includes both experimental local pipelines."
        ),
    )
    parser.add_argument(
        "--modifier",
        action="append",
        choices=modification.LLM_MODIFIER_NAMES,
        help=(
            "Run only this LLM modifier. Repeat to include both. Without it "
            "both run, which sends one set of requests to the primary modifier "
            "and another to the Qwen judge in the same invocation. Review-"
            "filter regeneration is unaffected either way, since it is "
            "performed by whichever model wrote the parent summary."
        ),
    )
    parser.add_argument(
        "--list-modifications",
        action="store_true",
        help="Print the accepted --modification values and exit.",
    )
    arguments = parser.parse_args(argv)

    if arguments.list_modifications:
        print("\n".join(modification.MODIFICATION_CHOICES))
        return

    llm_modifiers = (
        list(dict.fromkeys(arguments.modifier))
        if arguments.modifier
        else list(modification.LLM_MODIFIER_NAMES)
    )

    def modification_stage(config: dict[str, Any]) -> dict[str, Any]:
        selection = modification.resolve_modification_selection(
            config,
            arguments.modification,
        )
        records = modification.run(
            config=config,
            summaries_path=arguments.summaries,
            output=arguments.output,
            dry_run=arguments.dry_run,
            selected_modifications=arguments.modification,
            llm_modifiers=llm_modifiers,
        )
        selected_conditions = {
            request["modification_condition"]
            for request in modification.modification_requests(
                set(selection["families"]),
            )
        }
        selected_conditions.update(selection["review_filters"])
        selected_records = [
            record for record in records if record.condition in selected_conditions
        ]
        return {
            "attempts": len(selected_records),
            "valid_modifications": sum(
                record.valid for record in selected_records
            ),
            "stored_attempts": len(records),
            "selected_modifications": selection["requested"],
            "llm_modifiers": llm_modifiers,
        }

    execute_stage("modification", arguments, modification_stage)


def modify_manual_main(argv: list[str] | None = None) -> None:
    parser = base_parser(
        "Run the three modification prompts (factual error, aspect deletion, "
        "quantifier) against the same base summaries used by the "
        "hand-annotated manual review, for direct comparison against the "
        "manual edits. Runs all five count/direction slots per family; only "
        "count1_advantageous/count1_disadvantageous have a manual-edit "
        "counterpart, count2-4 have none."
    )
    parser.add_argument(
        "--summaries",
        help="Override the source summaries.jsonl. Defaults to outputs.summaries.",
    )
    parser.add_argument(
        "--manual-edits",
        help=(
            "Override the manual-edits JSONL whose parent_summary_ids define "
            "the base-summary set. Defaults to manual_modification.output."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Override the destination JSONL. Defaults to "
            "automatic_modifications_of_manual_edits.jsonl next to the "
            "manual-edits file."
        ),
    )
    parser.add_argument(
        "--modifier",
        action="append",
        choices=modification.LLM_MODIFIER_NAMES,
        help=(
            "Run only this modifier. Repeat to include both, e.g. "
            "--modifier primary_llm --modifier judge_model. Without it, both "
            "run (the primary modifier and the Qwen judge), doubling the API "
            "calls and "
            "attempt records over a single modifier."
        ),
    )
    arguments = parser.parse_args(argv)
    llm_modifiers = (
        list(dict.fromkeys(arguments.modifier))
        if arguments.modifier
        else list(modification.LLM_MODIFIER_NAMES)
    )

    def modification_stage(config: dict[str, Any]) -> dict[str, Any]:
        manual_edits_path = arguments.manual_edits or config["manual_modification"]["output"]
        manual_records = read_jsonl(manual_edits_path, Modification)

        if not manual_records:
            raise ValueError(
                f"No manual modification records found in {manual_edits_path}."
            )

        base_summary_ids = {record.parent_summary_id for record in manual_records}

        summaries_path = arguments.summaries or config["outputs"]["summaries"]
        base_summaries = [
            summary
            for summary in read_jsonl(summaries_path, Summary)
            if summary.text is not None and summary.summary_id in base_summary_ids
        ]
        found_ids = {summary.summary_id for summary in base_summaries}
        missing_ids = base_summary_ids - found_ids

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} of {len(base_summary_ids)} manual-edit "
                f"parent summaries are missing (or textless) in "
                f"{summaries_path}: {sorted(missing_ids)[:5]}"
                + (" ..." if len(missing_ids) > 5 else "")
            )

        output_path = arguments.output or str(
            Path(manual_edits_path).parent / "automatic_modifications_of_manual_edits.jsonl"
        )

        with tempfile.TemporaryDirectory(
            prefix="deception-modify-manual-"
        ) as temporary_directory:
            # modification.run() reads summaries from a path, so the filtered
            # base set is materialized to a scratch file for the duration of
            # this call. Every downstream cache/attempt key is derived from
            # summary content, not this path, so the file does not need to
            # persist for a later resumed (pending-batch) invocation.
            base_summaries_path = Path(temporary_directory) / "manual_base_summaries.jsonl"
            write_jsonl(base_summaries_path, base_summaries)

            records = modification.run(
                config=config,
                summaries_path=str(base_summaries_path),
                output=output_path,
                dry_run=arguments.dry_run,
                selected_modifications=list(modification.MODIFICATION_FAMILIES),
                llm_modifiers=llm_modifiers,
            )

        return {
            "base_summaries": len(base_summaries),
            "llm_modifiers": llm_modifiers,
            "attempts": len(records),
            "valid_modifications": sum(record.valid for record in records),
            "output": output_path,
        }

    execute_stage("modify_manual", arguments, modification_stage)


def score_main(argv: list[str] | None = None) -> None:
    parser = base_parser("Score faithful and valid modified summaries.")
    parser.add_argument("--summaries")
    parser.add_argument("--modifications")
    parser.add_argument("--output")
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Score the hand-annotated modifications instead of the generated "
            "ones, writing to outputs.manual_scores with its own checkpoint so "
            "the main scores file is left untouched."
        ),
    )
    parser.add_argument(
        "--state-name",
        help=(
            "Resume checkpoint filename under the state directory. Defaults to "
            "scores_manual.jsonl under --manual, scores.jsonl otherwise. Set "
            "this to something unused whenever --modifications points at a "
            "third modification set, or its scores get published into the "
            "main (or manual) parquet, which is rebuilt from that checkpoint."
        ),
    )
    parser.add_argument(
        "--only-modified-parents",
        action="store_true",
        help=(
            "Narrow the faithful baseline to the parent summaries the given "
            "modifications actually descend from, instead of the whole panel. "
            "Always on under --manual; useful on its own for a side run "
            "against a custom --modifications file."
        ),
    )
    parser.add_argument(
        "--metric",
        action="extend",
        nargs="+",
        metavar="METRIC",
        help=(
            "Score only these metrics. Accepts several names at once and may "
            "also be repeated. Without it every metric enabled in "
            "experiment.yaml runs. Applies to regular and --manual runs alike; "
            "scores already saved for the other metrics are kept and "
            "republished."
        ),
    )
    parser.add_argument(
        "--judge",
        help=(
            "Score with this G-Eval judge instead of experiment.yaml's "
            "`geval_judge`. The judge is part of every score's identity, so a "
            "second judge's scores extend the same checkpoint without "
            "colliding, and plot_scores.ipynb compares them. The matching "
            "deployment must already exist: deploy it with "
            "DECEPTION_JUDGE=<name> uv run modal deploy modal_app.py."
        ),
    )
    parser.add_argument(
        "--list-judges",
        action="store_true",
        help="Print the configured G-Eval judges and exit.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the configured metrics and whether each is enabled, then exit.",
    )
    arguments = parser.parse_args(argv)

    if arguments.list_judges:
        config = load_config(arguments.config)
        active = config.get("geval_judge")

        for name, judge in (config.get("geval_judges") or {}).items():
            marker = "*" if name == active else " "
            app = judge.get("app_name", config["modal"]["app_name"])
            print(f"{marker} {name}\t{judge['model']}\t{app}")
        return

    if arguments.list_metrics:
        config = load_config(arguments.config)
        for name, specification in config["metrics"].items():
            state = "enabled" if specification["enabled"] else "disabled"
            print(f"{name}\t{state}")
        return

    def scoring_stage(config: dict[str, Any]) -> dict[str, Any]:
        manual = config.get("manual_modification", {})

        if arguments.judge:
            config["geval_judge_override"] = arguments.judge
        records = scoring.run(
            config=config,
            summaries_path=arguments.summaries,
            modifications_path=(
                arguments.modifications
                or (manual["output"] if arguments.manual else None)
            ),
            output=(
                arguments.output
                or (config["outputs"]["manual_scores"] if arguments.manual else None)
            ),
            dry_run=arguments.dry_run,
            state_name=(
                arguments.state_name
                or ("scores_manual.jsonl" if arguments.manual else "scores.jsonl")
            ),
            # A manual run scores one hand-picked sample, so the faithful
            # baseline is narrowed to the parents of those edits.
            only_modified_parents=arguments.only_modified_parents or arguments.manual,
            selected_metrics=arguments.metric,
        )
        return {
            "scores": len(records),
            "manual": arguments.manual,
            "selected_metrics": sorted(
                scoring.resolve_metrics(config, arguments.metric)
            ),
        }

    execute_stage("scoring", arguments, scoring_stage)


def dry_run_main(argv: list[str] | None = None) -> None:
    parser = base_parser(
        "Run generation, modification, and scoring with recorded responses."
    )
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    apply_llm_mode(config, arguments)

    try:
        with tempfile.TemporaryDirectory(
            prefix="deception-dry-run-"
        ) as temporary_directory:
            root = Path(temporary_directory)
            local_config = copy.deepcopy(config)
            local_config["cache"] = str(root / "cache")
            local_config["state"] = str(root / "state")

            summaries_path = root / "summaries.jsonl"
            modifications_path = root / "modifications.jsonl"
            scores_path = root / "scores.parquet"

            summaries = generation.run(
                config=local_config,
                output=str(summaries_path),
                dry_run=True,
            )
            modifications = modification.run(
                config=local_config,
                summaries_path=str(summaries_path),
                output=str(modifications_path),
                dry_run=True,
            )
            scores = scoring.run(
                config=local_config,
                summaries_path=str(summaries_path),
                modifications_path=str(modifications_path),
                output=str(scores_path),
                dry_run=True,
            )

        observed = {
            "products": 4,
            "summaries": len(summaries),
            "attempts": len(modifications),
            "valid_modifications": sum(record.valid for record in modifications),
            "scores": len(scores),
        }
        write_json("outputs/dry_run_counts.json", observed)
        record_manifest(
            config=config,
            stage="dry_run",
            status="complete",
            detail=observed,
        )
        print(json.dumps(observed, sort_keys=True))

    except Exception as exception:
        record_manifest(
            config=config,
            stage="dry_run",
            status="failed",
            detail={"error": str(exception)},
        )
        raise
