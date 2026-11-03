"""Shared data preparation for the score-analysis notebooks.

The scoring output deliberately contains only identifiers and scoring metadata.
This module joins it to summaries and valid modifications so both notebooks use
the experiment's current taxonomy instead of maintaining independent mappings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Copied from the pipeline (`pipeline/common.py` and `pipeline/llms.py`), so the
# analysis does not depend on it.


def load_config(path: str = "experiment.yaml") -> dict[str, Any]:
    """The experiment configuration, as the pipeline read it."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def active_geval_judge(
    config: dict[str, Any],
    judge: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the name and model entry of the judge G-Eval should ask.

    Every judge inherits `geval_judge_defaults` and overrides only what makes
    it a different judge. A configuration with no `geval_judges` block falls
    back to the generation Qwen.
    """

    judges = config.get("geval_judges")

    if not judges:
        return "qwen", config["models"]["qwen"]

    name = judge or config.get("geval_judge")

    if name not in judges:
        raise ValueError(
            f"Unknown G-Eval judge: {name!r}. experiment.yaml defines "
            + ", ".join(sorted(judges))
            + "."
        )

    return name, {**config.get("geval_judge_defaults", {}), **judges[name]}


METRIC_LABELS = {
    "geval_consistency": "G-Eval consistency",
    "geval_relevance": "G-Eval relevance",
    "summac_zs": "SummaC-ZS",
    "summac_conv": "SummaC-Conv",
    "qafacteval": "QAFactEval",
    "supert": "SUPERT",
    "alignscore": "AlignScore",
    "factgraph": "FactGraph",
    "minicheck_7b": "MiniCheck-7B",
}

FAMILY_LABELS = {
    "faithful": "Styles",
    "factual_error": "Factual Errors",
    "aspect_deletion": "Aspect Deletions",
    "quantifier_swap": "Quantifier Swaps",
    "review_filtering": "Review Filtering",
}

FAMILY_COLORS = {
    "faithful": "#b0b0b0",
    "factual_error": "#e74c3c",
    "aspect_deletion": "#3498db",
    "quantifier_swap": "#27ae60",
    "review_filtering": "#f39c12",
}

CONDITION_LABELS = {
    "faithful": "Faithful",
    **{
        f"{prefix}_count{count}{suffix}": f"{count} Aspect{'s' if count > 1 else ''}"
        for prefix in ("factual_error", "aspect_deletion", "quantifier_swap")
        for count, suffix in (
            (1, "_advantageous"),
            (1, "_disadvantageous"),
            (2, ""),
            (3, ""),
            (4, ""),
        )
    },
    "remove_50pct_one_star": "Remove 50% 1-Star",
    "remove_100pct_one_star": "Remove 100% 1-Star",
    "remove_50pct_five_star": "Remove 50% 5-Star",
    "remove_100pct_five_star": "Remove 100% 5-Star",
}

SUMMARY_CONDITION_LABELS = {
    "claude_faithful": "Claude Faithful",
    "gemini_faithful": "Gemini Faithful",
    "openai_short_summary": "OpenAI Short Summary",
    "openai_long_summary": "OpenAI Long Summary",
    "openai_amazon_style": "OpenAI Amazon Style",
    "openai_gmaps_style": "OpenAI Google Maps Style",
    "openai_bullet_points": "OpenAI Bullet Points",
    "openai_positive_and_negative": "OpenAI Positive and Negative",
    "openai_majority_opinion": "OpenAI Majority Opinion",
    "openai_single_sentence": "OpenAI Single Sentence",
    "qwen_faithful": "Qwen Faithful",
}

def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing the experiment configuration."""

    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "experiment.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find experiment.yaml from {current}. Start Jupyter inside "
        "the project or set DECEPTION_PROJECT_ROOT."
    )


def _read_records(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summaries, modifications) from `summaries.jsonl`.

    The two kinds share one file but not their fields, so each is normalised on
    its own: one table would leave every summary-only field blank on the
    modifications and lose its type.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    summaries = [r for r in records if r["record_id"] == r["parent_summary_id"]]
    modifications = [r for r in records if r["record_id"] != r["parent_summary_id"]]
    return pd.json_normalize(summaries), pd.json_normalize(modifications)


def _display_label(row: pd.Series) -> str:
    if row["family"] == "faithful":
        return SUMMARY_CONDITION_LABELS.get(
            row["summary_condition"],
            str(row["summary_condition"]).replace("_", " ").title(),
        )

    if row["family"] == "review_filtering":
        return CONDITION_LABELS.get(row["condition"], row["condition"])

    pieces = [CONDITION_LABELS.get(row["condition"], row["condition"])]
    qualifiers = []
    if row["direction"] not in {"", "not_applicable"}:
        qualifiers.append(str(row["direction"]).capitalize())
    if row["generator"] == "judge_model":
        qualifiers.append("Judge")
    if row["generator"] == "manual_annotation":
        qualifiers.append("Manual")
    if qualifiers:
        pieces.append(f"({', '.join(qualifiers)})")
    return " ".join(pieces)


def load_score_analysis(
    *,
    project_root: str | Path | None = None,
    score_sets: tuple[str, ...] = ("main", "manual"),
    geval_judge: str | None = None,
    keep_all_judges: bool = False,
) -> pd.DataFrame:
    """Load one score per record/metric and attach experiment metadata.

    Reads `data/scores.parquet` and `data/summaries.jsonl`. `score_sets` picks
    which scores to load: "main" (the LLM-generated panel), "manual" (the
    hand-written edits) and "llm_matched" (LLM modifications of the products
    the hand-written edits cover, scored for that comparison).

    G-Eval scores are kept for one judge only: `geval_judge`, or by default the
    `geval_judge` of experiment.yaml. `keep_all_judges` keeps every judge
    instead; use it when the judges are the comparison, and group by `judge`.
    """

    root = find_project_root(project_root)
    scores = pd.read_parquet(root / "data/scores.parquet")
    scores = scores[scores["score_set"].isin(score_sets)].reset_index(drop=True)

    # `result.judge` names the G-Eval judge (qwen36, gemma4, olmo31); the other
    # metrics run without one and leave it empty.
    scores["judge"] = scores["result.judge"]
    if keep_all_judges:
        geval_judge = None
    elif geval_judge is None:
        geval_judge = load_config(str(root / "experiment.yaml"))["geval_judge"]

    if geval_judge is not None:
        from_geval = scores["result.metric"].astype(str).str.startswith("geval_")
        scores = scores.loc[~from_geval | scores["judge"].eq(geval_judge)]

    # One score per record and metric (and judge, when every judge is kept).
    # The "main" and "manual" sets were written as one run, so a record that
    # both hold resolves to its later score, the manual one.
    scores = scores.drop_duplicates(
        subset=["record_id", "result.metric"]
        + (["judge"] if keep_all_judges else []),
        keep="last",
    ).copy()

    summaries, modifications = _read_records(root / "data/summaries.jsonl")
    summary_metadata = summaries[
        [
            "record_id",
            "source",
            "category",
            "review_count_bin",
            "is_primary_panel",
            "is_model_ablation",
        ]
    ].rename(columns={"record_id": "parent_summary_id"})
    modification_metadata = modifications[
        [
            "record_id",
            "family",
            "condition",
            "severity",
            "direction",
            "generator",
            "modifier",
            "changed_word_percentage",
            "length_reduction",
            "similarity",
        ]
    ]

    frame = scores.merge(
        summary_metadata,
        on="parent_summary_id",
        how="left",
        validate="many_to_one",
    ).merge(
        modification_metadata,
        on="record_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_modification"),
    )

    faithful = frame["record_id"].eq(frame["parent_summary_id"])
    frame.loc[faithful, "family"] = "faithful"
    frame.loc[faithful, "condition"] = "faithful"
    frame.loc[faithful, "direction"] = "not_applicable"
    frame.loc[faithful, "severity"] = 0

    unmatched = frame["family"].isna()
    if unmatched.any():
        bad_ids = frame.loc[unmatched, "record_id"].astype(str).head(5).tolist()
        raise ValueError(
            "Some scored records match neither a faithful summary nor a "
            f"modification (examples: {bad_ids})."
        )

    frame["metric"] = frame["result.metric"].astype(str)
    frame["metric_label"] = frame["metric"].map(METRIC_LABELS).fillna(frame["metric"])
    frame["raw_score"] = pd.to_numeric(frame["result.score"], errors="coerce")
    higher_is_faithful = frame["result.higher_is_faithful"].astype(bool)
    frame["anomaly_score"] = frame["raw_score"].where(
        ~higher_is_faithful, -frame["raw_score"]
    )
    frame["family_label"] = frame["family"].map(FAMILY_LABELS).fillna(frame["family"])
    frame["condition_label"] = (
        frame["condition"].map(CONDITION_LABELS).fillna(frame["condition"])
    )
    frame["display_group"] = frame.apply(_display_label, axis=1)
    # Two families can share a condition label - `factual_error_count1_advantageous`
    # and `aspect_deletion_count1_advantageous` are both "1 Aspect" - so the visible label is
    # not a unique key. The family qualifies the key; the label itself stays
    # unchanged, because the plots already separate families into their own bands.
    frame["display_key"] = frame["family"].astype(str) + "|" + frame["display_group"]

    frame = frame.reset_index(drop=True)
    frame.attrs["geval_judge"] = geval_judge

    return frame


def metric_order(frame: pd.DataFrame) -> list[str]:
    configured = [name for name in METRIC_LABELS if name in set(frame["metric"])]
    extras = sorted(set(frame["metric"]).difference(configured))
    return configured + extras
