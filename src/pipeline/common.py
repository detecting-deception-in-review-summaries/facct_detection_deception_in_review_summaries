from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def digest(value: Any, n: int | None = None) -> str:
    serialized_value = value if isinstance(value, bytes) else str(value).encode()
    result = hashlib.sha256(serialized_value).hexdigest()

    if n is None:
        return result

    return result[:n]


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{digest(stable_json(parts), 20)}"


def load_config(path: str = "experiment.yaml") -> dict[str, Any]:
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)

        if isinstance(value, list):
            return [expand(item) for item in value]

        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}

        return value

    return expand(raw)


def active_geval_judge(
    config: dict[str, Any],
    judge: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the name and model entry of the judge G-Eval should ask.

    Every judge inherits `geval_judge_defaults` and overrides only what makes
    it a different judge - the weights, and where it is deployed. That is what
    keeps two judges comparable: nothing that could move a score, from the
    token cap to the pinned library versions, is stated per judge, so it cannot
    differ between them by accident.

    A configuration with no `geval_judges` block falls back to the generation
    Qwen, which is the model every score written before the block existed was
    produced by.
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


def geval_judge_app_name(config: dict[str, Any], judge: dict[str, Any]) -> str:
    """Return the Modal app serving one judge.

    Judges without their own `app_name` share the main deployment, so switching
    the active judge and redeploying replaces it. Giving a judge its own name
    deploys it alongside instead, which is what lets two judges serve at once.
    """

    return judge.get("app_name") or config["modal"]["app_name"]


def estimate_prompt_tokens(prompt_characters: int, max_tokens: int) -> int:
    """Estimate the tokens one request occupies, prompt plus reserved output.

    Three characters per token is deliberately pessimistic; English usually
    tokenizes nearer four, so the guard trips before a provider would.
    """

    return prompt_characters // 3 + max_tokens


def check_context_budget(
    model_key: str,
    model: dict[str, Any],
    context_budget: int,
    stage: str,
) -> None:
    """Fail when a stage budget exceeds what the model is configured to serve.

    A model's `context_tokens` is the window this experiment allows it, and for
    the Modal-hosted Qwen it is also the vLLM `--max-model-len` the server
    starts with, so a stage may never ask for more than the model declares.
    """

    model_context = model["context_tokens"]

    if context_budget > model_context:
        raise ValueError(
            f"The {stage} context budget is {context_budget} tokens, but "
            f"{model_key} declares only {model_context}."
        )


def check_prompt_budget(
    prompt_characters: int,
    decoding: dict[str, Any],
    subject: str,
    stage: str,
) -> None:
    """Fail when one prompt would not fit inside its stage's context budget."""

    context_budget = decoding["context_tokens"]
    estimated_tokens = estimate_prompt_tokens(
        prompt_characters,
        decoding["max_tokens"],
    )

    if estimated_tokens > context_budget:
        raise ValueError(
            f"{subject} needs approximately {estimated_tokens} tokens, but the "
            f"{stage} context budget is {context_budget}."
        )


def read_jsonl(path: str | Path, model: type[T]) -> list[T]:
    path = Path(path)

    if not path.exists():
        return []

    adapter = TypeAdapter(model)

    with path.open(encoding="utf-8") as file:
        return [adapter.validate_json(line) for line in file if line.strip()]


def append_jsonl(path: str | Path, record: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")
        file.flush()


def write_jsonl(
    path: str | Path,
    records: list[BaseModel],
) -> None:
    """Replace a JSONL file with one serialized line per record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(record.model_dump_json() + "\n" for record in records)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    serialized_value = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
    )
    path.write_text(
        serialized_value + "\n",
        encoding="utf-8",
    )


def record_manifest(
    config: dict,
    stage: str,
    status: str,
    detail: dict | None = None,
) -> None:
    path = Path(config["outputs"]["manifest"])

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"runs": []}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = "UNCOMMITTED"

    packages: dict[str, str] = {}

    for name in (
        "pydantic",
        "PyYAML",
        "pandas",
        "pyarrow",
        "openai",
        "anthropic",
        "google-genai",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    manifest["runs"].append(
        {
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": digest(stable_json(config)),
            "git_commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "models": config.get("models", {}),
            "detail": detail or {},
        }
    )
    write_json(path, manifest)


def score_id_for(item: dict[str, Any], scorer: Any) -> str:
    """Return the deterministic active-revision ID for one score.

    Kept here rather than beside the scorers because both the scoring stage and
    the reporting that summarizes it derive IDs, and neither should have to
    import the other to do so. `scorer` is used only for its name and revision.
    """

    return stable_id(
        "score",
        item["record_id"],
        scorer.name,
        scorer.revision,
    )


def metric_score_ids(items: list[dict[str, Any]], scorer: Any) -> set[str]:
    """Return every active-revision score ID expected for one metric."""

    return {score_id_for(item, scorer) for item in items}
