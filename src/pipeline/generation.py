from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

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
from .data import load_products
from .llms import (
    BatchSession,
    DeferredRequest,
    LLMClient,
    ResponseCache,
    build_provider_request,
    client,
    model_identity,
)
from .models import Error, GenerationMeta, Model, Product, Summary
from .reporting import print_generation_plan, report_generation_status


class Summarizer:
    """Generation workflow for one configured model."""

    def __init__(
        self,
        model: dict[str, Any],
        decoding: dict[str, Any],
        cache: ResponseCache,
        batch_session: BatchSession,
        mock: bool = False,
    ) -> None:
        self.model = model
        self.decoding = decoding
        self.cache = cache
        self.mock = mock
        self.client: LLMClient = client(
            provider=model["provider"],
            mock=mock,
            batch_session=batch_session,
        )

    def build_request(self, prompt: str) -> dict[str, Any]:
        """Build exactly the payload supported by this model's provider."""

        return build_provider_request(
            provider=self.model["provider"],
            model=self.model,
            prompt=prompt,
            decoding=self.decoding,
        )

    def summarize(
        self,
        prompt: str,
        source_hash: str,
    ) -> tuple[str, dict[str, Any], str, bool]:
        request = self.build_request(prompt)

        cache_key = {
            "provider": self.model["provider"],
            "model": self.model["model"],
            "revision": model_identity(self.model),
            "prompt": prompt,
            "source_hash": source_hash,
            "request": request,
        }

        if self.mock:
            cache_key["execution"] = "mock"

        response, was_cached = self.cache.call(
            key=cache_key,
            llm_client=self.client,
            payload=request,
        )

        request_hash = digest(stable_json(cache_key))
        return response.text.strip(), request, request_hash, was_cached


class GeneratedText(Model):
    text: str
    prompt_hash: str
    request: dict[str, Any]
    request_hash: str
    was_cached: bool


MOCK_SUMMARY_MARKER = "Recorded source marker "


def quarantine_mock_generation_artifacts(
    config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, int]:
    """Remove known mock responses from live storage without discarding them."""

    output_path = Path(output_path)
    summaries = read_jsonl(output_path, Summary)
    mock_summaries = [
        summary
        for summary in summaries
        if summary.text is not None
        and MOCK_SUMMARY_MARKER in summary.text
    ]

    if mock_summaries:
        mock_ids = {summary.summary_id for summary in mock_summaries}
        live_summaries = [
            summary
            for summary in summaries
            if summary.summary_id not in mock_ids
        ]
        quarantine_path = (
            output_path.parent
            / "quarantine"
            / "mock_summaries.jsonl"
        )
        quarantined_summaries = read_jsonl(quarantine_path, Summary)
        quarantined_by_id = {
            summary.summary_id: summary
            for summary in [
                *quarantined_summaries,
                *mock_summaries,
            ]
        }
        write_jsonl(
            quarantine_path,
            list(quarantined_by_id.values()),
        )
        write_jsonl(output_path, live_summaries)

    cache_directory = Path(config["cache"])
    cache_quarantine = cache_directory / "mock_quarantine"
    mock_cache_files = []

    if cache_directory.is_dir():
        for cache_path in cache_directory.glob("*.json"):
            if cache_path.name.startswith("._"):
                continue

            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if cached.get("raw", {}).get("mock") is True:
                mock_cache_files.append(cache_path)

    if mock_cache_files:
        cache_quarantine.mkdir(parents=True, exist_ok=True)

        for cache_path in mock_cache_files:
            cache_path.replace(cache_quarantine / cache_path.name)

    counts = {
        "summaries": len(mock_summaries),
        "cache_responses": len(mock_cache_files),
    }

    if any(counts.values()):
        tqdm.write(
            json.dumps(
                {
                    "event": "mock_artifacts_quarantined",
                    **counts,
                },
                sort_keys=True,
            )
        )

    return counts


def resolve_summary_conditions(
    config: dict[str, Any],
    selected: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Resolve repeatable style selectors into the conditions to generate.

    Selecting nothing generates every configured condition, which is what an
    unattended run does. Naming conditions restricts generation to those alone
    without touching the summaries already stored for the others.
    """

    available = config["summary_conditions"]

    if not selected:
        return dict(available)

    names = list(dict.fromkeys(selected))
    unsupported = sorted(set(names) - set(available))

    if unsupported:
        raise ValueError(
            "Unknown summary conditions: "
            + ", ".join(unsupported)
            + ". Choose from: "
            + ", ".join(available)
        )

    return {name: available[name] for name in names}


@lru_cache(maxsize=None)
def load_style_prompt(name: str) -> tuple[str, str]:
    """Read and hash one style prompt. Cached: a run reads each file once."""

    path = Path("prompts/styles") / name

    if not path.is_file():
        raise FileNotFoundError(f"Missing summary prompt: {path}")

    text = path.read_text(encoding="utf-8")

    if "{input_text}" not in text:
        raise ValueError(f"Prompt lacks the {{input_text}} placeholder: {path}")

    return text, digest(text)


def preflight(
    config: dict[str, Any],
    products: list[Product],
    batch_session: BatchSession,
    mock: bool = False,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Summarizer]:
    # The one-faithful-prompt rule is a property of the configuration, so it
    # holds whether or not this run generates every condition.
    faithful_prompt_hashes = {
        load_style_prompt(specification["prompt"])[1]
        for specification in config["summary_conditions"].values()
        if specification["ablation"]
    }

    if len(faithful_prompt_hashes) != 1:
        raise ValueError(
            "The four faithful summarizer conditions must use one exact prompt."
        )

    if conditions is None:
        conditions = config["summary_conditions"]

    cache = ResponseCache(config["cache"])
    summarizers: dict[str, Summarizer] = {}

    model_keys = {
        specification["model"] for specification in conditions.values()
    }

    context_budget = config["decoding"]["context_tokens"]

    for model_key in model_keys:
        model = config["models"][model_key]
        revision = model.get("revision")

        # A model entry may omit `revision` and be identified by name alone;
        # only an explicitly configured one has to be expanded.
        if revision and not mock and "$" in revision:
            raise ValueError(f"Missing pinned revision for {model_key}.")

        check_context_budget(model_key, model, context_budget, "generation")

        summarizers[model_key] = Summarizer(
            model=model,
            decoding=config["decoding"],
            cache=cache,
            batch_session=batch_session,
            mock=mock,
        )

    for product in products:
        for specification in conditions.values():
            template, _ = load_style_prompt(specification["prompt"])
            check_prompt_budget(
                prompt_characters=len(template) + len(product.source_text),
                decoding=config["decoding"],
                subject=product.product_id,
                stage="generation",
            )

    return summarizers


def generate_text(
    product: Product,
    condition: str,
    config: dict[str, Any],
    summarizers: dict[str, Summarizer],
) -> GeneratedText:
    specification = config["summary_conditions"][condition]
    template, prompt_hash = load_style_prompt(specification["prompt"])
    prompt = template.replace("{input_text}", product.source_text)
    summarizer = summarizers[specification["model"]]

    text, request, request_hash, was_cached = summarizer.summarize(
        prompt=prompt,
        source_hash=product.source_hash,
    )

    if not text:
        raise ValueError("The provider returned an empty summary.")

    return GeneratedText(
        text=text,
        prompt_hash=prompt_hash,
        request=request,
        request_hash=request_hash,
        was_cached=was_cached,
    )


def summary_id_for(
    product: Product,
    condition: str,
    config: dict[str, Any],
) -> str:
    """Return the stable identifier one product-condition pair is saved under."""

    specification = config["summary_conditions"][condition]
    _, prompt_hash = load_style_prompt(specification["prompt"])
    model = config["models"][specification["model"]]

    return stable_id(
        "sum",
        product.dataset_role,
        product.product_id,
        condition,
        model_identity(model),
        prompt_hash,
        product.source_hash,
    )


def run(
    config: dict[str, Any],
    output: str | None = None,
    dry_run: bool = False,
    selected_conditions: list[str] | tuple[str, ...] | None = None,
) -> list[Summary]:
    conditions = resolve_summary_conditions(config, selected_conditions)
    output_path = output or config["outputs"]["summaries"]
    quarantine_mock_generation_artifacts(
        config=config,
        output_path=output_path,
    )
    products = load_products(
        config["data"],
        dry_run=dry_run,
        order_seed=config.get("review_order_seed"),
    )
    batch_session = BatchSession(config=config)
    tqdm.write(
        json.dumps(
            {
                "stage": "generation",
                "status": "collecting_requests",
                "products": len(products),
                "summary_conditions": len(conditions),
                "selected_conditions": sorted(conditions),
            },
            sort_keys=True,
        )
    )
    summarizers = preflight(
        config=config,
        products=products,
        batch_session=batch_session,
        mock=dry_run,
        conditions=conditions,
    )
    batch_session.refresh()

    loaded_records = read_jsonl(output_path, Summary)
    latest_records = {
        record.summary_id: record
        for record in loaded_records
    }
    records = list(latest_records.values())

    if len(records) != len(loaded_records):
        write_jsonl(output_path, records)

    record_positions = {
        record.summary_id: index
        for index, record in enumerate(records)
    }
    completed_ids = {
        record.summary_id
        for record in records
        if record.error is None
    }
    report_generation_status(
        config=config,
        products=products,
        records=records,
        batch_session=batch_session,
        moment="before_generation",
        conditions=conditions,
    )
    expected_by_condition = {
        condition: {
            summary_id_for(product, condition, config) for product in products
        }
        for condition in conditions
    }
    print_generation_plan(
        config=config,
        products=products,
        conditions=conditions,
        expected_by_condition=expected_by_condition,
        completed_ids=completed_ids,
        output_path=output_path,
    )
    expected_ids = (
        set().union(*expected_by_condition.values()) if conditions else set()
    )
    counts = {"saved": 0, "cached": 0, "errors": 0, "deferred": 0}
    progress = tqdm(
        total=len(expected_ids),
        initial=len(expected_ids & completed_ids),
        desc="generation",
        unit="summary",
        dynamic_ncols=True,
        leave=True,
    )

    try:
        for product in products:
            for condition, specification in conditions.items():
                _, prompt_hash = load_style_prompt(specification["prompt"])
                model = config["models"][specification["model"]]
                summary_id = summary_id_for(product, condition, config)

                if summary_id in completed_ids:
                    continue

                summary_text = None
                error = None
                request: dict[str, Any] = {}
                request_hash = ""
                was_cached = False
                progress.set_postfix(
                    product=str(product.product_id)[:24],
                    condition=condition,
                    **counts,
                    refresh=True,
                )

                try:
                    generated = generate_text(
                        product=product,
                        condition=condition,
                        config=config,
                        summarizers=summarizers,
                    )
                    summary_text = generated.text
                    request = generated.request
                    request_hash = generated.request_hash
                    was_cached = generated.was_cached

                except DeferredRequest:
                    # Batch mode: the request is queued, not answered, so the bar
                    # must not advance until a later run collects the response.
                    counts["deferred"] += 1
                    continue

                except Exception as exception:
                    error = Error(
                        kind=type(exception).__name__,
                        message=str(exception),
                    )

                record = Summary(
                    summary_id=summary_id,
                    product_id=product.product_id,
                    source=product.source,
                    category=product.category,
                    review_count_bin=product.review_count_bin,
                    dataset_role=product.dataset_role,
                    condition=condition,
                    summarizer_model=model["model"],
                    is_primary_panel=specification["primary"],
                    is_model_ablation=specification["ablation"],
                    source_text=product.source_text,
                    source_hash=product.source_hash,
                    source_reviews=product.reviews,
                    text=summary_text,
                    error=error,
                    generation=GenerationMeta(
                        provider=model["provider"],
                        model=model["model"],
                        revision=model_identity(model),
                        prompt_hash=prompt_hash,
                        request=request,
                        request_hash=request_hash,
                        cached=was_cached,
                    ),
                )

                # Always append: rewriting the file whenever a previously
                # failed summary succeeds makes a long run quadratic in the
                # size of its own output. Loading keeps the last row per
                # `summary_id` and rewrites the file once when it finds
                # duplicates, so a superseded row never survives a rerun.
                append_jsonl(output_path, record)
                existing_position = record_positions.get(summary_id)

                if existing_position is None:
                    record_positions[summary_id] = len(records)
                    records.append(record)
                else:
                    records[existing_position] = record

                if record.error is None:
                    completed_ids.add(summary_id)
                    counts["saved"] += 1
                    counts["cached"] += int(was_cached)
                    # The bar tracks summaries that are actually done, so a failed
                    # attempt leaves it in place rather than reporting false
                    # progress for work the next run has to repeat.
                    progress.update(1)
                else:
                    counts["errors"] += 1
                    tqdm.write(
                        f"  {condition} / {product.product_id}: "
                        f"{error.kind}: {error.message}"
                    )

                progress.set_postfix(**counts, refresh=False)
    finally:
        progress.close()

    tqdm.write(
        f"Generation pass finished: {counts['saved']:,} saved "
        f"({counts['cached']:,} from cache), {counts['errors']:,} failed, "
        f"{counts['deferred']:,} awaiting a provider batch; "
        f"{len(expected_ids & completed_ids):,}/{len(expected_ids):,} complete."
    )

    report_generation_status(
        config=config,
        products=products,
        records=records,
        batch_session=batch_session,
        moment="after_generation",
        conditions=conditions,
    )

    batch_session.finish()
    return records
