from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .common import digest, load_config, stable_json, write_json


def report_progress(event: str, **details: Any) -> None:
    """Print one immediately visible, machine-readable progress event."""

    print(
        json.dumps(
            {
                "event": event,
                **details,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class BatchItemResult:
    response: LLMResponse | None = None
    error: str | None = None


class DeferredRequest(Exception):
    """Internal signal used while a stage collects requests for one batch."""


class BatchPending(Exception):
    """Signals that a stage paused normally while remote work is pending."""

    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs
        has_remote_wait = any(job["status"] in {"submitted", "pending"} for job in jobs)

        if has_remote_wait:
            message = (
                "Remote batches are still processing. Rerun the same command "
                "later to poll them and continue from the response cache."
            )
        else:
            message = (
                "Batch results are now cached. Rerun the same command once more "
                "to materialize the stage records."
            )

        super().__init__(message)


# Failure text that a later run can reasonably expect to succeed on. Shared by
# the response cache, the modification stage, and the scoring stage so one
# classification decides what is worth attempting again everywhere.
RETRYABLE_ERROR_FRAGMENTS = (
    "unsupported parameter: 'max_tokens'",
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "connection error",
    "connection reset",
    "paper implementation is not checked out",
    "coco-ex virtual environment was not found",
    "sentiment-steered model was not found",
    "generation model was not found",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "worker is unavailable",
    "batch expired",
    "batch failed",
    "could not be parsed as the expected json object",
    # A funding failure says nothing about the request, so replaying it would
    # write an exhausted account into the results as though the model had
    # declined. Topping up and re-running is what fixes it, and that only works
    # if these are re-sent rather than treated as a settled answer.
    "credit_balance",
    "credit balance",
    "insufficient_quota",
    "billing_hard_limit_reached",
)


def is_retryable_error(error: str | None) -> bool:
    """Return whether a recorded failure should be attempted again."""

    if not error:
        return False

    normalized = error.lower()

    return any(fragment in normalized for fragment in RETRYABLE_ERROR_FRAGMENTS)


class ResponseCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def cache_id(self, key: dict[str, Any]) -> str:
        return digest(stable_json(key))

    def path_for(self, cache_id: str) -> Path:
        return self.directory / f"{cache_id}.json"

    def get(self, cache_id: str) -> LLMResponse | None:
        """Return the cached response, or None when the request must be re-sent.

        A cached failure is replayed rather than re-requested, which is what
        keeps a deterministic provider rejection from being paid for once per
        run. Transient failures are the opposite case: an expired or failed
        provider batch writes one error per request in it, and replaying those
        forever would make a whole job permanently unrecoverable. Those entries
        are therefore discarded on read, so the next run re-sends the request.
        """

        cache_path = self.path_for(cache_id)

        if not cache_path.exists():
            return None

        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        error = cached.get("error")

        if error:
            if cached.get("retryable") or is_retryable_error(error):
                cache_path.unlink(missing_ok=True)
                return None

            raise RuntimeError(error)

        return LLMResponse(
            text=cached["text"],
            raw=cached["raw"],
        )

    def put_response(
        self,
        cache_id: str,
        response: LLMResponse,
    ) -> None:
        serialized = {
            "text": response.text,
            "raw": response.raw,
        }
        write_json(self.path_for(cache_id), serialized)

    def put_error(
        self,
        cache_id: str,
        error: str,
        raw: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        serialized = {
            "error": error,
            "raw": raw or {},
            "retryable": bool(retryable) or is_retryable_error(error),
        }
        write_json(self.path_for(cache_id), serialized)

    def call(
        self,
        key: dict[str, Any],
        llm_client: LLMClient,
        payload: dict[str, Any],
    ) -> tuple[LLMResponse, bool]:
        cache_id = self.cache_id(key)
        cached_response = self.get(cache_id)

        if cached_response is not None:
            return cached_response, True

        response = llm_client.complete(
            payload=payload,
            cache_id=cache_id,
        )
        self.put_response(cache_id, response)

        return response, False


class LLMClient(ABC):
    """Superclass shared by direct and batched provider clients."""

    provider: str
    uses_remote_batch = False

    def __init__(self, batch_session: BatchSession | None = None) -> None:
        self.batch_session = batch_session

        if batch_session is not None:
            batch_session.register_client(self)

    def complete(
        self,
        payload: dict[str, Any],
        cache_id: str,
    ) -> LLMResponse:
        if self.batch_session is None:
            raise RuntimeError(
                f"{self.provider} requests require a stage batch session."
            )

        if self.batch_session.mode == "direct":
            self.batch_session.ensure_direct_request_is_safe(cache_id)
            return self.complete_now(payload)

        self.batch_session.defer(
            llm_client=self,
            cache_id=cache_id,
            payload=payload,
        )
        raise AssertionError("BatchSession.defer always raises DeferredRequest.")

    def complete_now(self, payload: dict[str, Any]) -> LLMResponse:
        """Call the provider's normal synchronous endpoint."""

        raise NotImplementedError(
            f"{self.provider} does not implement direct LLM requests."
        )

    def batch_group(self, payload: dict[str, Any]) -> str:
        return self.provider

    def batch_sort_key(
        self,
        cache_id: str,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        """Keep the default deterministic request ordering."""

        del payload
        return cache_id, cache_id

    @abstractmethod
    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        """Submit a remote batch and return its durable provider job ID."""

    @abstractmethod
    def poll_batch(self, job_id: str) -> str:
        """Return pending, completed, or failed for one provider job."""

    @abstractmethod
    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        """Download terminal results keyed by the submitted custom IDs."""

    def execute_local_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, BatchItemResult]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    provider = "openai"
    uses_remote_batch = True

    def __init__(self, batch_session: BatchSession | None = None) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")

        super().__init__(batch_session)
        self.client = OpenAI(api_key=api_key)

    def complete_now(self, payload: dict[str, Any]) -> LLMResponse:
        result = self.client.chat.completions.create(**payload)
        text = result.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw=result.model_dump(mode="json"),
        )

    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        lines = [
            json.dumps(
                {
                    "custom_id": cache_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": payload,
                },
                ensure_ascii=False,
            )
            for cache_id, payload in requests
        ]
        batch_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        uploaded = self.client.files.create(
            file=("deception-batch.jsonl", batch_bytes),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"project": "deception-review-summaries"},
        )

        return batch.id

    def poll_batch(self, job_id: str) -> str:
        batch = self.client.batches.retrieve(job_id)

        if batch.status == "completed":
            return "completed"

        if batch.status in {"failed", "expired", "cancelled", "canceled"}:
            return "failed"

        return "pending"

    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        batch = self.client.batches.retrieve(job_id)
        results: dict[str, BatchItemResult] = {}

        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue

            content = self.client.files.content(file_id).text

            for line in content.splitlines():
                if not line.strip():
                    continue

                item = json.loads(line)
                cache_id = item["custom_id"]
                response = item.get("response")

                if response and response.get("status_code") == 200:
                    raw = response["body"]
                    text = raw["choices"][0]["message"].get("content") or ""
                    results[cache_id] = BatchItemResult(
                        response=LLMResponse(text=text, raw=raw)
                    )
                else:
                    error = item.get("error") or response or item
                    results[cache_id] = BatchItemResult(
                        error=f"OpenAI batch item failed: {stable_json(error)}"
                    )

        return fill_missing_results(
            provider=self.provider,
            request_ids=request_ids,
            results=results,
        )


class AnthropicClient(LLMClient):
    provider = "anthropic"
    uses_remote_batch = True

    def __init__(self, batch_session: BatchSession | None = None) -> None:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY.")

        super().__init__(batch_session)
        self.client = Anthropic(api_key=api_key)

    def complete_now(self, payload: dict[str, Any]) -> LLMResponse:
        message = self.client.messages.create(**payload)
        text = "".join(block.text for block in message.content if block.type == "text")

        return LLMResponse(
            text=text,
            raw=message.model_dump(mode="json"),
        )

    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        batch = self.client.messages.batches.create(
            requests=[
                {
                    "custom_id": cache_id,
                    "params": payload,
                }
                for cache_id, payload in requests
            ]
        )

        return batch.id

    def poll_batch(self, job_id: str) -> str:
        batch = self.client.messages.batches.retrieve(job_id)

        if batch.processing_status == "ended":
            return "completed"

        return "pending"

    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        results: dict[str, BatchItemResult] = {}

        for item in self.client.messages.batches.results(job_id):
            raw_item = item.model_dump(mode="json")
            cache_id = item.custom_id

            if item.result.type == "succeeded":
                message = item.result.message
                text = "".join(
                    block.text for block in message.content if block.type == "text"
                )
                results[cache_id] = BatchItemResult(
                    response=LLMResponse(
                        text=text,
                        raw=message.model_dump(mode="json"),
                    )
                )
            else:
                results[cache_id] = BatchItemResult(
                    error=(
                        "Anthropic batch item failed: "
                        f"{stable_json(raw_item['result'])}"
                    )
                )

        return fill_missing_results(
            provider=self.provider,
            request_ids=request_ids,
            results=results,
        )


class GoogleClient(LLMClient):
    provider = "google"
    uses_remote_batch = True

    def __init__(self, batch_session: BatchSession | None = None) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")

        super().__init__(batch_session)
        self.client = genai.Client(api_key=api_key)

    def complete_now(self, payload: dict[str, Any]) -> LLMResponse:
        response = self.client.models.generate_content(**payload)

        return LLMResponse(
            text=response.text or "",
            raw=response.model_dump(mode="json"),
        )

    def batch_group(self, payload: dict[str, Any]) -> str:
        return f"{self.provider}:{payload['model']}"

    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        model_names = {payload["model"] for _, payload in requests}

        if len(model_names) != 1:
            raise ValueError("A Gemini batch must contain exactly one model.")

        model_name = model_names.pop()
        inline_requests = [
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": payload["contents"]}],
                    }
                ],
                "config": payload["config"],
            }
            for _, payload in requests
        ]
        batch = self.client.batches.create(
            model=model_name,
            src=inline_requests,
            config={
                "display_name": (
                    "deception-review-summaries-"
                    f"{digest(stable_json([item[0] for item in requests]), 12)}"
                )
            },
        )

        return batch.name

    def poll_batch(self, job_id: str) -> str:
        batch = self.client.batches.get(name=job_id)
        state = batch.state.name

        if state == "JOB_STATE_SUCCEEDED":
            return "completed"

        if state in {
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }:
            return "failed"

        return "pending"

    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        batch = self.client.batches.get(name=job_id)
        inline_responses = batch.dest.inlined_responses or []
        results: dict[str, BatchItemResult] = {}

        # Inline Gemini responses carry no custom ID, so a request is matched to
        # its response by position alone. A short or long response list would
        # therefore attribute summaries to the wrong products silently; refuse
        # the whole job instead, and let the next run resubmit it.
        if len(inline_responses) != len(request_ids):
            return fill_missing_results(
                provider=self.provider,
                request_ids=request_ids,
                results={},
                error=(
                    f"Gemini batch {job_id} returned {len(inline_responses)} "
                    f"responses for {len(request_ids)} requests, so responses "
                    "cannot be matched to requests by position. Batch failed."
                ),
            )

        for cache_id, item in zip(request_ids, inline_responses, strict=True):
            if item.response is not None:
                results[cache_id] = BatchItemResult(
                    response=LLMResponse(
                        text=item.response.text or "",
                        raw=item.response.model_dump(mode="json"),
                    )
                )
            else:
                raw_item = item.model_dump(mode="json")
                results[cache_id] = BatchItemResult(
                    error=f"Gemini batch item failed: {stable_json(raw_item)}"
                )

        return fill_missing_results(
            provider=self.provider,
            request_ids=request_ids,
            results=results,
        )


class ModalQwenClient(LLMClient):
    """Qwen served by a Modal-hosted OpenAI-compatible vLLM endpoint."""

    provider = "modal_qwen"

    @staticmethod
    def resolve_base_url() -> str:
        configured_url = os.getenv("QWEN_MODAL_BASE_URL")

        if configured_url:
            return configured_url

        try:
            import modal
        except ImportError as exception:
            raise RuntimeError(
                "Install the Modal dependency with `uv sync --extra modal`, "
                "or set QWEN_MODAL_BASE_URL explicitly."
            ) from exception

        app_name = load_config()["modal"]["app_name"]

        try:
            server = modal.Server.from_name(
                app_name,
                "QwenServer",
            )
            return server.get_url()
        except Exception as exception:
            raise RuntimeError(
                f"Could not resolve QwenServer in deployed Modal app {app_name!r}. "
                "Deploy modal_app.py or set QWEN_MODAL_BASE_URL explicitly."
            ) from exception

    def __init__(self, batch_session: BatchSession | None = None) -> None:
        import httpx
        from openai import OpenAI

        base_url = self.resolve_base_url()

        modal_key = os.getenv("MODAL_KEY")
        modal_secret = os.getenv("MODAL_SECRET")

        if not modal_key or not modal_secret:
            raise RuntimeError(
                "Missing Modal proxy credentials. Set both MODAL_KEY and "
                "MODAL_SECRET after running "
                "`uv run modal workspace proxy-tokens create`."
            )

        default_headers = {
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        }

        normalized_url = base_url.rstrip("/")

        if not normalized_url.endswith("/v1"):
            normalized_url += "/v1"

        super().__init__(batch_session)
        self.base_url = normalized_url
        self.http_client = httpx.Client(
            base_url=normalized_url,
            headers=default_headers,
            timeout=3600.0,
        )
        self.client = OpenAI(
            base_url=normalized_url,
            api_key="modal-proxy-auth",
            default_headers=default_headers,
            max_retries=0,
            timeout=3600.0,
        )

    def batch_group(self, payload: dict[str, Any]) -> str:
        shared_payload = {
            key: value for key, value in payload.items() if key != "messages"
        }
        return f"{self.provider}:{digest(stable_json(shared_payload), 16)}"

    def batch_sort_key(
        self,
        cache_id: str,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        """Place requests with one exact G-Eval prefix in the same GPU chunk."""

        messages = payload.get("messages", [])

        if not messages:
            return cache_id, cache_id

        content = str(messages[-1].get("content", ""))
        summary_marker = "\nSummary:\n\n"

        if summary_marker not in content:
            return cache_id, cache_id

        shared_prefix = content.split(summary_marker, 1)[0] + summary_marker

        return digest(shared_prefix), cache_id

    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        raise NotImplementedError("Modal work uses vLLM's native batch endpoint.")

    def poll_batch(self, job_id: str) -> str:
        raise NotImplementedError("Modal work uses vLLM's native batch endpoint.")

    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        raise NotImplementedError("Modal work uses vLLM's native batch endpoint.")

    def complete_now(self, payload: dict[str, Any]) -> LLMResponse:
        startup_wait_seconds = 900

        if self.batch_session is not None:
            startup_wait_seconds = int(
                self.batch_session.config.get("batch", {}).get(
                    "modal_startup_wait_seconds",
                    startup_wait_seconds,
                )
            )

        deadline = time.monotonic() + startup_wait_seconds
        started_at = time.monotonic()
        last_reported_at = -15.0
        attempt = 0

        while True:
            attempt += 1

            try:
                result = self.client.chat.completions.create(**payload)
                break
            except Exception as exception:
                status_code = getattr(exception, "status_code", None)

                if status_code not in {429, 502, 503, 504}:
                    raise

                now = time.monotonic()
                elapsed_seconds = now - started_at

                if elapsed_seconds - last_reported_at >= 15:
                    report_progress(
                        "modal_qwen_direct_waiting_for_server",
                        attempt=attempt,
                        elapsed_seconds=round(elapsed_seconds),
                        http_status=status_code,
                    )
                    last_reported_at = elapsed_seconds

                if now >= deadline:
                    raise RuntimeError(
                        "The Modal Qwen endpoint did not become ready within "
                        f"{startup_wait_seconds} seconds. Last response: "
                        f"HTTP {status_code}."
                    ) from exception

                time.sleep(5)

        text = result.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw=result.model_dump(mode="json"),
        )

    def execute_local_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, BatchItemResult]:
        first_payload = requests[0][1]
        batch_payload = {
            key: value
            for key, value in first_payload.items()
            if key not in {"messages", "extra_body"}
        }
        batch_payload["messages"] = [payload["messages"] for _, payload in requests]
        batch_payload.update(first_payload.get("extra_body", {}))

        startup_wait_seconds = 900

        if self.batch_session is not None:
            startup_wait_seconds = int(
                self.batch_session.config.get("batch", {}).get(
                    "modal_startup_wait_seconds",
                    startup_wait_seconds,
                )
            )

        deadline = time.monotonic() + startup_wait_seconds
        started_at = time.monotonic()
        last_reported_at = -15.0
        attempt = 0

        report_progress(
            "modal_qwen_batch_starting",
            requests=len(requests),
            startup_timeout_seconds=startup_wait_seconds,
        )

        while True:
            attempt += 1

            try:
                readiness_response = self.http_client.get(
                    "/models",
                    timeout=30.0,
                )
                readiness_status: int | str = readiness_response.status_code
            except httpx.TimeoutException:
                readiness_response = None
                readiness_status = "timeout"

            if (
                readiness_response is not None
                and readiness_response.status_code == 200
            ):
                break

            now = time.monotonic()
            elapsed_seconds = now - started_at

            if elapsed_seconds - last_reported_at >= 15:
                report_progress(
                    "modal_qwen_waiting_for_server",
                    attempt=attempt,
                    elapsed_seconds=round(elapsed_seconds),
                    http_status=readiness_status,
                    requests=len(requests),
                )
                last_reported_at = elapsed_seconds

            if now >= deadline:
                raise RuntimeError(
                    "The Modal Qwen endpoint did not become ready within "
                    f"{startup_wait_seconds} seconds. Last readiness response: "
                    f"{readiness_status}."
                )

            time.sleep(5)

        attempt = 0

        while True:
            attempt += 1
            response = self.http_client.post(
                "/chat/completions/batch",
                json=batch_payload,
            )

            if response.status_code not in {429, 502, 503, 504}:
                break

            now = time.monotonic()
            elapsed_seconds = now - started_at

            if elapsed_seconds - last_reported_at >= 15:
                report_progress(
                    "modal_qwen_waiting_for_server",
                    attempt=attempt,
                    elapsed_seconds=round(elapsed_seconds),
                    http_status=response.status_code,
                    requests=len(requests),
                )
                last_reported_at = elapsed_seconds

            if now >= deadline:
                raise RuntimeError(
                    "The Modal Qwen endpoint did not become ready within "
                    f"{startup_wait_seconds} seconds. Last response: "
                    f"HTTP {response.status_code}."
                )

            time.sleep(5)

        response.raise_for_status()
        report_progress(
            "modal_qwen_batch_completed",
            elapsed_seconds=round(time.monotonic() - started_at),
            requests=len(requests),
        )
        raw_batch = response.json()
        choices_by_index = {
            int(choice["index"]): choice for choice in raw_batch.get("choices", [])
        }
        results: dict[str, BatchItemResult] = {}

        for index, (cache_id, _) in enumerate(requests):
            choice = choices_by_index.get(index)

            if choice is None:
                results[cache_id] = BatchItemResult(
                    error=(
                        "Modal Qwen batch returned no choice for conversation "
                        f"index {index}."
                    )
                )
                continue

            raw_response = {
                key: value for key, value in raw_batch.items() if key != "choices"
            }
            raw_response["choices"] = [choice]
            text = choice["message"].get("content") or ""
            results[cache_id] = BatchItemResult(
                response=LLMResponse(
                    text=text,
                    raw=raw_response,
                )
            )

        return results

class MockClient(LLMClient):
    """Deterministic recorded-response client used only by the dry run."""

    provider = "mock"

    def __init__(self) -> None:
        super().__init__(batch_session=None)

    def complete(
        self,
        payload: dict[str, Any],
        cache_id: str,
    ) -> LLMResponse:
        del cache_id

        if "messages" in payload:
            prompt = payload["messages"][-1]["content"]
        else:
            prompt = payload["contents"]

        if "Evaluation Form (scores ONLY)" in prompt:
            return LLMResponse(
                text="4",
                raw={
                    "top_logprobs": {
                        "1": -4.0,
                        "2": -3.0,
                        "3": -2.0,
                        "4": -0.2,
                        "5": -2.3,
                    }
                },
            )

        source_marker = digest(prompt, 8)

        return LLMResponse(
            text=(
                "Many customers report good performance. "
                "Some customers report bad value. "
                "The product is easy to use. "
                "Packaging is ordinary. "
                f"Recorded source marker {source_marker}."
            ),
            raw={"mock": True},
        )

    def submit_batch(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        raise NotImplementedError

    def poll_batch(self, job_id: str) -> str:
        raise NotImplementedError

    def download_batch(
        self,
        job_id: str,
        request_ids: list[str],
    ) -> dict[str, BatchItemResult]:
        raise NotImplementedError


class BatchSession:
    """Collects, submits, polls, and caches resumable provider batch jobs."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = config.get("llm_execution", "batch")

        if self.mode not in {"batch", "direct"}:
            raise ValueError(
                f"llm_execution must be either 'batch' or 'direct', not {self.mode!r}."
            )

        self.cache = ResponseCache(config["cache"])
        self.state_path = Path(config["state"]) / "batch_jobs.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.queued: dict[
            str,
            dict[str, tuple[dict[str, Any], LLMClient]],
        ] = {}
        self.clients: dict[str, LLMClient] = {}
        self.deferred_requests = False
        self.active_request_ids: set[str] = set()
        batch_config = config.get("batch", {})
        self.max_requests_per_job = int(batch_config.get("max_requests_per_job", 1000))
        self.max_payload_bytes = int(batch_config.get("max_payload_bytes", 15_000_000))
        self.modal_requests_per_job = int(
            batch_config.get("modal_requests_per_job", 64)
        )

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"jobs": {}}

        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        write_json(self.state_path, self.state)

    def register_client(self, llm_client: LLMClient) -> None:
        self.clients[llm_client.provider] = llm_client

    def refresh(self) -> None:
        """Poll known jobs before a stage rebuilds its missing work."""

        if self.mode == "direct":
            return

        self._poll_existing_jobs()
        self._save_state()

    def ensure_direct_request_is_safe(self, cache_id: str) -> None:
        """Prevent a direct request from duplicating pending batch work."""

        for job in self.state["jobs"].values():
            is_pending = job["status"] in {"submitted", "pending"}
            contains_request = cache_id in job["request_ids"]

            if is_pending and contains_request:
                raise RuntimeError(
                    "This request is already part of pending "
                    f"{job['provider']} batch {job['job_id']}. Run in batch mode "
                    "to poll it instead of paying for a duplicate direct request."
                )

    def defer(
        self,
        llm_client: LLMClient,
        cache_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.deferred_requests = True
        self.active_request_ids.add(cache_id)
        self.clients[llm_client.provider] = llm_client

        for job in self.state["jobs"].values():
            if job["status"] == "completed" and cache_id in job["request_ids"]:
                job["status"] = "submitted"

            if (
                job["status"] in {"submitted", "pending"}
                and cache_id in job["request_ids"]
            ):
                raise DeferredRequest

        group = llm_client.batch_group(payload)
        queued_group = self.queued.setdefault(group, {})
        queued_group[cache_id] = (payload, llm_client)
        raise DeferredRequest

    def _chunks(
        self,
        requests: list[tuple[str, dict[str, Any]]],
        max_requests: int | None = None,
    ) -> list[list[tuple[str, dict[str, Any]]]]:
        chunks: list[list[tuple[str, dict[str, Any]]]] = []
        current: list[tuple[str, dict[str, Any]]] = []
        current_bytes = 0

        request_limit = max_requests or self.max_requests_per_job

        for request in requests:
            request_bytes = len(stable_json(request).encode("utf-8"))
            exceeds_count = len(current) >= request_limit
            exceeds_bytes = (
                current and current_bytes + request_bytes > self.max_payload_bytes
            )

            if exceeds_count or exceeds_bytes:
                chunks.append(current)
                current = []
                current_bytes = 0

            current.append(request)
            current_bytes += request_bytes

        if current:
            chunks.append(current)

        return chunks

    def _cache_results(
        self,
        results: dict[str, BatchItemResult],
    ) -> None:
        for cache_id, result in results.items():
            if result.response is not None:
                self.cache.put_response(cache_id, result.response)
            else:
                self.cache.put_error(
                    cache_id=cache_id,
                    error=result.error or "Unknown batch item failure.",
                )

    def _poll_existing_jobs(self) -> None:
        for job in self.state["jobs"].values():
            if job["status"] not in {"submitted", "pending"}:
                continue

            llm_client = self.clients.get(job["provider"])

            if llm_client is None:
                continue

            status = llm_client.poll_batch(job["job_id"])
            job["status"] = status
            job["last_checked_at"] = datetime.now(timezone.utc).isoformat()

            if status == "completed":
                results = llm_client.download_batch(
                    job_id=job["job_id"],
                    request_ids=job["request_ids"],
                )
                self._cache_results(results)
                job["completed_at"] = datetime.now(timezone.utc).isoformat()

            elif status == "failed":
                # A whole job fails when the provider expires, cancels, or
                # errors it - never because of anything specific to one
                # request - so these are recorded as retryable. The next run
                # re-queues them instead of replaying the failure forever.
                for cache_id in job["request_ids"]:
                    self.cache.put_error(
                        cache_id,
                        f"{job['provider']} batch {job['job_id']} failed.",
                        retryable=True,
                    )

            self._save_state()

    def _submit_queued_jobs(self) -> None:
        for group, queued_requests in self.queued.items():
            requests = sorted(
                queued_requests.items(),
                key=lambda item: item[1][1].batch_sort_key(
                    cache_id=item[0],
                    payload=item[1][0],
                ),
            )
            normalized_requests = [
                (cache_id, payload_and_client[0])
                for cache_id, payload_and_client in requests
            ]
            llm_client = requests[0][1][1]

            request_limit = (
                self.max_requests_per_job
                if llm_client.uses_remote_batch
                else self.modal_requests_per_job
            )

            for chunk in self._chunks(
                normalized_requests,
                max_requests=request_limit,
            ):
                request_ids = [cache_id for cache_id, _ in chunk]
                state_key = digest(
                    stable_json(
                        {
                            "group": group,
                            "request_ids": request_ids,
                        }
                    )
                )

                if state_key in self.state["jobs"]:
                    continue

                created_at = datetime.now(timezone.utc).isoformat()

                if llm_client.uses_remote_batch:
                    report_progress(
                        "provider_batch_submitting",
                        provider=llm_client.provider,
                        requests=len(chunk),
                    )
                    job_id = llm_client.submit_batch(chunk)
                    status = "submitted"
                    report_progress(
                        "provider_batch_submitted",
                        job_id=job_id,
                        provider=llm_client.provider,
                        requests=len(chunk),
                    )
                else:
                    job_id = f"modal-local-{state_key[:16]}"
                    results = llm_client.execute_local_batch(chunk)
                    self._cache_results(results)
                    status = "completed"

                self.state["jobs"][state_key] = {
                    "provider": llm_client.provider,
                    "group": group,
                    "job_id": job_id,
                    "request_ids": request_ids,
                    "status": status,
                    "created_at": created_at,
                }
                self._save_state()

    def finish(self) -> None:
        if self.mode == "direct":
            return

        if not self.deferred_requests:
            return

        self._submit_queued_jobs()
        self._save_state()

        jobs = [
            {
                "provider": job["provider"],
                "job_id": job["job_id"],
                "status": job["status"],
                "requests": len(job["request_ids"]),
            }
            for job in self.state["jobs"].values()
            if (
                job["status"] in {"submitted", "pending", "completed"}
                and self.active_request_ids.intersection(job["request_ids"])
            )
        ]
        raise BatchPending(jobs)


def fill_missing_results(
    provider: str,
    request_ids: list[str],
    results: dict[str, BatchItemResult],
    error: str | None = None,
) -> dict[str, BatchItemResult]:
    for cache_id in request_ids:
        if cache_id not in results:
            results[cache_id] = BatchItemResult(
                error=error
                or f"{provider} batch returned no result for {cache_id}."
            )

    return results


def client(
    provider: str,
    mock: bool = False,
    batch_session: BatchSession | None = None,
) -> LLMClient:
    if mock:
        return MockClient()

    load_dotenv()

    client_classes: dict[str, type[LLMClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "modal_qwen": ModalQwenClient,
    }

    try:
        client_class = client_classes[provider]
    except KeyError as exception:
        raise ValueError(f"Unsupported LLM provider: {provider}") from exception

    return client_class(batch_session=batch_session)


# Sampling parameters that a decoding block may omit when the target model
# rejects them. `max_tokens` is always required.
OPTIONAL_SAMPLING_PARAMETERS = ("temperature", "top_p", "seed")

# Providers whose API can be asked for a JSON object directly. The Anthropic
# Messages API has no equivalent response mode.
JSON_OBJECT_PROVIDERS = frozenset({"openai", "google", "qwen", "modal_qwen"})


def request_model_name(model: dict[str, Any]) -> str:
    """Return the identifier the provider expects in the request body."""

    return model.get("request_model", model["model"])


def model_identity(model: dict[str, Any]) -> str:
    """Return the string that stamps results produced by this model.

    It is hashed into summary, modification, and score IDs, so it is what keeps
    results from two different checkpoints apart instead of colliding. A model
    entry may pin `revision`; when it does not, the model name carries that role
    on its own, which is enough whenever the name changes with the checkpoint
    (`Qwen/Qwen3.5-27B` to `Qwen/Qwen3.6-27B`). It does not protect against new
    weights pushed under an unchanged name - pin a revision when that matters.
    """

    return model.get("revision") or model["model"]


def sampling_parameters(
    decoding: dict[str, Any],
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the optional sampling parameters this decoding block sets.

    A model may declare `unsupported_sampling` to refuse individual parameters
    it rejects outright - OpenAI reasoning models answer `temperature: 0` with
    a 400. The refusal belongs to the model rather than the decoding block,
    because one block is shared by every model in a stage; declaring it here
    keeps that block portable instead of forcing it down to the weakest model.
    """

    unsupported = set((model or {}).get("unsupported_sampling", ()))

    return {
        name: decoding[name]
        for name in OPTIONAL_SAMPLING_PARAMETERS
        if name in decoding and name not in unsupported
    }


def provider_supports_json(provider: str) -> bool:
    """Report whether this provider can serve a stage that requires JSON."""

    return provider in JSON_OBJECT_PROVIDERS


def _chat_completions_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool,
    token_parameter: str = "max_tokens",
) -> dict[str, Any]:
    """Build the chat-completions body every OpenAI-compatible provider shares."""

    request: dict[str, Any] = {
        "model": request_model_name(model),
        "messages": [{"role": "user", "content": prompt}],
        **sampling_parameters(decoding, model),
        token_parameter: decoding["max_tokens"],
    }

    if json_object:
        request["response_format"] = {"type": "json_object"}

    return request


def _openai_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool,
) -> dict[str, Any]:
    # Chat models expose no thinking parameter. Reasoning models take
    # `reasoning_effort` and reject `max_tokens`; both are named in the
    # decoding block so a modifier swap needs no code change.
    request = _chat_completions_request(
        model,
        prompt,
        decoding,
        json_object,
        token_parameter=decoding.get("openai_max_tokens_parameter", "max_tokens"),
    )

    if "openai_reasoning_effort" in decoding:
        request["reasoning_effort"] = decoding["openai_reasoning_effort"]

    return request


def _anthropic_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool,
) -> dict[str, Any]:
    if json_object:
        raise ValueError(
            "The Anthropic Messages API has no JSON-object response mode, so an "
            "Anthropic model cannot serve a stage that requires JSON output. "
            "Use an OpenAI-compatible or Google model instead."
        )

    # Anthropic has no seed parameter and rejects temperature together with
    # top_p, so temperature alone carries the decoding contract here.
    request: dict[str, Any] = {
        "model": request_model_name(model),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": decoding["max_tokens"],
        "thinking": {"type": "disabled"},
    }

    if "temperature" in sampling_parameters(decoding, model):
        request["temperature"] = decoding["temperature"]

    return request


def _google_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        **sampling_parameters(decoding, model),
        "max_output_tokens": decoding["max_tokens"],
        # A zero budget is how the 2.5 family spells "no thinking tokens".
        "thinking_config": {"thinking_budget": 0},
    }

    if json_object:
        generation_config["response_mime_type"] = "application/json"

    return {
        "model": request_model_name(model),
        "contents": prompt,
        "config": generation_config,
    }


def _qwen_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool,
) -> dict[str, Any]:
    # Off unless a decoding block opts in. Generation and G-Eval scoring never
    # set this, so they keep the greedy, non-reasoning behavior everything else
    # in the module assumes; the modification judge sets it to reason before
    # producing its edits. vLLM's --reasoning-parser strips the reasoning into
    # `message.reasoning_content`, so `message.content` stays clean JSON either
    # way and callers need no change to read it.
    thinking = bool(decoding.get("qwen_enable_thinking", False))
    request = _chat_completions_request(model, prompt, decoding, json_object)
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    # vLLM forces the model into its `reasoning_end_str` once this many
    # reasoning tokens are spent, so a long chain-of-thought no longer costs
    # the full max_tokens budget. Requires the server's `--reasoning-config`;
    # against a server without it this key is simply ignored, not an error.
    # Only meaningful with thinking on - a budget alongside disabled thinking
    # has nothing to cap.
    if thinking and "qwen_thinking_token_budget" in decoding:
        extra_body["thinking_token_budget"] = int(
            decoding["qwen_thinking_token_budget"]
        )

    request["extra_body"] = extra_body

    return request


RequestBuilder = Callable[[dict[str, Any], str, dict[str, Any], bool], dict[str, Any]]

REQUEST_BUILDERS: dict[str, RequestBuilder] = {
    "openai": _openai_request,
    "anthropic": _anthropic_request,
    "google": _google_request,
    "qwen": _qwen_request,
    "modal_qwen": _qwen_request,
}


def build_provider_request(
    provider: str,
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
    json_object: bool = False,
) -> dict[str, Any]:
    """Express one decoding configuration in a provider's request format.

    Every stage routes through this function so a decoding block means the same
    thing whichever provider serves it. Thinking is disabled everywhere.
    Parameters a provider rejects are dropped rather than silently
    reinterpreted, so the block stays portable across models.
    """

    try:
        builder = REQUEST_BUILDERS[provider]
    except KeyError as error:
        raise ValueError(f"Unsupported provider: {provider}") from error

    return builder(model, prompt, decoding, json_object)


def build_qwen_score_request(
    model: dict[str, Any],
    prompt: str,
    decoding: dict[str, Any],
) -> dict[str, Any]:
    """Shape a G-Eval score request: one token, constrained to the scale."""

    request = build_provider_request(
        provider="qwen",
        model=model,
        prompt=prompt,
        decoding={**decoding, "max_tokens": 1},
    )
    request["logprobs"] = True
    request["top_logprobs"] = 5
    request["extra_body"]["structured_outputs"] = {
        "choice": ["1", "2", "3", "4", "5"],
    }

    return request
