"""
Every call to Claude goes through call_claude(), so token usage, timing, and the
stop reason are recorded the same way for every agent. Agents that must answer
in JSON use call_claude_json(), which also checks the reply and asks again if
it's broken.
"""

import json
import re
import time
from dataclasses import asdict, dataclass

import anthropic

from .config import Config

_client = None


def _get_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment (run.py loads .env first).
    # The SDK retries rate limits and server errors on its own (2 retries by default).
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class AgentError(RuntimeError):
    """An agent couldn't produce a usable answer (e.g. invalid JSON after every retry)."""


@dataclass
class LLMResult:
    role: str             # which step made the call, e.g. "srag_generate", "critic"
    text: str
    model: str
    stop_reason: str      # "end_turn" is normal; "max_tokens" = cut off; "refusal" = declined
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    latency_s: float
    request_id: str | None

    def log_entry(self) -> dict:
        entry = asdict(self)
        del entry["text"]  # the text is logged where it's used, not twice
        return entry


def call_claude(cfg: Config, *, role: str, system: str, messages: list | str,
                json_schema: dict | None = None) -> LLMResult:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    output_config = {"effort": cfg.effort}
    if json_schema:
        # Structured outputs: the API makes the reply follow this JSON schema.
        output_config["format"] = {"type": "json_schema", "schema": json_schema}

    started = time.perf_counter()
    response = _get_client().messages.create(
        model=cfg.generation_model,
        max_tokens=cfg.max_tokens,
        output_config=output_config,
        system=system,
        messages=messages,
    )
    usage = response.usage
    return LLMResult(
        role=role,
        # The response can also hold thinking blocks; we keep only the answer text.
        text="".join(block.text for block in response.content if block.type == "text"),
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        latency_s=round(time.perf_counter() - started, 3),
        request_id=response._request_id,
    )


def parse_json(text: str) -> tuple[object | None, str | None]:
    """
    Parse a JSON reply. Returns (data, None) on success or (None, error message).
    Tolerates a ```json ... ``` fence around the JSON, which models sometimes add.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = (fenced.group(1) if fenced else text).strip()
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


def call_claude_json(cfg: Config, log, *, role: str, system: str, user: str, schema: dict, validate=None):
    """
    Call Claude for a JSON answer and return the parsed data.

    `validate(data)` returns an error message or None. Structured outputs already
    guarantee the JSON's shape; validate checks what a schema can't (e.g. "every
    concept I asked about is answered"). On a bad reply Claude sees its answer and
    the problem, and gets cfg.json_retries more tries before we give up.
    """
    messages = [{"role": "user", "content": user}]
    for attempt in range(cfg.json_retries + 1):
        result = call_claude(cfg, role=role, system=system, messages=messages, json_schema=schema)
        log.add_llm_call(result)
        if result.stop_reason == "refusal":
            raise AgentError(f"{role}: Claude declined the request")

        data, error = parse_json(result.text)
        if error is None and validate:
            error = validate(data)
        if error is None:
            return data

        log.record["llm_calls"][-1]["rejected"] = {"error": error, "reply": result.text[:2000]}
        messages += [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": f"That reply was rejected: {error}. Reply again with corrected JSON."},
        ]
    raise AgentError(f"{role}: no valid answer after {cfg.json_retries + 1} attempts ({error})")
