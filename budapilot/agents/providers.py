"""LLM providers. One narrow interface, two implementations, zero agent changes.

Every agent in this system does the same thing: send a stable system prompt and a
per-bar user prompt, and get back an instance of a Pydantic model. That is the entire
contract, and it is the only thing that touches a vendor SDK. Everything else -- the
eight agents, their prompts, their stubs, the bus, the risk engine, execution -- is
provider-agnostic and does not import ``anthropic`` or ``google.genai`` at all.

So swapping providers is this file, and nothing else.

The two implementations are deliberately *not* unified beyond the ``complete()``
signature. Anthropic's ``messages.parse(output_format=...)`` and Gemini's
``response_schema=`` reach the same place by different routes, and pretending
otherwise -- a shared "options" object, a translation layer -- would add a second
thing to debug every time one vendor changes. Each adapter speaks its own SDK
natively and converts at the boundary.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

# (input_tokens, output_tokens) -- zero when a provider does not report them.
Usage = tuple[int, int]


class LLMProvider(Protocol):
    """What an agent needs from a model vendor. Nothing more."""

    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output_model: type[BaseModel],
        max_tokens: int,
        effort: str | None,
    ) -> tuple[BaseModel, Usage]:
        """Return a validated ``output_model`` instance, or raise.

        Raising is correct here: ``Agent.run`` catches everything and degrades to the
        stub. A provider that silently returns a half-filled object would put made-up
        numbers in front of the risk engine, which is far worse than a grey chip.
        """
        ...

    async def aclose(self) -> None: ...


# ======================================================================================
# Anthropic
# ======================================================================================


class AnthropicProvider:
    """Claude via ``messages.parse``, which does constrained decoding server-side.

    The system prompt is sent as a cached block. System prompts are byte-identical
    across bars, so ephemeral caching removes most of the input cost on the Sonnet and
    Opus agents -- the ones called most often per bar.
    """

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def client(self) -> Any:
        return self._client

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output_model: type[BaseModel],
        max_tokens: int,
        effort: str | None,
    ) -> tuple[BaseModel, Usage]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user}],
            "output_format": output_model,
        }
        if effort:
            kwargs["output_config"] = {"effort": effort}

        resp = await self._client.messages.parse(**kwargs)
        parsed = resp.parsed_output
        if parsed is None:
            raise ValueError("model returned no parsable structured output")
        usage = (
            getattr(resp.usage, "input_tokens", 0) or 0,
            getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return parsed, usage

    async def aclose(self) -> None:
        await self._client.close()


# ======================================================================================
# Gemini
# ======================================================================================

# Thinking budget in output tokens, by the agent's declared effort. Only the deep
# analysis agent sets an effort at all.
_THINKING_BUDGET = {None: 0, "low": 512, "medium": 2048, "high": 8192, "max": 24576}


class GeminiProvider:
    """Gemini via ``response_schema``, which also constrains decoding server-side.

    Two things here are not obvious and both cost real debugging time:

    **Thinking eats the output budget.** Gemini 2.5 models think by default, and
    reasoning tokens are drawn from ``max_output_tokens``. With the 1024-token budget
    these agents declare, a thinking model can spend the lot deliberating and return
    an empty candidate -- ``response.parsed is None`` with no error to explain it.
    So thinking is off unless the agent asked for it, and the token budget is raised
    to cover the thinking that remains. Silent empty responses on the PM agent would
    have meant a demo where the arbiter mysteriously degraded every bar.

    **Gemini has no prompt-cache control.** Anthropic's ephemeral cache block has no
    equivalent to set; Gemini caches implicitly above a model-dependent token floor.
    Nothing to do, but it means Gemini's cost profile here is worse than Claude's, not
    better, despite the cheaper per-token price.
    """

    name = "gemini"

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    @property
    def client(self) -> Any:
        return self._client

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output_model: type[BaseModel],
        max_tokens: int,
        effort: str | None,
    ) -> tuple[BaseModel, Usage]:
        from google.genai import types

        budget = _THINKING_BUDGET.get(effort, 0)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=output_model,
            # Headroom for the thinking tokens, which come out of the same budget.
            max_output_tokens=max_tokens + budget,
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
        )
        resp = await self._client.aio.models.generate_content(
            model=model, contents=user, config=config
        )

        parsed = resp.parsed
        if parsed is None:
            # The SDK only populates .parsed on a clean schema match. Falling back to
            # the raw text gives a real validation error to journal instead of an
            # unexplained None.
            text = (resp.text or "").strip()
            if not text:
                why = (
                    getattr(resp.candidates[0], "finish_reason", "?")
                    if resp.candidates
                    else "no candidates"
                )
                raise ValueError(f"gemini returned no content (finish_reason={why})")
            parsed = output_model.model_validate_json(text)
        if not isinstance(parsed, output_model):
            raise TypeError(f"expected {output_model.__name__}, got {type(parsed).__name__}")

        um = resp.usage_metadata
        usage = (
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
        )
        return parsed, usage

    async def aclose(self) -> None:
        # google-genai manages its own httpx lifecycle and exposes no close().
        return None


PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def build_provider(name: str, api_key: str) -> LLMProvider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}"
        ) from None
    return cls(api_key)


__all__ = ["PROVIDERS", "AnthropicProvider", "GeminiProvider", "LLMProvider", "build_provider"]
