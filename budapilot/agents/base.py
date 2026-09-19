"""The agent base class: one Anthropic call, schema-validated, never raises.

Structured output goes through ``client.messages.parse(output_format=Model)``, which
transforms the Pydantic schema into the constrained-decoding subset and validates the
response for us. The system prompt is sent as a cached block -- system prompts are
identical across bars, so ephemeral caching removes most of the input cost on the
Sonnet and Opus agents.

The contract every agent honours: **run() does not raise.** On timeout, API error or
schema violation it returns its stub output with ``status=DEGRADED`` and the error
recorded. A dead news feed must never stop the trading loop. (NFR2)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from budapilot.agents.providers import LLMProvider, build_provider
from budapilot.config import MODELS, PROVIDER_MODELS, TIMEOUTS, models_for, settings
from budapilot.contracts import AgentResult, AgentStatus

log = logging.getLogger("budapilot.agents")

TOut = TypeVar("TOut", bound=BaseModel)


class AgentRuntime:
    """Shared LLM provider. ``offline=True`` forces every agent to its stub.

    ``provider`` selects the vendor: ``"anthropic"``, ``"gemini"``, or ``None`` to
    resolve from ``LLM_PROVIDER`` / whichever key is present. The agents themselves
    know nothing about this -- they get a model id and a validated Pydantic object.

    ``fallback`` names a second vendor to try when the primary call fails or times
    out: ``"auto"`` picks whichever other provider has a key, an explicit name pins
    it, and ``None`` disables it. The fallback is per call, not per session -- one
    rate-limited Claude request is served by Gemini and the next goes back to Claude.
    """

    def __init__(
        self,
        *,
        offline: bool = False,
        api_key: str | None = None,
        provider: str | None = None,
        fallback: str | None = None,
    ) -> None:
        resolved, resolved_key = settings.resolve_provider(provider)
        key = api_key if api_key is not None else resolved_key

        self.provider_name = resolved
        self.models = models_for(resolved)
        self.offline = offline or not key
        self._provider: LLMProvider | None = None
        self.fallback_name: str | None = None
        self.fallback_models: dict[str, str] = {}
        self._fallback: LLMProvider | None = None
        if not self.offline:
            self._provider = build_provider(resolved, key)
            fb = self._pick_fallback(fallback)
            if fb:
                self.fallback_name = fb
                self.fallback_models = models_for(fb)
                self._fallback = build_provider(fb, settings.key_for(fb))

    def _pick_fallback(self, requested: str | None) -> str | None:
        if not requested:
            return None
        if requested == "auto":
            return next(
                (
                    p
                    for p in PROVIDER_MODELS
                    if p != self.provider_name and settings.key_for(p)
                ),
                None,
            )
        if requested == self.provider_name or requested not in PROVIDER_MODELS:
            return None
        return requested if settings.key_for(requested) else None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            raise RuntimeError("AgentRuntime is offline; no LLM provider available.")
        return self._provider

    @property
    def fallback(self) -> LLMProvider | None:
        return self._fallback

    @property
    def client(self) -> Any:
        """The underlying vendor SDK client. Tests patch this."""
        return self.provider.client

    async def aclose(self) -> None:
        if self._provider is not None:
            await self._provider.aclose()
        if self._fallback is not None:
            await self._fallback.aclose()


class Agent(Generic[TOut]):
    """Subclasses set ``name``, ``output_model``, ``system`` and implement ``user_prompt``."""

    name: str = "agent"
    output_model: type[BaseModel]
    system: str = ""
    max_tokens: int = 1024
    effort: str | None = None

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.model = runtime.models.get(self.name) or MODELS.get(self.name, "claude-sonnet-5")
        self.timeout_s = TIMEOUTS.get(self.name, 10.0)

    # -- subclass hooks ----------------------------------------------------------------

    def user_prompt(self, ctx: Any) -> str:
        raise NotImplementedError

    def stub(self, ctx: Any) -> TOut:
        """Deterministic output used offline, on failure, and by every test."""
        raise NotImplementedError

    def symbol_of(self, ctx: Any) -> str | None:
        return getattr(ctx, "symbol", None)

    # -- the only entry point ----------------------------------------------------------

    async def run(self, ctx: Any) -> AgentResult[TOut]:
        started = time.perf_counter()
        symbol = self.symbol_of(ctx)

        if self.runtime.offline:
            return AgentResult[Any](
                agent=self.name,
                model="stub",
                output=self.stub(ctx),
                status=AgentStatus.STUB,
                latency_ms=int((time.perf_counter() - started) * 1000),
                symbol=symbol,
            )

        try:
            result = await asyncio.wait_for(self._call(ctx), timeout=self.timeout_s)
        except TimeoutError:
            error = f"timeout after {self.timeout_s}s"
        except Exception as exc:  # noqa: BLE001 -- the loop must survive anything
            error = f"{type(exc).__name__}: {exc}"
        else:
            return self._ok(result, self.model, started, symbol)

        # Primary failed. Before giving up and using the stub, try the other vendor
        # once. A rate-limited Claude call answered by Gemini is a real opinion; the
        # stub is not.
        fallback = self.runtime.fallback
        if fallback is not None:
            fb_model = self.runtime.fallback_models.get(self.name, self.model)
            try:
                result = await asyncio.wait_for(
                    self._call_with(ctx, fallback, fb_model), timeout=self.timeout_s
                )
            except TimeoutError:
                error += f"; fallback {fb_model} timeout after {self.timeout_s}s"
            except Exception as exc:  # noqa: BLE001
                error += f"; fallback {fb_model} {type(exc).__name__}: {exc}"
            else:
                log.warning(
                    "%s: %s failed (%s); served by %s", self.name, self.model, error, fb_model
                )
                return self._ok(result, fb_model, started, symbol)

        return self._degraded(ctx, started, symbol, error)

    async def _call(self, ctx: Any) -> tuple[TOut, tuple[int, int]]:
        return await self._call_with(ctx, self.runtime.provider, self.model)

    async def _call_with(
        self, ctx: Any, provider: LLMProvider, model: str
    ) -> tuple[TOut, tuple[int, int]]:
        output, usage = await provider.complete(
            model=model,
            system=self.system,
            user=self.user_prompt(ctx),
            output_model=self.output_model,
            max_tokens=self.max_tokens,
            effort=self.effort,
        )
        return output, usage  # type: ignore[return-value]

    def _ok(
        self, result: tuple[TOut, tuple[int, int]], model: str, started: float, symbol: str | None
    ) -> AgentResult[TOut]:
        output, usage = result
        return AgentResult[Any](
            agent=self.name,
            model=model,
            output=output,
            status=AgentStatus.OK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens_in=usage[0],
            tokens_out=usage[1],
            symbol=symbol,
        )

    def _degraded(
        self, ctx: Any, started: float, symbol: str | None, error: str
    ) -> AgentResult[TOut]:
        return AgentResult[Any](
            agent=self.name,
            model=self.model,
            output=self.stub(ctx),
            status=AgentStatus.DEGRADED,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=error,
            symbol=symbol,
        )
