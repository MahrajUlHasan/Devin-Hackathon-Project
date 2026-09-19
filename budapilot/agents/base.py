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
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from budapilot.config import MODELS, TIMEOUTS, settings
from budapilot.contracts import AgentResult, AgentStatus

TOut = TypeVar("TOut", bound=BaseModel)


class AgentRuntime:
    """Shared Anthropic client. ``offline=True`` forces every agent to its stub."""

    def __init__(self, *, offline: bool = False, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else settings.anthropic_key
        self.offline = offline or not key
        self._client: Any = None
        if not self.offline:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=key)

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("AgentRuntime is offline; no Anthropic client available.")
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


class Agent(Generic[TOut]):
    """Subclasses set ``name``, ``output_model``, ``system`` and implement ``user_prompt``."""

    name: str = "agent"
    output_model: type[BaseModel]
    system: str = ""
    max_tokens: int = 1024
    effort: str | None = None

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.model = MODELS.get(self.name, "claude-sonnet-5")
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
            return self._degraded(ctx, started, symbol, f"timeout after {self.timeout_s}s")
        except Exception as exc:  # noqa: BLE001 -- the loop must survive anything
            return self._degraded(ctx, started, symbol, f"{type(exc).__name__}: {exc}")

        output, usage = result
        return AgentResult[Any](
            agent=self.name,
            model=self.model,
            output=output,
            status=AgentStatus.OK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens_in=usage[0],
            tokens_out=usage[1],
            symbol=symbol,
        )

    async def _call(self, ctx: Any) -> tuple[TOut, tuple[int, int]]:
        output_config: dict[str, Any] | None = {"effort": self.effort} if self.effort else None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": self.user_prompt(ctx)}],
            "output_format": self.output_model,
        }
        if output_config:
            kwargs["output_config"] = output_config

        resp = await self.runtime.client.messages.parse(**kwargs)
        parsed = resp.parsed_output
        if parsed is None:
            raise ValueError("model returned no parsable structured output")
        usage = (
            getattr(resp.usage, "input_tokens", 0) or 0,
            getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return parsed, usage

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
