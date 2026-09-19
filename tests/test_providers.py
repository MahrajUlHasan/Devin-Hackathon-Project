"""Provider swapping: Claude and Gemini must be interchangeable behind one interface.

The bar these tests hold is not "the adapter runs" -- it is that **swapping the vendor
changes nothing an agent can observe except the model id**. Same validated Pydantic
object out, same degradation behaviour on failure, same refusal to hand the risk
engine anything it did not fully validate.

The live tests at the bottom are the ones that actually matter, and they are the ones
that need a key. A mocked Gemini response proves our plumbing; only a real call proves
Gemini will honour the schemas in ``contracts.py``, which is where a provider swap
really breaks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from budapilot.agents.base import Agent, AgentRuntime
from budapilot.agents.providers import (
    AnthropicProvider,
    GeminiProvider,
    build_provider,
)
from budapilot.config import AGENT_TIERS, PROVIDER_MODELS, Settings, models_for
from budapilot.contracts import (
    AgentStatus,
    DebateCase,
    DeepAnalysis,
    Lesson,
    NewsOutput,
    RegimeOutput,
    RiskAnalystOutput,
    ScoutOutput,
    TechnicalOutput,
    TradeProposal,
)


class Tiny(BaseModel):
    verdict: str
    score: float


# ======================================================================================
# Provider resolution
# ======================================================================================


def _settings(monkeypatch, **env) -> Settings:
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings()


def test_auto_prefers_anthropic_when_both_keys_exist(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a", GEMINI_API_KEY="g-1")
    assert s.resolve_provider() == ("anthropic", "sk-a")


def test_auto_falls_to_gemini_when_only_gemini_is_configured(monkeypatch):
    s = _settings(monkeypatch, GEMINI_API_KEY="g-1")
    assert s.resolve_provider() == ("gemini", "g-1")


def test_google_api_key_is_accepted_as_an_alias(monkeypatch):
    s = _settings(monkeypatch, GOOGLE_API_KEY="g-2")
    assert s.resolve_provider() == ("gemini", "g-2")


def test_env_provider_is_honoured_over_auto(monkeypatch):
    s = _settings(
        monkeypatch, ANTHROPIC_API_KEY="sk-a", GEMINI_API_KEY="g-1", LLM_PROVIDER="gemini"
    )
    assert s.resolve_provider() == ("gemini", "g-1")


def test_explicit_request_beats_the_env_default(monkeypatch):
    s = _settings(
        monkeypatch, ANTHROPIC_API_KEY="sk-a", GEMINI_API_KEY="g-1", LLM_PROVIDER="gemini"
    )
    assert s.resolve_provider("anthropic") == ("anthropic", "sk-a")


def test_asking_for_a_provider_you_have_no_key_for_does_not_silently_switch(monkeypatch):
    """Falling back to the other vendor here would be actively harmful.

    You would be reading Gemini's output while debugging a Claude prompt. Better to
    return an empty key, which puts the runtime offline and says so on startup.
    """
    s = _settings(monkeypatch, GEMINI_API_KEY="g-1")
    assert s.resolve_provider("anthropic") == ("anthropic", "")
    assert AgentRuntime(provider="anthropic", api_key="").offline is True


def test_unknown_provider_is_rejected_loudly(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a")
    with pytest.raises(ValueError, match="unknown provider"):
        s.resolve_provider("gpt5")
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("gpt5", "k")


def test_no_keys_at_all_is_offline_not_an_error(monkeypatch):
    s = _settings(monkeypatch)
    provider, key = s.resolve_provider()
    assert key == ""
    assert provider in PROVIDER_MODELS


# ======================================================================================
# Model tier mapping
# ======================================================================================


def test_every_agent_has_a_tier_in_every_provider():
    for provider in PROVIDER_MODELS:
        mapped = models_for(provider)
        assert set(mapped) == set(AGENT_TIERS)
        assert all(mapped.values()), f"{provider} has a blank model id"


def test_the_tier_shape_is_identical_across_providers():
    """Gemini must not quietly collapse three tiers into one model.

    If it did, the cost argument in the README ("Opus only where it becomes an order")
    would be false for half the configurations, and the PM would be running on a
    headline-scoring model.
    """
    shapes = {}
    for provider in PROVIDER_MODELS:
        mapped = models_for(provider)
        # Group agents by which model they share -- the partition must match.
        by_model: dict[str, set[str]] = {}
        for agent, model in mapped.items():
            by_model.setdefault(model, set()).add(agent)
        shapes[provider] = sorted(tuple(sorted(v)) for v in by_model.values())
    assert len(set(map(str, shapes.values()))) == 1, shapes


def test_runtime_reports_gemini_models_when_gemini_is_selected():
    rt = AgentRuntime(provider="gemini", api_key="g-fake")
    assert rt.provider_name == "gemini"
    assert rt.offline is False
    assert rt.models["pm"].startswith("gemini-")
    assert rt.models["scout"] != rt.models["pm"]


def test_agents_pick_up_the_active_provider_model():
    class Probe(Agent[Tiny]):
        name = "pm"
        output_model = Tiny

    assert Probe(AgentRuntime(provider="gemini", api_key="g")).model.startswith("gemini-")
    assert Probe(AgentRuntime(provider="anthropic", api_key="sk")).model.startswith("claude-")


# ======================================================================================
# Anthropic adapter -- regression guard, this logic must not drift
# ======================================================================================


class _FakeAnthropic:
    def __init__(self, parsed=None, raise_=None):
        self.kwargs: dict = {}
        outer = self

        class _Messages:
            async def parse(self, **kwargs):
                outer.kwargs = kwargs
                if raise_:
                    raise raise_
                return SimpleNamespace(
                    parsed_output=parsed,
                    usage=SimpleNamespace(input_tokens=11, output_tokens=22),
                )

        self.messages = _Messages()

    async def close(self):
        return None


async def test_anthropic_still_sends_a_cached_system_block_and_output_format():
    p = AnthropicProvider.__new__(AnthropicProvider)
    fake = _FakeAnthropic(parsed=Tiny(verdict="ok", score=0.5))
    p._client = fake

    out, usage = await p.complete(
        model="claude-opus-5",
        system="SYS",
        user="USR",
        output_model=Tiny,
        max_tokens=512,
        effort=None,
    )

    assert out.verdict == "ok"
    assert usage == (11, 22)
    assert fake.kwargs["model"] == "claude-opus-5"
    assert fake.kwargs["output_format"] is Tiny
    # Prompt caching is most of the input cost saving on the hot-path agents.
    assert fake.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert fake.kwargs["messages"] == [{"role": "user", "content": "USR"}]
    assert "output_config" not in fake.kwargs


async def test_anthropic_passes_effort_only_when_the_agent_asks():
    p = AnthropicProvider.__new__(AnthropicProvider)
    fake = _FakeAnthropic(parsed=Tiny(verdict="ok", score=0.1))
    p._client = fake
    await p.complete(
        model="claude-opus-5",
        system="S",
        user="U",
        output_model=Tiny,
        max_tokens=64,
        effort="high",
    )
    assert fake.kwargs["output_config"] == {"effort": "high"}


async def test_anthropic_refuses_an_unparsable_response():
    p = AnthropicProvider.__new__(AnthropicProvider)
    p._client = _FakeAnthropic(parsed=None)
    with pytest.raises(ValueError, match="no parsable structured output"):
        await p.complete(
            model="m", system="s", user="u", output_model=Tiny, max_tokens=8, effort=None
        )


# ======================================================================================
# Gemini adapter
# ======================================================================================


class _FakeGemini:
    def __init__(self, *, parsed=None, text="", candidates=None):
        self.config = None
        self.model = None
        outer = self

        class _Models:
            async def generate_content(self, *, model, contents, config):
                outer.model, outer.contents, outer.config = model, contents, config
                return SimpleNamespace(
                    parsed=parsed,
                    text=text,
                    candidates=candidates,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=7, candidates_token_count=9
                    ),
                )

        self.aio = SimpleNamespace(models=_Models())


def _gemini(fake) -> GeminiProvider:
    p = GeminiProvider.__new__(GeminiProvider)
    p._client = fake
    return p


async def test_gemini_returns_the_validated_object_and_usage():
    fake = _FakeGemini(parsed=Tiny(verdict="BUY", score=0.8))
    out, usage = await _gemini(fake).complete(
        model="gemini-pro-latest",
        system="SYS",
        user="USR",
        output_model=Tiny,
        max_tokens=256,
        effort=None,
    )
    assert (out.verdict, out.score) == ("BUY", 0.8)
    assert usage == (7, 9)
    assert fake.model == "gemini-pro-latest"
    assert fake.contents == "USR"
    assert fake.config.system_instruction == "SYS"
    assert fake.config.response_mime_type == "application/json"
    assert fake.config.response_schema is Tiny


async def test_gemini_disables_thinking_by_default():
    """The failure this prevents is silent and expensive to diagnose.

    Gemini 2.5 thinks by default and draws reasoning tokens from max_output_tokens.
    At the 1024 these agents declare, the model can spend the entire budget thinking
    and return an empty candidate -- no error, just ``parsed is None`` every bar.
    """
    fake = _FakeGemini(parsed=Tiny(verdict="x", score=0.0))
    await _gemini(fake).complete(
        model="m", system="s", user="u", output_model=Tiny, max_tokens=1024, effort=None
    )
    assert fake.config.thinking_config.thinking_budget == 0
    assert fake.config.max_output_tokens == 1024


async def test_gemini_buys_headroom_for_thinking_when_effort_is_requested():
    fake = _FakeGemini(parsed=Tiny(verdict="x", score=0.0))
    await _gemini(fake).complete(
        model="m", system="s", user="u", output_model=Tiny, max_tokens=1024, effort="high"
    )
    budget = fake.config.thinking_config.thinking_budget
    assert budget > 0
    # Reasoning tokens must not come out of the answer's budget.
    assert fake.config.max_output_tokens == 1024 + budget


async def test_gemini_falls_back_to_raw_text_when_parsed_is_empty():
    fake = _FakeGemini(parsed=None, text='{"verdict": "SELL", "score": 0.3}')
    out, _ = await _gemini(fake).complete(
        model="m", system="s", user="u", output_model=Tiny, max_tokens=64, effort=None
    )
    assert out.verdict == "SELL"


async def test_gemini_raises_on_an_empty_response_rather_than_inventing_one():
    fake = _FakeGemini(
        parsed=None, text="", candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]
    )
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        await _gemini(fake).complete(
            model="m", system="s", user="u", output_model=Tiny, max_tokens=8, effort=None
        )


async def test_gemini_raises_when_there_are_no_candidates_at_all():
    with pytest.raises(ValueError, match="no candidates"):
        await _gemini(_FakeGemini(parsed=None, text="", candidates=[])).complete(
            model="m", system="s", user="u", output_model=Tiny, max_tokens=8, effort=None
        )


async def test_gemini_rejects_a_wrong_typed_parse():
    """A different model type reaching the risk engine is worse than a grey chip."""

    class Other(BaseModel):
        x: int

    with pytest.raises(TypeError, match="expected Tiny"):
        await _gemini(_FakeGemini(parsed=Other(x=1))).complete(
            model="m", system="s", user="u", output_model=Tiny, max_tokens=8, effort=None
        )


# ======================================================================================
# The contract that matters: a broken provider degrades, it does not crash the loop
# ======================================================================================


async def test_a_gemini_failure_degrades_the_agent_like_any_other():
    from budapilot.agents.technical import TechnicalAgent, TechnicalContext
    from budapilot.contracts import TechnicalOutput
    from tests.test_agents import bundle  # type: ignore[import-not-found]

    rt = AgentRuntime(provider="gemini", api_key="g-fake")
    rt._provider = _gemini(  # type: ignore[assignment]
        _FakeGemini(parsed=None, text="", candidates=[SimpleNamespace(finish_reason="SAFETY")])
    )

    result = await TechnicalAgent(rt).run(TechnicalContext(features=bundle("BTC/USD")))

    assert result.status is AgentStatus.DEGRADED
    # The provider's reason survives into the journal, not a generic "call failed".
    assert "SAFETY" in (result.error or "")
    # Still a usable, fully-valid output -- the stub's deterministic trend read.
    assert isinstance(result.output, TechnicalOutput)
    assert 0.0 <= result.output.conviction <= 1.0
    # And the card names the model that actually failed.
    assert result.model.startswith("gemini-")


# ======================================================================================
# Cross-vendor fallback: a failed Claude call is answered by Gemini, not by the stub
# ======================================================================================


class _Recorder:
    """A provider that either answers or raises, and remembers what it was asked."""

    name = "fake"

    def __init__(self, *, answer=None, raise_=None):
        self.calls: list[str] = []
        self._answer, self._raise = answer, raise_

    async def complete(self, *, model, **_):
        self.calls.append(model)
        if self._raise:
            raise self._raise
        return self._answer, (1, 1)

    async def aclose(self):
        return None


def _runtime_with_fallback(primary: _Recorder, fallback: _Recorder | None) -> AgentRuntime:
    rt = AgentRuntime(provider="anthropic", api_key="sk-fake")
    rt._provider = primary  # type: ignore[assignment]
    if fallback is not None:
        rt.fallback_name = "gemini"
        rt.fallback_models = models_for("gemini")
        rt._fallback = fallback  # type: ignore[assignment]
    return rt


def _technical_ctx():
    from budapilot.agents.technical import TechnicalContext
    from tests.test_agents import bundle  # type: ignore[import-not-found]

    return TechnicalContext(features=bundle("BTC/USD"))


def _technical_answer():
    from budapilot.contracts import Action

    return TechnicalOutput(
        symbol="BTC/USD", direction=Action.BUY, conviction=0.8, horizon_bars=6, rationale="up"
    )


async def test_primary_failure_is_served_by_the_fallback_vendor():
    from budapilot.agents.technical import TechnicalAgent

    primary = _Recorder(raise_=RuntimeError("429 rate limited"))
    fallback = _Recorder(answer=_technical_answer())
    result = await TechnicalAgent(_runtime_with_fallback(primary, fallback)).run(_technical_ctx())

    assert result.status is AgentStatus.OK
    assert result.error is None  # a real opinion was produced; this is not a degradation
    assert result.model.startswith("gemini-")  # the card names the model that answered
    assert primary.calls == [models_for("anthropic")["technical"]]
    assert fallback.calls == [models_for("gemini")["technical"]]


async def test_fallback_is_not_consulted_when_the_primary_succeeds():
    from budapilot.agents.technical import TechnicalAgent

    primary = _Recorder(answer=_technical_answer())
    fallback = _Recorder(answer=_technical_answer())
    result = await TechnicalAgent(_runtime_with_fallback(primary, fallback)).run(_technical_ctx())

    assert result.status is AgentStatus.OK
    assert result.model.startswith("claude-")
    assert fallback.calls == []


async def test_both_vendors_failing_degrades_with_both_errors_recorded():
    from budapilot.agents.technical import TechnicalAgent

    primary = _Recorder(raise_=RuntimeError("claude down"))
    fallback = _Recorder(raise_=ValueError("gemini down"))
    result = await TechnicalAgent(_runtime_with_fallback(primary, fallback)).run(_technical_ctx())

    assert result.status is AgentStatus.DEGRADED
    assert "claude down" in result.error and "gemini down" in result.error
    assert isinstance(result.output, TechnicalOutput)


async def test_no_fallback_configured_degrades_immediately():
    from budapilot.agents.technical import TechnicalAgent

    primary = _Recorder(raise_=RuntimeError("claude down"))
    result = await TechnicalAgent(_runtime_with_fallback(primary, None)).run(_technical_ctx())

    assert result.status is AgentStatus.DEGRADED
    assert "fallback" not in (result.error or "")


def test_auto_fallback_picks_the_other_vendor_only_when_it_has_a_key(monkeypatch):
    import budapilot.agents.base as base

    both = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a", GEMINI_API_KEY="g-1")
    monkeypatch.setattr(base, "settings", both)
    rt = AgentRuntime(provider="anthropic", fallback="auto")
    assert rt.fallback_name == "gemini"
    assert rt.fallback_models["pm"].startswith("gemini-")

    only_claude = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a")
    monkeypatch.setattr(base, "settings", only_claude)
    rt = AgentRuntime(provider="anthropic", fallback="auto")
    assert rt.fallback_name is None and rt.fallback is None


def test_fallback_is_off_unless_asked_for(monkeypatch):
    import budapilot.agents.base as base

    both = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a", GEMINI_API_KEY="g-1")
    monkeypatch.setattr(base, "settings", both)
    assert AgentRuntime(provider="anthropic").fallback is None
    # Asking for the primary as its own fallback is a no-op, not a second client.
    assert AgentRuntime(provider="anthropic", fallback="anthropic").fallback is None


# ======================================================================================
# Live: does Gemini actually honour our real contract schemas?
# ======================================================================================

LIVE_MODELS = [
    ScoutOutput,
    TechnicalOutput,
    NewsOutput,
    RegimeOutput,
    RiskAnalystOutput,
    TradeProposal,
    Lesson,
    DeepAnalysis,
    DebateCase,
]

live = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="needs a real Gemini key; run with GEMINI_API_KEY set",
)


@live
@pytest.mark.parametrize("model_cls", LIVE_MODELS, ids=lambda m: m.__name__)
async def test_gemini_honours_every_real_contract_schema(model_cls):
    """Mocks prove our plumbing; only this proves the swap is real.

    ``contracts.py`` uses enums, nested models, constrained floats and optional
    fields. Gemini's structured-output subset is not identical to Anthropic's, and a
    schema it silently refuses shows up as every agent degrading at once.
    """
    from budapilot.config import PROVIDER_MODELS as PM

    provider = build_provider("gemini", os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    out, usage = await provider.complete(
        model=PM["gemini"]["fast"],
        system="You are a test fixture. Emit a plausible, well-formed object.",
        user=f"Produce one example {model_cls.__name__} for BTC/USD. Any plausible values.",
        output_model=model_cls,
        max_tokens=2048,
        effort=None,
    )
    assert isinstance(out, model_cls)
    assert usage[0] > 0
