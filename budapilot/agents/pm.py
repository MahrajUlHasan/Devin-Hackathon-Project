"""A6 -- Portfolio Manager (arbiter). Model: claude-opus-5.

The only place Opus is spent, because this is the only call that becomes an order. One
call per bar: read every specialist's structured opinion plus the portfolio and the
lessons A7 has written, emit exactly one `TradeProposal`.

The `overrode` field is not decoration. Requiring the PM to name which analyst it
overruled and why is what turns a black-box decision into something a human can audit
on the dashboard -- and it is the moment the demo is built around.

Its fallback is `DeterministicArbiter`, not a stub. See arbiter.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from budapilot.agents.arbiter import deterministic_arbitrate
from budapilot.agents.base import Agent
from budapilot.agents.fmt import portfolio_block
from budapilot.config import (
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_CONVICTION,
)
from budapilot.contracts import (
    Lesson,
    PortfolioState,
    RegimeOutput,
    SymbolOpinions,
    TradeProposal,
)

SYSTEM = f"""You are the Portfolio Manager on a crypto trading desk. Your specialists have \
each filed a structured opinion. You make the single call.

You emit exactly one TradeProposal per bar. Most bars that proposal is HOLD -- a desk \
that trades every five minutes is a desk that pays fees for noise.

LONG ONLY. This is a hard property of the venue, not a preference. BUY opens or adds to a \
long. SELL reduces or closes an existing long. There is no shorting. Proposing SELL with \
no open position is an error the risk engine will reject.

How to weigh your specialists:
- The Technical Analyst gives direction and conviction. It is the primary input, but it \
sees only the chart.
- The News Analyst gives sentiment and a risk rating. news_risk CRITICAL is an absolute \
veto -- propose HOLD. news_risk HIGH means you need a clearly better-than-usual technical \
setup to act.
- The Risk Analyst gives a sizing multiplier. You may lower it. Raising it above what the \
Risk Analyst advised requires a specific, stated reason.
- Lessons from closed trades are your own desk's history. Weigh them; they are evidence, \
not rules.
- Where a BULL and BEAR case are shown, they are advocates who were assigned a side, not \
neutral analysts. Judge the arguments, not the confidence numbers. Pay particular \
attention to what each side conceded -- an advocate who concedes nothing has told you \
less than one who concedes something real.

`conviction` below {MIN_CONVICTION:.2f} will be forced to HOLD downstream, so do not \
propose an action you cannot honestly rate above it.

`overrode` is mandatory whenever you disagree with a specialist. One string per \
disagreement, naming the analyst and the reason. Example: "risk_analyst: raised size from \
0.4 to 0.6, the ATR spike is a single print and has already normalised". If you agree \
with everyone, leave it empty.

`rationale` is two to four sentences, written for a human reading it live. Say what you \
are doing and what would change your mind.

Portfolio limits enforced downstream: max {MAX_POSITION_PCT:.0%} of equity per symbol, \
max {MAX_TOTAL_EXPOSURE_PCT:.0%} gross exposure, max {MAX_OPEN_POSITIONS} open positions. \
Each candidate below is annotated with its remaining headroom. Proposing a BUY in a \
symbol already at its cap wastes the bar -- the trade will be rejected. Either pick a \
different symbol or propose HOLD."""


@dataclass
class PMContext:
    opinions: list[SymbolOpinions]
    portfolio: PortfolioState
    regime: RegimeOutput | None = None
    lessons: list[Lesson] = field(default_factory=list)


class PMAgent(Agent[TradeProposal]):
    name = "pm"
    output_model = TradeProposal
    system = SYSTEM
    max_tokens = 1200

    def symbol_of(self, ctx: PMContext) -> str | None:
        return None

    def user_prompt(self, ctx: PMContext) -> str:
        parts: list[str] = []
        if ctx.regime:
            parts += [
                f"REGIME: {ctx.regime.regime.value} "
                f"(realized vol pct {ctx.regime.realized_vol_pct:.0f}, "
                f"BTC dominance drift {ctx.regime.btc_dominance_drift * 100:+.2f}%)",
                f"  {ctx.regime.rationale}",
                "",
            ]

        parts.append("CANDIDATES")
        for op in ctx.opinions:
            f, t, n, r = op.features, op.technical, op.news, op.risk
            position = ctx.portfolio.positions.get(op.symbol)
            held = position is not None

            # Tell the PM where the caps already bind. Without this it proposes the
            # same capped symbol every bar and collects an identical rejection.
            if held and ctx.portfolio.equity > 0:
                pct = position.market_value / ctx.portfolio.equity
                room = MAX_POSITION_PCT - pct
                capacity = (
                    f"AT THE {MAX_POSITION_PCT:.0%} CAP - a BUY here will be rejected"
                    if room <= 0.005
                    else f"{pct:.1%} of equity, {room:.1%} of headroom left"
                )
            else:
                capacity = "no position"

            parts += [
                f"\n--- {op.symbol} ({capacity}) ---",
                f"  price {f.close:.4f} | rsi {f.rsi_14 or 0:.1f} | "
                f"atr {(f.atr_pct or 0) * 100:.2f}% of px | trend {f.trend_score:+.2f}",
                f"  TECHNICAL: {t.direction.value} conv={t.conviction:.2f} "
                f"horizon={t.horizon_bars}b invalidation={t.invalidation}",
                f"    {t.rationale}",
                f"    factors: {'; '.join(t.key_factors) if t.key_factors else 'none given'}",
                f"  NEWS: sentiment={n.sentiment:+.2f} risk={n.news_risk.value} "
                f"({n.headline_count} headlines)",
                f"    catalysts: {', '.join(n.catalysts) if n.catalysts else 'none'}",
                f"    {n.rationale}",
                f"  RISK: size_multiplier={r.size_multiplier:.2f} "
                f"stop={r.suggested_stop_atr:.1f}xATR",
                f"    concerns: {'; '.join(r.concerns) if r.concerns else 'none'}",
                f"    {r.rationale}",
            ]

            if op.debate:
                for case in (op.debate.bull, op.debate.bear):
                    parts += [
                        f"  {case.side} (confidence {case.confidence:.2f}): "
                        f"{case.strongest_point}",
                        *[f"    - {c}" for c in case.claims],
                    ]
                    if case.rebuttal:
                        parts.append(f"    rebuttal: {case.rebuttal}")
                    if case.conceded:
                        parts.append(f"    concedes: {case.conceded}")

        parts += ["", "PORTFOLIO", portfolio_block(ctx.portfolio)]

        if ctx.lessons:
            parts += ["", "LESSONS FROM CLOSED TRADES (most recent first)"]
            parts += [
                f"  - {le.symbol} {le.outcome_pct:+.2f}% ({le.exit_reason.value}): {le.lesson}"
                for le in ctx.lessons
            ]

        parts += ["", "Make the call."]
        return "\n".join(parts)

    def stub(self, ctx: PMContext) -> TradeProposal:
        """Not a canned HOLD -- the deterministic arbiter, so the desk keeps trading."""
        return deterministic_arbitrate(ctx.opinions, ctx.portfolio)
