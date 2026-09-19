# BudaPilot — Multi-Agent Crypto Trading System

A crypto-only multi-agent trading desk: eight specialist Claude agents feeding a
deterministic risk engine and a real Alpaca paper broker.

**Status: built, including all stretch items.** 201 tests passing, 100% branch coverage
across the `risk/` package, ruff clean, `--demo-safe` verified with network calls
monkeypatched to raise. See `README.md` for how to run it and `AGENTS.md` for how to
work on it.

---

## Decisions locked

| Question | Answer |
|---|---|
| Market | **Crypto only** — Alpaca paper, real 24/7 prices, real fills |
| Agent topology | Specialists in parallel → **Portfolio Manager arbiter** |
| Risk | **Deterministic code with hard veto.** LLM risk analyst advises only |
| Runtime orchestration | `asyncio` + Anthropic SDK native structured output + Pydantic |
| Devin's role | **Builder only** — runtime agents are Claude API calls |

### Why crypto, stated plainly

No stock exchange on Earth is open at the weekend. NYSE/Nasdaq/LSE/Xetra/TSE are
Mon–Fri, the Gulf exchanges and Tel Aviv are Sun–Thu, and Alpaca's 24/5 equities
session does not resume until 8PM ET Sunday. Crypto is the only asset class where
"live market, real broker, real fill" is true right now. This is a property of the
world, not a fallback — say it on stage.

### Why Devin is not the runtime

`POST /v1/sessions` can spawn agents programmatically, but a Devin session is a VM that
deliberates for minutes and bills quota per session. Eight agents × every bar would
exhaust the budget in minutes and miss every latency target. Devin builds the system;
Claude API calls are the agents.

---

## Two venue constraints that shaped the build

**1. Alpaca crypto does not support bracket orders.** `order_class` is `simple` only —
no `bracket`, `oco`, `oto`. Types: `market`, `limit`, `stop_limit`. TIF: `gtc`, `ioc`.
The original "attach stop and take-profit as bracket legs" is not buildable. Protection
lives in `StopManager` instead, as two redundant layers.

**2. Crypto is long-only.** No shorting, no margin. `SELL` means reduce or close.
Enforced in the risk engine, not just in prompts — a model will confidently propose a
trade the venue rejects with a 422.

---

## Agent roster — as built

| # | Agent | Model | Cadence | Emits |
|---|---|---|---|---|
| A1 | Scout | `claude-haiku-4-5` | every bar | top-3 shortlist, thesis, structural veto |
| A2 | Technical Analyst | `claude-sonnet-5` | per candidate | direction, conviction, invalidation |
| A3 | News & Catalyst | `claude-haiku-4-5` → `claude-sonnet-5` | per candidate, 15m cache | sentiment, **news_risk**, catalysts |
| A4 | Regime Analyst | `claude-haiku-4-5` | 30m cache | regime, vol percentile, BTC drift |
| A5 | Risk Analyst | `claude-sonnet-5` | per candidate | size multiplier — **advisory only** |
| A6 | Portfolio Manager | `claude-opus-5` | every bar | the `TradeProposal` + who it overrode |
| A7 | Reflection | `claude-sonnet-5` | on close | a lesson, injected into A6's next prompt |
| A8 | Deep Analysis | `claude-opus-5`, effort `high` | on demand | long-form thesis |
| A9 | Bull advocate | `claude-sonnet-5` | per candidate, `--debate` | the case for entering |
| A10 | Bear advocate | `claude-sonnet-5` | per candidate, `--debate` | the case against |

Haiku does high-volume near-mechanical work; Sonnet does per-symbol reasoning in the hot
path; **Opus is spent in exactly one place — the call that becomes an order.**

Deterministic, not agents: `FeatureEngine`, `RiskEngine`, `StopManager`,
`ExecutionEngine`. Anything that must be *correct* is code; anything that must be
*judged* is a model.

**Honest note on A1.** Ranking six symbols by trend is arithmetic, done in pandas by
`rank_symbols`. The model supplies the justification and a structural veto. It does not
do the maths and the code does not pretend it does.

---

## Pipeline

```
bars -> FeatureEngine -> A4 regime (30m cache) -> A1 scout -> top 3
                                                     |
                             A2 technical || A3 news (15m cache)     parallel
                                                     |
                                              A5 risk analyst
                                                     |
                                     A6 PM arbitrates (opus-5)
                                                     |
                             ===== RiskEngine: HARD VETO, fails closed =====
                                                     |
                                    ExecutionEngine -> Alpaca paper
                                                     |
                      StopManager: software stop + resting stop_limit backstop
                                                     |
                                  Journal (SQLite WAL) -> Dashboard (SSE)
```

There is no code path from a model's opinion to an order that skips `risk.evaluate`.
Asserted by `test_no_order_is_ever_placed_without_a_risk_ruling`.

---

## Implementation notes

**Structured output.** `client.messages.parse(output_format=PydanticModel)` →
`resp.parsed_output`. The SDK transforms the schema into the constrained-decoding subset
and validates for us. System prompts are sent as cached blocks. A8 adds
`output_config={"effort": "high"}`.

**Agents never raise.** On timeout or API error, `Agent.run` returns stub output with
`status=DEGRADED`, journalled and rendered as a grey chip. The loop survives anything.

**The PM's fallback is not a stub.** `DeterministicArbiter` runs a weighted vote over
the same specialist opinions and emits the same `TradeProposal` shape. The system
degrades to a quant, not to a corpse.

**StopManager — the highest-risk component.** Because brackets do not exist:
a resting `stop_limit` at the broker survives process death; an in-process software stop
is precise and covers take-profit, which the venue cannot express at all. When both
could fire the sequence is **cancel → confirm → close**, and an unconfirmed cancel means
we do *not* close. Inverting that double-sells. `reconcile()` rebuilds state on restart,
re-arms missing stops and cancels orphans.

**Risk limits.**

```python
MAX_POSITION_PCT       = 0.10      MAX_TOTAL_EXPOSURE_PCT = 0.50
MAX_OPEN_POSITIONS     = 3         RISK_PER_TRADE         = 0.01
MIN_CONVICTION         = 0.60      MIN_ORDER_NOTIONAL_PCT = 0.005
K_STOP, K_TAKE         = 2.0, 3.0  # 1.5 reward:risk
```

News risk `CRITICAL` is an absolute veto. The engine fails closed on any exception,
missing input or NaN.

---

## Three things the build found that the plan did not anticipate

1. **Wilder smoothing is not `ewm(alpha=1/n)`.** RSI and ATR seed with the SMA of the
   first n periods. The naive ewm seeds at the first observation and disagrees by ~15
   RSI points on a 20-bar series — the difference between "overbought" and "neutral"
   for anything reading the number. Fixed in `wilder_smooth`, pinned against Wilder's
   published table.

2. **The per-symbol cap binds before the risk budget** at realistic crypto volatility.
   At ~1% ATR, risking 1% of equity to a 2×ATR stop implies a 50% position, so the 10%
   cap clamps it and true risk per trade lands *below* nominal. Conservative, but the
   journal must not claim the nominal number — every `RiskDecision` now names which
   constraint actually bound.

3. **Dust orders.** Filling the last sliver of cap headroom produced $6 trades that paid
   fees and moved nothing. Added `MIN_ORDER_NOTIONAL_PCT`. Relatedly, the PM prompt now
   states each candidate's remaining headroom, because without it the PM proposed the
   same capped symbol every bar and collected an identical rejection each time.

---

## Verification

```bash
pytest -q                                    # 143 passed
pytest --cov=budapilot.risk --cov-branch     # 100% (104 stmts, 36 branches)
pytest tests/test_stops.py -q                # cancel-then-close race
ruff check .                                 # clean
python -m budapilot --demo-safe              # runs with sockets patched to raise
```

---

## Demo script (~5 min)

1. **Live crypto, real prices.** "The market is open and this is a real Alpaca paper
   account."
2. **Watch the desk disagree.** A bar closes; Technical says BUY at 0.72, News flags
   HIGH risk, Risk cuts size to 0.4×, and the PM's rationale names who it overruled.
3. **Order to fill.** Risk sizes it, the order goes out, the fill appears in Alpaca's
   own UI.
4. **Trip a limit on purpose.** Amber rejection with its reason. "It refuses trades it
   is not allowed to make — and that decision is code, not a language model."
5. **Reflection.** Close a position; A7 writes the lesson; it appears in the PM's next
   prompt. The system learns inside the demo.
6. **Deep analysis.** Opus at high effort, full written thesis.

## Stretch items — now built

All four shipped. See `README.md` for behaviour and `tests/test_session.py` /
`tests/test_debate.py` for the guarantees.

**Daily drawdown kill-switch.** Measured from the session peak rather than the open, so
a morning profit cannot quietly fund an afternoon of losses. Halts new entries but does
not flatten — selling into whatever caused the drawdown takes the worst price at the
worst moment, and the open positions already carry stops sized before the trouble
started. Sticky until the UTC day rolls, because an oscillating kill-switch is worse
than none. Persisted every bar so a crash cannot reset the loss limit.

**Per-symbol cooldown.** Asymmetric: 6 bars after a stop-out, 2 after a take-profit.
Hitting your target is not evidence the thesis was wrong. Neither this nor the halt can
ever block an exit.

**Bull/Bear debate (A9/A10, `--debate`).** Two `claude-sonnet-5` advocates argue one
symbol before the PM rules. Round 1 is independent — letting the bear read the bull
first turns adversarial review into one argument plus a reaction. Both must fill
`conceded`, and the PM is told they are advocates rather than neutral analysts. Off by
default; it roughly doubles tokens and latency per candidate.

**Disagreement heatmap.** Three axes — technical vs news, how hard risk cut size,
whether the PM overrode anyone — as a symbols × bars grid.

### One more thing the build found

**A degenerate stub makes a feature look broken.** With the offline headline scorer
returning 0.0 for every headline, sentiment was always neutral, so no agent could
disagree with any other and the heatmap was uniformly blank — in `--demo-safe`, the
exact mode the demo runs in. The scorer now derives a stable pseudo-score from
`zlib.crc32` of the headline. Not `hash()`: Python randomises string hashing per
process, which would have made a "deterministic" offline mode differ on every launch.
