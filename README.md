---
title: BudaPilot
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# BudaPilot

A multi-agent crypto trading desk. Eight specialist Claude agents produce opinions, a
deterministic risk engine holds the veto, and an Alpaca paper account executes.

**Paper and simulated money only.** The live-trading path raises `NotImplementedError`
by construction.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"
cp .env.example .env                                 # optional; works without keys

python -m budapilot --demo-safe                      # no network at all
```

Then open http://127.0.0.1:8000.

## Why crypto

Every stock exchange on Earth is shut at the weekend. NYSE, Nasdaq, LSE, Xetra and TSE
run Monday to Friday; the Gulf exchanges and Tel Aviv run Sunday to Thursday; Alpaca's
24/5 US equities session does not resume until 8PM ET Sunday. Crypto is the only asset
class where "live market, real broker, real fill" is true right now. That is a property
of the world, not a limitation of the build.

## The desk

| # | Agent | Model | Cadence | Emits |
|---|---|---|---|---|
| A1 | Scout | `claude-haiku-4-5` | every bar | top-3 shortlist, thesis, structural veto |
| A2 | Technical Analyst | `claude-sonnet-5` | per candidate | direction, conviction, invalidation price |
| A3 | News & Catalyst | `claude-haiku-4-5` → `claude-sonnet-5` | per candidate, 15m cache | sentiment, **news_risk**, catalysts |
| A4 | Regime Analyst | `claude-haiku-4-5` | 30m cache | regime label, vol percentile, BTC drift |
| A5 | Risk Analyst | `claude-sonnet-5` | per candidate | size multiplier, concerns — **advisory only** |
| A6 | Portfolio Manager | `claude-opus-5` | every bar | the single `TradeProposal` + who it overrode |
| A7 | Reflection | `claude-sonnet-5` | on position close | a lesson, injected into A6's next prompt |
| A8 | Deep Analysis | `claude-opus-5`, effort `high` | on demand | long-form written thesis |
| A9 | Bull advocate | `claude-sonnet-5` | per candidate, `--debate` | the case for entering |
| A10 | Bear advocate | `claude-sonnet-5` | per candidate, `--debate` | the case against |

Haiku does the high-volume, near-mechanical work. Sonnet does per-symbol reasoning in
the hot path. **Opus is spent in exactly one place: the call that becomes an order.**

Deterministic, not agents: `FeatureEngine`, `RiskEngine`, `StopManager`,
`ExecutionEngine`. The split is the point — anything that must be *correct* is code,
anything that must be *judged* is a model.

## The pipeline

```
bars -> FeatureEngine -> A4 regime -> A1 scout -> top 3
                                        |
                        A2 technical || A3 news      (parallel)
                                        |
                                   A5 risk analyst
                                        |
                              A6 PM arbitrates (opus)
                                        |
                        ===== RiskEngine: HARD VETO =====
                                        |
                          ExecutionEngine -> Alpaca paper
                                        |
                        StopManager: software stop + resting stop_limit
                                        |
                              Journal (SQLite) -> Dashboard (SSE)
```

There is no code path from a model's opinion to an order that skips `risk.evaluate`.
`tests/test_loop.py::test_no_order_is_ever_placed_without_a_risk_ruling` asserts it.

## Two venue constraints that shaped the design

**Alpaca crypto has no bracket orders.** `order_class` is `simple` only — no bracket,
no OCO, no OTO; order types are `market`, `limit`, `stop_limit`; TIF is `gtc` or `ioc`.
Stops cannot be attached to the entry, so `StopManager` runs two redundant layers: a
precise in-process software stop, and a resting `stop_limit` at the broker that survives
the process dying. When both could fire, the sequence is always **cancel → confirm the
cancel → close**, and an unconfirmed cancel means we do *not* close. Inverting that
order double-sells the position.

**Crypto is long-only.** No shorting, no margin. `SELL` always means reduce or close.
This is enforced in the risk engine, not just in the prompts, because a model will
confidently propose a trade the venue rejects with a 422.

## Risk

```python
MAX_POSITION_PCT       = 0.10   # per symbol
MAX_TOTAL_EXPOSURE_PCT = 0.50
MAX_OPEN_POSITIONS     = 3
RISK_PER_TRADE         = 0.01
MIN_CONVICTION         = 0.60   # below this -> forced HOLD
MIN_ORDER_NOTIONAL_PCT = 0.005  # dust guard
K_STOP, K_TAKE         = 2.0, 3.0   # 1.5 reward:risk
MAX_DAILY_DRAWDOWN_PCT = 0.05   # kill-switch
COOLDOWN_BARS          = 6      # after a stop-out; 2 after a take-profit
```

News risk `CRITICAL` is an absolute veto regardless of conviction. The engine **fails
closed**: any exception, missing input or NaN produces a rejection with a stated reason
and no order.

### Kill-switch and cooldowns

Drawdown is measured **from the session peak, not the open**. A desk that is up 8% and
gives back 5% has lost control of the day just as much as one that started flat — from
the open, a morning profit would silently fund an afternoon of losses.

A halt **blocks new entries; it does not flatten**. Force-liquidating into whatever
caused the drawdown is how a bad day becomes a catastrophic one: you sell the bottom and
pay the spread to do it. Open positions keep the stops they were given before the
trouble started. `HALT_FLATTENS_POSITIONS = True` inverts this.

The halt is sticky — recovering above the threshold does not un-halt, because an
oscillating kill-switch is worse than none. It clears at the UTC day boundary. Both the
halt and the cooldowns are persisted to the journal every bar, so **a crash cannot be
used to reset the daily loss limit**.

Cooldowns are asymmetric: 6 bars after a stop-out, 2 after a take-profit. Hitting your
target is not evidence the thesis was wrong; being stopped out is evidence you were
early or wrong, and re-entering immediately is how one bad read becomes five.

Neither a halt nor a cooldown can block an **exit**. Refusing to let the desk out of a
position is not a risk control, it is a trap.

### Bull/Bear debate (opt-in)

```bash
python -m budapilot --debate
```

Two advocates are assigned a side and argue one symbol before the PM rules. In round 1
they do **not** see each other: otherwise whoever goes first anchors the other and the
second "case" is really just a reaction. `DEBATE_ROUNDS = 2` adds a rebuttal round where
they do.

Each advocate must fill `conceded` — the strongest point against their own side that
they accept. An advocate who concedes nothing is cheerleading, and the PM is told to
discount them accordingly. The PM is also told these are advocates rather than neutral
analysts, so it judges the arguments and not the confidence numbers.

Off by default: it roughly doubles tokens and latency per candidate.

### Disagreement heatmap

Unanimity reads as confidence when it is often just correlation, so the spread is
measured rather than eyeballed. Each candidate is scored on three axes — technical
direction vs news sentiment, how hard the risk analyst cut size, and whether the PM
overrode anyone — and rendered as a symbols × bars grid on the dashboard.

One honest note on sizing: at typical 5-minute crypto ATR (~1% of price), risking 1% of
equity to a 2×ATR stop implies a position around 50% of equity, so the 10% per-symbol
cap binds first and real risk per trade lands *below* the nominal 1%. The risk budget
only becomes the binding constraint when ATR exceeds ~5% of price. Every `RiskDecision`
names which constraint actually bound, so the journal never claims a risk the desk is
not taking. Pinned by `test_position_cap_binds_at_realistic_crypto_volatility`.

## Running it

```bash
python -m budapilot --demo-safe        # fixtures + stub agents + sim broker, no network
python -m budapilot                    # live crypto data, real agents, sim broker
python -m budapilot --live             # ... and real orders to an Alpaca PAPER account
python -m budapilot --stub-agents      # real data, deterministic agents (no API spend)
python -m budapilot --debate           # add the A9/A10 adversarial round

python scripts/fetch_fixtures.py       # freeze real data so --demo-safe is real
```

Useful flags: `--interval 3` (seconds per bar, for a fast demo), `--max-bars 40`,
`--cash`, `--port`, `-v`.

## Verification

```bash
pytest -q                                              # 201 tests
pytest --cov=budapilot.risk --cov-branch               # 100% branch coverage
pytest tests/test_stops.py -q                          # the cancel-then-close race
pytest tests/test_session.py -q                        # kill-switch and cooldowns
ruff check .
```

`tests/test_loop.py::test_fixture_feed_makes_no_network_calls` monkeypatches
`socket.connect` to raise, so the `--demo-safe` guarantee is asserted rather than
assumed.

## Configuration

Everything lives in `budapilot/config.py`: watchlist, bar size, per-agent model and
timeout, cache TTLs, risk limits. `contracts.py` holds every Pydantic model and
`Protocol` and is the contract boundary — modules implement against it rather than
importing each other.

Secrets go in `.env` only (`.env.example` is committed). No key is ever logged.
