# BudaPilot — Budapest-Aware Autonomous Trading Agent

## Context

**The goal.** Build an agent that watches a stock's live status, reads its history, ingests
real-world market events, predicts a near-term trend from both, and executes trades inside a
hard risk envelope. Budapest-themed, demoed live today (Saturday), on test money only.

**The constraint that shapes everything.** The Budapest Stock Exchange trades Mon–Fri
09:00–17:00 CET. It is closed today. No API can produce live BÉT ticks on a Saturday — this is
a property of the world, not a tooling gap. Every design below is downstream of that fact.

**The resolution: dual track.** One asset-agnostic core, two adapter pairs.

| | Track A — Budapest | Track B — Crypto |
|---|---|---|
| Instruments | OTP.BD, MOL.BD, RICHTER.BD, MTEL.BD, ^BUX.BD | BTC/USD, ETH/USD |
| Data | Yahoo Finance via `yfinance` | Alpaca live feed (real Saturday prices) |
| Clock | **Replay** — Friday's real bars at ~60× | Real time |
| Broker | `SimBroker`, HUF ledger | **Alpaca paper account — real orders, real fills** |
| News | portfolio.hu / vg.hu / hvg.hu RSS (Hungarian) | Alpaca news feed (English) |
| Proves | The Budapest story, on real Hungarian prices | The execution path is genuinely real |

Track A gives the local narrative on real instruments and real historical prices. Track B proves
that when the agent says "submitted," an actual broker actually filled it. Neither alone answers
both questions a judge will ask.

**Decisions already locked with the user:** dual track; hybrid predictor (deterministic indicators
+ news sentiment, Claude adjudicates); FastAPI + HTMX dashboard; risk engine = position/exposure
caps + per-trade ATR stop-loss/take-profit; Devin orchestrator with parallel managed Devins.

**Non-negotiable:** paper and simulated money only. No live-broker credentials anywhere in the
repo. The live-trading adapter is deliberately left unimplemented behind a flag that raises.

---

## Architecture

A single decision loop, expressed as ports and adapters. The ports are the reason five Devins can
work in parallel without colliding.

```
             ┌──────────── loop, once per closed bar ────────────┐
             │                                                   │
  MarketDataPort ──► FeatureEngine ──┐                           │
  (bars, live or      (SMA/EMA/RSI/  │                           │
   replayed)           MACD/ATR/vol)  ├──► SignalEngine ──► RiskEngine ──► BrokerPort
                                      │    (Claude, returns      │ (caps +      (Alpaca
  NewsPort ────────► SentimentEngine ─┘     action+confidence   │  ATR stop)    paper
  (RSS / Alpaca)     (scored, cached)        +rationale)         │              or Sim)
                                                                 │
                              everything above ──► Journal (SQLite) ──► Dashboard (SSE)
```

**Ports** (`Protocol` classes, frozen at T+0:30):

- `MarketDataPort` → `AlpacaCryptoFeed` (live), `YahooBarsFeed` (BÉT), `ReplayFeed` (clock-driven wrapper)
- `BrokerPort` → `AlpacaPaperBroker` (real paper), `SimBroker` (in-process HUF ledger)
- `NewsPort` → `HungarianRssFeed`, `AlpacaNewsFeed`
- `SignalPort` → `ClaudeSignalEngine`, `StubSignalEngine` (deterministic — used by every test and as
  the runtime fallback when the API is slow or down)

Every port has a stub implementation from hour zero. The loop runs end to end on stubs before any
real adapter exists — that walking skeleton is what the parallel work attaches to.

### Why ports, specifically

Devin sessions collide when two of them edit the same file. Ports mean each module is a directory
with one owner, one test file, and one interface it may not change. Integration at T+3:00 becomes
wiring, not merging.

---

## The predictor

Deterministic features first, LLM last. The LLM never sees raw prices — it sees a compact,
computed feature bundle, which keeps the prompt small, the latency low, and the reasoning auditable.

**Features** (`features/indicators.py`): SMA(20/50), EMA(12/26), EMA cross state, RSI(14),
MACD + histogram, ATR(14), volume z-score(20), return over 1/5/20 bars, distance from 20-bar
high/low.

> **Feasibility note:** hand-roll these five indicators in pandas (~40 lines) rather than take
> `pandas-ta`. That library has a history of breaking on numpy/pandas version bumps, and a
> dependency resolution failure at hour 1 of a 5-hour build is unrecoverable. `ta` is the fallback
> if hand-rolling stalls.

**Sentiment** (`features/sentiment.py`): pull headlines per symbol, score each in one batched
Claude call to `{-1..+1}` with a one-line reason, aggregate time-decayed over 24h. Claude reads
Hungarian natively, so portfolio.hu and vg.hu headlines need no translation step — this is a real
differentiator and costs nothing.

**Adjudication** (`signal/claude_engine.py`): one call per closed bar per symbol, structured output:

```python
class Decision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float          # 0.0–1.0
    horizon_bars: int
    rationale: str             # 2-3 sentences, shown verbatim on the dashboard
    key_factors: list[str]     # ≤4, each naming a feature or headline
```

Use **`claude-sonnet-5`** for the loop — fast and cheap enough to call per bar under a 60× replay
clock. Wire **`claude-opus-5`** behind a "Deep analysis" button on the dashboard for a single
symbol on demand; it makes a good demo beat without putting Opus latency in the hot path.

A `MIN_CONFIDENCE = 0.6` constant gates action → HOLD below it. This is not the confidence/cooldown
feature that was scoped out; without some floor the agent acts on every bar, so it is load-bearing.

---

## Risk engine

Pure functions, no I/O, exhaustively unit-tested. It is the one module where a bug is expensive,
and the one module whose tests can be written before anything else exists.

```python
MAX_POSITION_PCT      = 0.10   # per symbol, of equity
MAX_TOTAL_EXPOSURE_PCT= 0.60   # all positions combined
MAX_OPEN_POSITIONS    = 4
RISK_PER_TRADE        = 0.01   # 1% of equity at risk per entry
K_STOP                = 2.0    # stop  = entry − 2.0 × ATR(14)
K_TAKE                = 3.0    # take  = entry + 3.0 × ATR(14)   → 1.5 R:R
```

Sizing: `qty = (equity × RISK_PER_TRADE) / (K_STOP × ATR)`, then clamped by `MAX_POSITION_PCT`,
then by remaining headroom under `MAX_TOTAL_EXPOSURE_PCT`. Stop and take-profit are attached at
submission time as bracket legs, so an open position stays bounded even if the agent process dies.

**Fails closed.** Any exception, any missing input, any NaN → reject the order, journal the
rejection with its reason, place no trade. Rejections render on the dashboard in amber; a visibly
blocked trade is a better demo than a silently allowed one.

Scoped out, per the user: daily drawdown kill-switch, per-symbol cooldown. The kill-switch is ~20
lines against the existing journal if a spare 15 minutes appears late — listed under Stretch, not
assumed.

---

## Requirements

### Accounts and keys — start these at T-0:00, before any code

| # | Item | Where | Notes |
|---|---|---|---|
| R1 | Alpaca **paper** account + API key/secret | alpaca.markets | Free, global, email signup. `paper-api.alpaca.markets`. **Do this first** — account provisioning is the one thing you cannot parallelize away. |
| R2 | Anthropic API key | console.anthropic.com | Sonnet 5 + Opus 5 access. |
| R3 | Devin seat with managed/parallel sessions | app.devin.ai | Confirm the plan supports managed Devins and check the ACU balance before scoping 5 sessions. |
| R4 | GitHub repo, empty, Devin integration authorized | github.com | Devin needs repo access configured *before* T+0:30, not at it. |

No key for Yahoo (`yfinance` is unauthenticated) and none for the Hungarian RSS feeds. That is
deliberate — the Budapest track has zero signup latency.

### Python dependencies

```
alpaca-py  yfinance  pandas  numpy  feedparser  anthropic
fastapi  uvicorn  jinja2  sse-starlette  pydantic  python-dotenv
pytest  pytest-asyncio  ruff
```

Persistence is `sqlite3` from the stdlib — one file, no dependency, and it handles the dashboard
reading while the loop writes. No ORM.

### Functional requirements

| ID | Requirement |
|---|---|
| FR1 | Stream or replay OHLCV bars for a configured watchlist across both tracks |
| FR2 | Fetch ≥6 months of daily and ≥5 days of 1-minute history per symbol; cache to disk |
| FR3 | Compute the indicator set on every closed bar |
| FR4 | Ingest and score news per symbol, refreshed at most every 15 min |
| FR5 | Produce a `Decision` per symbol per bar with confidence and a human-readable rationale |
| FR6 | Validate every decision against the risk engine before it can become an order |
| FR7 | Submit accepted orders with stop-loss and take-profit legs attached |
| FR8 | Journal every bar, decision, rejection, order, and fill with a timestamp |
| FR9 | Dashboard: positions, equity curve, P&L, live decision feed with rationale, risk-limit status |
| FR10 | Replay BÉT sessions at a configurable speed multiplier with pause/resume |
| FR11 | `--demo-safe` flag runs the whole system from cached data with zero network calls |

### Non-functional requirements

| ID | Requirement |
|---|---|
| NFR1 | Decision latency < 3s per symbol; the loop must keep up with a 60× replay clock |
| NFR2 | Any external failure (Yahoo, RSS, Anthropic, Alpaca) degrades to cache or stub — never crashes the loop |
| NFR3 | No secret in git; `.env` only, `.env.example` committed |
| NFR4 | The live-trading code path raises `NotImplementedError` by construction |
| NFR5 | Risk engine at 100% branch coverage; it is the only module with that bar |
| NFR6 | Cold start to running dashboard in one command |

### Explicitly out of scope

Backtesting framework and performance report. Options, shorting, margin, leverage. Portfolio
optimization. User accounts or auth. Multi-user. Deployment beyond localhost. Any real-money path.

---

## Five-hour timeline

Times are from kickoff. The gates are the important part — each one has a stated cut.

### T+0:00 → T+0:30 — Foundation. One person. No parallelism.

This half hour is the whole plan's critical path. Nothing else may start.

- Repo scaffold, `pyproject.toml`, ruff, pytest, `.env.example`
- **`contracts.py`** — every pydantic model and every `Protocol`. Then frozen.
- Stub adapter for all four ports, returning canned data
- `python -m budapilot --demo-safe` runs the loop on stubs and serves a dashboard showing fake data
- Push to `main`. Tag it. This is the walking skeleton.

> **GATE 1 (T+0:35):** contracts frozen and skeleton green? If not, **cut Track A** and run
> crypto-only for the rest of the build. Do not negotiate with this gate.

### T+0:30 → T+0:45 — Freeze the data, in parallel with Devin kickoff

Run `scripts/fetch_fixtures.py` immediately: pull 6 months daily + 5 days 1-minute for all five
BÉT symbols and both crypto pairs, plus a snapshot of every RSS feed, into `fixtures/`. Commit it.

This is the highest-leverage 15 minutes in the plan. `yfinance` rate-limits without warning and
scrapers rot; a frozen fixture set means hour 4 cannot be destroyed by Yahoo deciding it dislikes
you, and it is what makes `--demo-safe` real.

### T+0:30 → T+3:00 — Five managed Devins in parallel

Each owns one directory, one test file, one branch. **No Devin may edit `contracts.py`** — if one
believes the contract is wrong, it stops and reports to the orchestrator.

| Devin | Owns | Definition of done |
|---|---|---|
| **D1 — Data** | `data/alpaca_feed.py`, `data/yahoo_feed.py`, `data/replay.py` | Replay emits Friday's OTP bars in order on an accelerated clock; live feed yields a real BTC bar |
| **D2 — Features** | `features/indicators.py`, `features/sentiment.py`, `features/news_rss.py` | Indicators match hand-checked values on a fixture; RSS parses all three Hungarian feeds |
| **D3 — Signal** | `signal/claude_engine.py`, `signal/prompt.py`, `signal/stub.py` | Returns a valid `Decision` for a fixture bundle; falls back to stub on API error without raising |
| **D4 — Risk + Execution** | `risk/engine.py`, `broker/sim.py`, `broker/alpaca.py` | 100% branch coverage on risk; SimBroker settles a HUF round trip; Alpaca adapter fills a real paper BTC order |
| **D5 — Surface** | `web/app.py`, `web/templates/`, `journal/store.py` | Dashboard renders live from the journal over SSE; positions, equity, decision feed with rationale |

**Spec template for each session** — Cognition's own guidance is that a spec must name the repo,
the branch, the files in scope, and the explicit non-goals, and must state how the agent will know
it is done:

```
Repo: <org>/budapilot     Branch: feat/<module>     Base: main @ <skeleton tag>
In scope:   <exact file list>
Non-goals:  do not modify contracts.py, do not add dependencies outside pyproject.toml,
            do not touch another module's directory, do not modify CI
Contract:   implement <Port> from contracts.py exactly as written
Done when:  `pytest tests/test_<module>.py` passes AND <specific observable behaviour>
Report:     open a PR against main; if the contract appears wrong, STOP and report — do not edit it
```

The orchestrator Devin holds the coordinator role: it scopes, dispatches, watches for a session
drifting outside its file list, and merges in the order D4 → D1 → D2 → D3 → D5 (risk first, surface
last, so an integration failure surfaces on the least demo-visible module).

> **GATE 2 (T+2:30):** D1 and D4 merged? Those two are the spine — data in, orders out. If either
> is still open, pull its work in by hand and let that Devin finish the remainder as polish.

### T+3:00 → T+4:00 — Integration

Real keys in. Live BTC smoke test: one real paper order, confirm the fill appears in the Alpaca
dashboard and in the journal. Then a full BÉT replay of Friday's OTP session end to end. Tune the
replay multiplier so a full session plays in 6–8 minutes — long enough to watch decisions land,
short enough to sit through.

### T+4:00 → T+4:40 — Demo hardening

Verify `--demo-safe` genuinely runs with the network off (test it with wifi disabled — this is the
difference between believing it works and knowing). Dashboard polish: the rationale text is the
star, give it room. Write the demo script.

### T+4:40 → T+5:00 — Rehearsal and buffer

Run the demo twice, start to finish. Reserve this; do not spend it on features.

---

## Demo script (~5 minutes)

1. **Dashboard, live.** BTC/USD ticking on real Saturday prices. "This is a real Alpaca paper
   account and these are real market prices, right now."
2. **A live decision.** Feature bundle → Claude's rationale → risk check → order → fill. Show the
   fill in Alpaca's own UI. The execution path is real.
3. **Switch to Budapest.** OTP Bank, in HUF, Friday's real session replayed at 60×. Hungarian
   headlines from portfolio.hu feeding the sentiment score — the agent is reading Hungarian.
4. **Trip a risk limit on purpose.** Force a size above `MAX_POSITION_PCT`; the rejection renders
   in amber with its reason. "It refuses trades it is not allowed to make."
5. **Deep analysis.** Opus 5 on one symbol, full written thesis.

---

## Risks, ranked by what actually kills this build

| Risk | Mitigation |
|---|---|
| **Foundation overruns; contracts churn under five parallel sessions** | Hard 30-min box and GATE 1. Contract changes require stopping every session — that is the failure mode that ends the day. |
| **`yfinance` rate-limits or breaks at hour 4** | Freeze fixtures at T+0:45. Never fetch live during the demo. |
| **Alpaca paper account not ready when needed** | Create it at T-0:00, before the repo exists. |
| **Parallel Devins conflict** | One directory per session, explicit non-goals, `contracts.py` off-limits to all. |
| **Anthropic latency stalls the 60× replay loop** | Per-symbol decision cache, calls only on bar close, `StubSignalEngine` fallback on timeout. |
| **Venue wifi dies mid-demo** | `--demo-safe` runs everything from `fixtures/`, verified with the network physically off. |
| **Hungarian RSS layout changes** | Headlines snapshotted into fixtures; live fetch is best-effort on top of the snapshot. |

---

## Verification

- `pytest` — full suite green; `pytest --cov=risk --cov-branch` at 100% on the risk engine
- `python -m budapilot --demo-safe` with the network disabled → dashboard serves, loop runs, decisions appear
- `python -m budapilot --track crypto --live` → a real paper order appears in the Alpaca web dashboard
- `python -m budapilot --track bux --replay 2026-09-18 --speed 60` → OTP session completes, journal shows entries with stops attached
- Force an oversized order → risk rejection is journaled and rendered, no order reaches the broker
- `git grep -iE "sk-|APCA|secret"` returns nothing outside `.env.example`

---

## Stretch, only if a gate closes early

Daily drawdown kill-switch (~20 lines on the existing journal). Per-symbol cooldown. Shadow ML
classifier logged but not traded. Backtest report over the frozen fixtures.

## Sources

- [Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading) · [crypto 24/7](https://docs.alpaca.markets/us/docs/crypto-trading-1) · [market data](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Devin advanced capabilities / managed Devins](https://docs.devin.ai/work-with-devin/advanced-capabilities) · [Devin can now manage Devins](https://cognition.ai/blog/devin-can-now-manage-devins)
- [OTP.BD](https://finance.yahoo.com/quote/OTP.BD/) · [MOL.BD](https://finance.yahoo.com/quote/MOL.BD/) · [RICHTER.BD](https://finance.yahoo.com/quote/RICHTER.BD/) · [^BUX.BD](https://finance.yahoo.com/quote/%5EBUX.BD/history/)
- [BÉT hours & holidays](https://www.tradinghours.com/markets/bse-budapest) · [BUX index](https://www.bse.hu/Products-and-Services/Indices/BUX)
- [hvg.hu gazdaság RSS](https://hvg.hu/rss/gazdasag) · [Portfolio.hu](https://www.portfolio.hu/)
