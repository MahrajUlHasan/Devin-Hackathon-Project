---
title: Goldfish
emoji: 🐟
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Goldfish

A multi-agent crypto trading desk. Eight specialist LLM agents argue; a deterministic
risk engine with the veto decides; an Alpaca paper account executes.

**Paper and simulated money only.** The live-trading path raises `NotImplementedError`
by construction, and this system has never been validated with real money.

## Why "Goldfish"

In March 2022 Michael Reeves published
[*I Gave My Goldfish $50,000 to Trade Stocks*](https://www.youtube.com/watch?v=USKD3vPD6ZA).
Every morning his goldfish, Frederick, was shown two random tickers. A camera tracked the
orange pixels in the tank; whichever side the fish spent more time on was the stock it
"picked", and the pick was executed through the Alpaca API. As competition, Reeves trained
a sentiment model on r/wallstreetbets and let it buy whatever the day's top post was
hyping.

After three months the fish was up about $1,007. The WallStreetBets bot was down about
$6,091. Frederick also beat the Nasdaq by roughly 13 points over the period.

The joke lands because it is not entirely a joke. A fish with no opinions, choosing at
random, inside a system that sized positions sanely and never doubled down, beat a room
full of confident humans. The fish had no edge; it also had no ego, no revenge trades and
no conviction it could not justify. Most of what went wrong on the other side of that
tank was not bad information but unbounded behaviour.

That is the design thesis here. Goldfish gives the *opinions* to a desk of specialist
models — which is where language models are genuinely good — and gives the *behaviour* to
plain, tested, deterministic code that the models cannot talk their way past. The agents
are smarter than Frederick. The point is to make sure that does not matter when they are
wrong.

## What it does

Every bar (five-minute candles by default), the desk:

1. Computes indicators for the watchlist (`FeatureEngine`, pandas, Wilder-smoothed RSI/ATR).
2. Asks a **Regime** agent what kind of market this is (cached 30 minutes).
3. Asks a **Scout** to shortlist three candidates and veto anything structurally broken.
4. Runs a **Technical Analyst** and a **News & Catalyst** analyst on each candidate in
   parallel.
5. Runs a **Risk Analyst** on each, who may cut size but cannot approve anything.
6. Optionally runs a **Bull** and a **Bear** advocate who argue one symbol without seeing
   each other, and must each concede the strongest point against their own side.
7. Hands everything to the **Portfolio Manager**, the one call on the most capable model,
   who returns exactly one `TradeProposal` and names every analyst it overruled.
8. Sends that proposal through the **risk engine**, which can only say yes or no.
9. If yes, executes on Alpaca paper and arms two independent stops.
10. When a position closes, a **Reflection** agent writes a one-line lesson that is
    injected into the PM's next prompt.

A live dashboard shows prices, every agent's structured opinion, how much they disagreed,
what the risk engine vetoed and why, orders, positions, lessons, and session analytics.
A **Deep Analysis** agent can be triggered by hand for a full written thesis on any symbol.

## Why crypto

Every stock exchange on Earth is shut at the weekend. Crypto is the only asset class
where "live market, real broker, real fill" is true on a Saturday. That is a property of
the world, not a limitation of the build — Alpaca equities would slot in behind the same
`MarketDataPort` and `BrokerPort` protocols.

## The desk

| # | Agent | Tier | Cadence | Emits |
|---|---|---|---|---|
| A1 | Scout | fast | every bar | top-3 shortlist, thesis, structural veto |
| A2 | Technical Analyst | mid | per candidate | direction, conviction, invalidation price |
| A3 | News & Catalyst | fast → mid | per candidate, 15m cache | sentiment, **news_risk**, catalysts |
| A4 | Regime Analyst | fast | 30m cache | regime label, vol percentile, BTC drift |
| A5 | Risk Analyst | mid | per candidate | size multiplier, concerns — **advisory only** |
| A6 | Portfolio Manager | deep | every bar | the single `TradeProposal` + who it overrode |
| A7 | Reflection | mid | on position close | a lesson, injected into A6's next prompt |
| A8 | Deep Analysis | deep, effort `high` | on demand | long-form written thesis |
| A9 | Bull advocate | mid | per candidate, `--debate` | the case for entering |
| A10 | Bear advocate | mid | per candidate, `--debate` | the case against |

Agents are assigned a *tier*, not a vendor. With Claude the tiers are
`claude-haiku-4-5` / `claude-sonnet-5` / `claude-opus-5`; with Gemini,
`gemini-3.6-flash` / `gemini-3.7-flash` / `gemini-3.8-flash`. Fast does the high-volume
mechanical work, mid does per-symbol reasoning, and **the deep model is spent in exactly
one place: the call that becomes an order.**

**Claude is the default. Gemini is the fallback.** If a Claude call fails or times out,
that one call is retried on Gemini's model for the same tier before the agent falls back
to its deterministic stub. The dashboard marks any card that was served by the fallback.

Deterministic, not agents: `FeatureEngine`, `RiskEngine`, `SessionRules`, `StopManager`,
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
                          A9 bull || A10 bear         (--debate)
                                        |
                              A6 PM arbitrates (deep)
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

**Agents never raise.** Every agent catches everything and returns its stub output with
`status=DEGRADED`. A dead news feed does not stop the desk. The PM's stub is not a canned
HOLD but a deterministic arbiter, so an Opus timeout degrades the desk to a quant, not a
corpse.

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
and no order. Crypto is long-only, so `SELL` always means reduce or close; this is
enforced in the engine, not just in prompts, because a model will confidently propose a
trade the venue rejects with a 422.

**Drawdown is measured from the session peak, not the open.** A desk that is up 8% and
gives back 5% has lost control of the day just as much as one that started flat. A halt
**blocks new entries; it does not flatten** — selling into whatever caused the drawdown
is how a bad day becomes a catastrophic one. The halt is sticky until the UTC day
boundary, and both it and the cooldowns are persisted every bar so **a crash cannot reset
the daily loss limit**. Neither a halt nor a cooldown can ever block an exit.

**Alpaca crypto has no bracket orders**, so stops cannot be attached to the entry.
`StopManager` runs two redundant layers — a precise software stop and a resting
`stop_limit` at the broker that survives the process dying. When both could fire, the
sequence is always cancel → confirm → close; an unconfirmed cancel means we do *not*
close, because the alternative double-sells the position.

One honest note on sizing: at typical five-minute crypto ATR (~1% of price), the 10%
per-symbol cap binds before the 1% risk budget does, so real risk per trade lands *below*
the nominal figure. Every `RiskDecision` names which constraint actually bound.

## The dashboard

- **Live market tape** — spot price, change, sparkline, RSI/trend/ATR tags per symbol.
- **Watchlist** — toggle any of 14 Alpaca pairs on or off; changes apply on the next bar.
- **Suggest best** — the whole universe ranked by the same deterministic trend score the
  Scout uses, with a one-line reason. Not an LLM; you can check its arithmetic.
- **The desk** — one compact card per agent, expandable to the full structured output,
  with model badge, latency and status. Cached agents show their last value; agents that
  did not run say why.
- **Disagreement heatmap** — unanimity reads as confidence when it is usually just
  correlation, so the spread between technical, news, risk and PM is scored per symbol
  per bar and drawn as a grid. The interesting trades are the amber ones.
- **Analytics** — per-agent latency and status mix, token totals, PM decision split, top
  veto reasons, trade win rate.
- **Start / Stop** — pauses decisions. Stops keep watching prices; the broker-side
  `stop_limit` never stopped. A finished run can be restarted with a fresh budget.

## Running it

### Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows      (source .venv/bin/activate on Unix)
pip install -e ".[dev]"
cp .env.example .env              # then fill in whatever keys you have
```

Then open http://127.0.0.1:8000 after any of the commands below.

### The modes

Every mode is chosen by which of three things is real: the **data**, the **agents**, and
the **broker**. The default at each step is the safer option; you opt *in* to the real
one.

| Command | Data | Agents | Broker | Keys needed |
|---|---|---|---|---|
| `python -m budapilot --demo-safe` | frozen fixtures / synthetic | deterministic stubs | simulated | none |
| `python -m budapilot --stub-agents` | live Alpaca | deterministic stubs | simulated | none (Alpaca optional) |
| `python -m budapilot` | live Alpaca | real LLMs | simulated | Anthropic or Gemini |
| `python -m budapilot --live` | live Alpaca | real LLMs | **Alpaca paper** | Anthropic/Gemini + Alpaca |

**`--demo-safe`** makes zero network calls; a test monkeypatches `socket.connect` to
prove it. This is the mode to run on a plane, in a judging room with bad wifi, or when
you just want to see the machine move. Run `python scripts/fetch_fixtures.py` once with
Alpaca keys to freeze real market data so the replay is real rather than synthetic.

**`--stub-agents`** exercises the whole pipeline on live prices without spending a token.
Useful for checking Alpaca connectivity and the broker plumbing.

**Default (no flags)** is the honest test of the agents: real market, real models, but
every order goes to an in-process simulator. Nothing touches the paper account.

**`--live`** submits real orders to your Alpaca **paper** account. `ALPACA_PAPER=true` is
required and enforced. Without `--max-minutes` it stops after 120 minutes by default.

### Modifiers

| Flag | Effect |
|---|---|
| `--provider anthropic\|gemini` | Which vendor runs the agents. Default: `LLM_PROVIDER` in `.env`, else Claude if its key is present, else Gemini. |
| `--no-fallback` | Do not retry a failed call on the other vendor. |
| `--debate` | Run the A9/A10 bull-vs-bear round before the PM. Roughly doubles tokens and latency per candidate. |
| `--interval N` | Seconds between bars. Default 300. `--interval 20` for a fast demo (bars will run back-to-back, since a bar takes ~30 s of agent calls). |
| `--max-bars N` | Stop after N bars. |
| `--max-minutes N` | Stop after N minutes of wall clock. `0` disables. Default 120 with `--live`. |
| `--serve-after-done` | Keep the dashboard up after the loop's budget is spent instead of exiting. For hosting. Also `BUDAPILOT_SERVE_AFTER_DONE=true`. |
| `--cash N` | Starting equity for the simulated broker. Default 100 000. |
| `--db PATH` | Journal location. Use a fresh file per experiment; the dashboard is session-scoped but the audit trail is not. |
| `--host`, `--port` | Bind address. A platform-injected `PORT` wins and flips the bind to `0.0.0.0`. |
| `-v` | Debug logging, including every HTTP call. |

### Recipes

```bash
# Offline demo, fast bars, stops on its own after 40 bars
python -m budapilot --demo-safe --interval 3 --max-bars 40

# Real market, real Claude agents, no orders, one bar, then exit
python -m budapilot --interval 3 --max-bars 1 --db check.db

# Full paper session: 60-second bars for 45 minutes, with the debate
python -m budapilot --live --interval 60 --max-minutes 45 --debate --db paper.db

# Force Gemini, no Claude fallback
python -m budapilot --provider gemini --no-fallback

# What the hosted demo runs (see Dockerfile)
python -m budapilot --live --interval 20 --max-minutes 20 --serve-after-done
```

### Environment

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Alpaca **paper** credentials. Market data works without them at lower rate limits; `--live` requires them. |
| `ALPACA_PAPER` | Must be `true`. `false` makes the broker refuse to start. |
| `ANTHROPIC_API_KEY` | Claude. Default provider when present. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini. Fallback when Claude is primary, primary when Claude is absent. |
| `LLM_PROVIDER` | `anthropic`, `gemini`, or `auto` (default). |
| `CLAUDE_MODEL_{FAST,MID,DEEP}`, `GEMINI_MODEL_{FAST,MID,DEEP}` | Pin a model per tier. |
| `BUDAPILOT_DB`, `BUDAPILOT_HOST`, `BUDAPILOT_PORT` | Runtime defaults for `--db`, `--host`, `--port`. |
| `BUDAPILOT_SERVE_AFTER_DONE` | `true` keeps the dashboard up after the loop finishes. |
| `BUDAPILOT_DASHBOARD_TOKEN` | If set, the dashboard's buttons (start/stop, watchlist, deep analysis) require it as an `X-Dashboard-Token` header. Reads stay public. |

Secrets go in `.env` only (`.env.example` is committed). No key is ever logged.

### Hosting

A `Dockerfile` and `railway.json` are included. The container runs
`--live --interval 20 --max-minutes 20` with the dashboard staying up afterwards; press
**Restart loop** for another run. Mount a volume at `/data` (the default `BUDAPILOT_DB`
points there) or the kill-switch state resets on every deploy. The platform's `PORT` is
honoured automatically. Set `BUDAPILOT_DASHBOARD_TOKEN` on a public URL — an unlocked
Deep Analysis button is an Opus bill anyone can run up.

This is a long-running process with SQLite, SSE and a background loop. It needs a
container host (Railway, Fly, a VPS, a Hugging Face Docker Space), not serverless
functions.

## Dashboard API

| Route | What |
|---|---|
| `GET /api/snapshot` | Everything the page renders. `GET /stream` pushes the same over SSE. |
| `GET /api/market` | Spot price per watched symbol plus sparkline, range and indicators. |
| `GET /api/suggest` | The universe ranked by trend score, with reasons. |
| `POST /api/watchlist` | `{"symbols": [...]}`, validated against the universe, 1–8 symbols. Applies next bar. |
| `POST /api/loop/stop`, `POST /api/loop/start` | Pause / resume decisions. `start` on a finished loop begins a fresh run. |
| `POST /api/deep/BTC/USD` | Run A8 on a symbol. |

No route reaches the broker or the risk engine.

## Verification

```bash
pytest -q -k "not honours_every_real_contract"         # 249 tests, offline
pytest --cov=budapilot.risk --cov-branch               # 100% branch coverage on risk/
pytest tests/test_stops.py -q                          # the cancel-then-close race
pytest tests/test_session.py -q                        # kill-switch and cooldowns
pytest tests/test_web.py -q                            # dashboard, session scoping, token gate
ruff check .
```

The nine deselected tests call Gemini for real to prove it honours every schema in
`contracts.py`; they skip without a key and spend quota with one.

## Layout

```
budapilot/
  contracts.py        every Pydantic model and Protocol -- the contract boundary
  config.py           watchlist, tiers, timeouts, risk limits, environment
  agents/             one file per agent + providers.py (Claude, Gemini) + bus.py
  features/           indicators; Wilder smoothing done properly
  risk/               engine.py (the veto) and session.py (kill-switch, cooldowns)
  execution/          broker_alpaca.py, broker_sim.py, stops.py, engine.py
  data/               alpaca_crypto.py, fixtures.py
  journal/            SQLite audit trail
  web/                FastAPI app + one HTML file
tests/                249 of them
AGENTS.md             rules of the codebase and every gotcha hit during the build
```

The Python package is still called `budapilot` — the project was renamed after the code
was written, and a package rename buys nothing but a noisier diff.

## Built with Devin

The codebase was written with [Devin](https://devin.ai) across several sessions for the
LaunchLoop hackathon. `AGENTS.md` is the file it left for the next agent: the
non-negotiable rules, the venue facts that bite, and every mistake it made and fixed.

Frederick the goldfish is not affiliated with this project and has, as far as anyone
knows, retired from active trading.
