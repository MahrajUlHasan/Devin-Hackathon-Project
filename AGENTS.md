# Working on BudaPilot

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                              # full suite
.venv/Scripts/python.exe -m pytest --cov=budapilot.risk --cov-branch --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check . --fix
.venv/Scripts/python.exe -m budapilot --demo-safe --interval 3 --max-bars 40 --port 8080
.venv/Scripts/python.exe -m budapilot --live --interval 60 --port 8080 --db budapilot-paper.db  # real data, Claude agents, Alpaca PAPER orders
```

Run `pytest -k "not honours_every_real_contract"` unless you want the 9 live Gemini
tests to spend quota; they are skipped automatically without a Gemini key.

Python 3.13 in `.venv`. Editable install: `pip install -e ".[dev]"`.

## Rules that are not negotiable

1. **`contracts.py` is the contract boundary.** Every Pydantic model and `Protocol`
   lives there. Modules implement against it and do not import each other. If the
   contract looks wrong, stop and raise it rather than editing it in passing.

2. **The whole `risk/` package holds 100% branch coverage** — `engine.py` and
   `session.py`. Adding a branch means adding a test. These are the only modules with
   that bar and the only ones where a bug costs money.

3. **No code path may reach the broker without `risk.evaluate`.** Guarded by
   `test_no_order_is_ever_placed_without_a_risk_ruling`.

4. **Agents never raise.** `Agent.run` catches everything and returns stub output with
   `status=DEGRADED`. A dead feed must not stop the loop.

5. **The PM's fallback is `DeterministicArbiter`, not a stub.** A canned HOLD on Opus
   timeout means the desk stops trading mid-demo. It degrades to a quant, not a corpse.

6. **Paper only.** `Settings.assert_paper_only()` raises if `ALPACA_PAPER=false`.

7. **Session controls gate entries, never exits.** The kill-switch and cooldowns both
   check `action is Action.BUY` first. Blocking an exit is not a risk control.

8. **Session state is persisted every bar.** If a crash could reset the drawdown
   counter, a halted day could restart and resume losing money.

## Venue facts that are easy to forget

- Alpaca crypto: `order_class` **simple only**. No bracket/OCO/OTO. Types: `market`,
  `limit`, `stop_limit`. TIF: `gtc`, `ioc`. Protective exits are ours to manage.
- Crypto is **long-only**. `SELL` = reduce/close. Enforced in the risk engine, not just
  in prompts.
- Quantities must snap to `min_trade_increment` or the venue returns 422.
- Alpaca returns crypto symbols as `BTCUSD`; the system speaks `BTC/USD`. Normalise at
  the adapter boundary only.
- The news endpoint wants `BTCUSD`, not `BTC/USD`.

## Gotchas hit during the build

- **Wilder smoothing is not `ewm(alpha=1/n)`.** RSI and ATR seed with the SMA of the
  first n periods, then smooth recursively. The naive ewm seeds at the first
  observation and disagrees by ~15 RSI points on short histories. Use `wilder_smooth`.
- **The per-symbol cap usually binds before the risk budget** at realistic crypto ATR,
  so `size_multiplier` nudges below ~0.5 have no effect on order size. `RiskDecision`
  reports which constraint bound.
- **The stop-validity check must run before sizing**, or the dust guard masks it.
- The `anthropic` SDK supports native structured output:
  `client.messages.parse(output_format=PydanticModel)` → `resp.parsed_output`. It merges
  with `output_config={"effort": "high"}` for A8.
- **Never use `hash()` for anything that must reproduce across runs.** Python randomises
  string hashing per process, so a "deterministic" offline score silently differs every
  launch. Use `zlib.crc32` (see `_stable_pseudo_score`).
- **Stubs must not be degenerate.** The offline headline scorer originally returned 0.0
  for everything, which meant no agent could disagree with another and the disagreement
  heatmap was permanently blank in `--demo-safe` — the exact mode the demo runs in. A
  stub has to be representative, not merely valid.
- The drawdown check must run *before* the decision each bar, and `advance_bar` *after*,
  or cooldowns expire one bar early.
- **Gemini's `-latest` aliases are not stable.** `gemini-flash-latest` / `gemini-pro-latest`
  returned 400/404 for a new key; the API's own error named `gemini-3.6-flash`. Pin to a
  model that `client.models.list()` actually returns for *your* key. The free tier is
  5 requests/min/model, which one bar of this desk exceeds — Gemini is a fallback, not
  a primary, unless billing is on.
- **Provider fallback is per call, inside `Agent.run`.** Primary fails or times out →
  one attempt on `runtime.fallback` with that vendor's model for the same tier → stub.
  Tests override `_call(ctx)`; the fallback goes through `_call_with(ctx, provider,
  model)` so those tests stay meaningful. `AgentRuntime(fallback=...)` is off unless
  asked for, so unit tests never build a second client from a real key in `.env`.
- **The dashboard is session-scoped.** `web/app.py` records `started_at` and every
  panel query filters `ts >= started_at`. Without this, a cached agent that skips a bar
  falls back to "last 3 rows for that agent" and shows a stale 429 from a previous
  process against a different vendor — which is exactly what was mistaken for "Google
  errors on a Claude run". Bull/Bear are omitted (not shown empty) when `--debate` is
  off.

## Dashboard API

- `GET /api/snapshot` — everything the page renders; `GET /stream` pushes it over SSE.
- `GET /api/market` — spot price per watched symbol (`feed.latest_price`, 4s cache) plus
  sparkline/high/low/features from the bars the loop already fetched (`loop._frames`).
- `GET /api/suggest` — the whole `UNIVERSE` ranked by the scout's deterministic
  `trend_score` on fresh 5m bars (60s cache), with a one-line `why`.
- `POST /api/watchlist {"symbols": [...]}` — validated against `UNIVERSE`, 1–8 symbols,
  sets `loop.symbols` for the next bar.
- `POST /api/loop/stop` / `POST /api/loop/start` — pause/resume decisions. **Paused is
  not dead**: `run_forever` still calls `_tick_stops()` every interval, so software
  stops fire and the broker-side stop_limit is untouched. `start` returns 409 once the
  loop has finished (bar or time budget) — that needs a process restart.
- None of these routes reach the broker or the risk engine.

## Time budget

`--max-minutes N` ends `run_forever` after N minutes of wall clock. `--live` without it
defaults to 120 (`DEFAULT_LIVE_MINUTES`); `--max-minutes 0` disables. Budgets are
per *run*: the dashboard's Start button relaunches a finished loop with the full
allowance (`app.state.relaunch`, set in `serve`). By default the process exits when the
loop finishes; `--serve-after-done` / `BUDAPILOT_SERVE_AFTER_DONE=true` keeps the
dashboard up instead, which is what hosting needs.

## Hosting

- `Dockerfile` + `railway.json` at the root. Python 3.13-slim, one process, `CMD` runs
  `--live --interval 300 --max-minutes 20`. Override the start command on the host to
  change flags.
- The platform's `PORT` wins over `BUDAPILOT_PORT` and flips the bind to `0.0.0.0`
  (`config.Settings`). Locally, with no `PORT`, it stays on loopback.
- SQLite must live on a **volume** (`BUDAPILOT_DB=/data/budapilot.db`, volume mounted at
  `/data`) or the kill-switch state and audit trail reset on every deploy — Rule 8.
- `BUDAPILOT_DASHBOARD_TOKEN` gates every POST (`X-Dashboard-Token` header, constant-time
  compare). Reads stay public. A public URL to a paper account with an unlocked pause
  button and an Opus-on-demand endpoint is a bill, not a demo.
- Not Vercel: serverless functions cap execution time and have no persistent disk. This
  is a long-running process with SSE and a background loop; it wants a container host.

## Adding an agent

1. Add its output model to `contracts.py`.
2. Add the model name and timeout to `config.py` (`MODELS`, `TIMEOUTS`).
3. Subclass `Agent`: set `name`, `output_model`, `system`; implement `user_prompt` and
   `stub`. The stub must return usable output, not a placeholder.
4. Wire it into `AgentBus.run_bar` at the right point in the dependency order.
5. Add it to `AGENT_ORDER` in `web/app.py` and give it a render branch in
   `templates/index.html`.
6. Add a case to `test_model_allocation_is_what_the_plan_says`.
