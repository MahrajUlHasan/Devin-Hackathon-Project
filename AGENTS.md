# Working on BudaPilot

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                              # full suite
.venv/Scripts/python.exe -m pytest --cov=budapilot.risk --cov-branch --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check . --fix
.venv/Scripts/python.exe -m budapilot --demo-safe --interval 3 --max-bars 40 --port 8080
```

Python 3.13 in `.venv`. Editable install: `pip install -e ".[dev]"`.

## Rules that are not negotiable

1. **`contracts.py` is the contract boundary.** Every Pydantic model and `Protocol`
   lives there. Modules implement against it and do not import each other. If the
   contract looks wrong, stop and raise it rather than editing it in passing.

2. **The risk engine holds 100% branch coverage.** Adding a branch means adding a test.
   It is the only module with that bar and the only one where a bug costs money.

3. **No code path may reach the broker without `risk.evaluate`.** Guarded by
   `test_no_order_is_ever_placed_without_a_risk_ruling`.

4. **Agents never raise.** `Agent.run` catches everything and returns stub output with
   `status=DEGRADED`. A dead feed must not stop the loop.

5. **The PM's fallback is `DeterministicArbiter`, not a stub.** A canned HOLD on Opus
   timeout means the desk stops trading mid-demo. It degrades to a quant, not a corpse.

6. **Paper only.** `Settings.assert_paper_only()` raises if `ALPACA_PAPER=false`.

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

## Adding an agent

1. Add its output model to `contracts.py`.
2. Add the model name and timeout to `config.py` (`MODELS`, `TIMEOUTS`).
3. Subclass `Agent`: set `name`, `output_model`, `system`; implement `user_prompt` and
   `stub`. The stub must return usable output, not a placeholder.
4. Wire it into `AgentBus.run_bar` at the right point in the dependency order.
5. Add it to `AGENT_ORDER` in `web/app.py` and give it a render branch in
   `templates/index.html`.
6. Add a case to `test_model_allocation_is_what_the_plan_says`.
