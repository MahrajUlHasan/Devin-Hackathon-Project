"""FastAPI dashboard. HTMX for fragments, SSE for the live feed.

The panel that matters is the agent grid: one card per specialist showing its latest
structured output, its model badge and its latency. A judge should be able to watch the
desk disagree -- Technical says BUY, News flags HIGH risk, Risk cuts size to 0.4x, and
the PM's rationale names who it overruled.

The dashboard only ever reads the journal. It holds no trading state, so it cannot
affect a decision, and killing it does not touch the loop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from budapilot.config import (
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MODELS,
)
from budapilot.journal.store import Journal
from budapilot.risk.session import drawdown_from

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

AGENT_ORDER = [
    ("regime", "A4 Regime"),
    ("scout", "A1 Scout"),
    ("technical", "A2 Technical"),
    ("news", "A3 News & Risk"),
    ("risk_analyst", "A5 Risk Analyst"),
    ("bull", "A9 Bull"),
    ("bear", "A10 Bear"),
    ("pm", "A6 Portfolio Manager"),
]


def create_app(journal: Journal, loop: Any = None) -> FastAPI:
    app = FastAPI(title="BudaPilot")
    app.state.journal = journal
    app.state.loop = loop

    # The header badges must name the models actually in use, not the Claude defaults.
    # A dashboard that says "claude-opus-5" while Gemini is answering is worse than no
    # badge at all -- it is a wrong answer to the first question a judge asks.
    runtime = getattr(getattr(loop, "bus", None), "runtime", None)
    agent_models: dict[str, str] = getattr(runtime, "models", None) or MODELS
    provider_name: str = getattr(runtime, "provider_name", "anthropic")
    agents_stubbed: bool = bool(getattr(runtime, "offline", False))

    def snapshot() -> dict[str, Any]:
        bar_id = journal.latest_bar_id()
        runs = journal.runs_for_bar(bar_id) if bar_id else []

        # Latest run per agent for this bar; technical/news/risk have one per symbol.
        panel: list[dict[str, Any]] = []
        for key, label in AGENT_ORDER:
            matching = [r for r in runs if r["agent"] == key]
            if not matching:
                # Cached agents (regime every 30m, news every 15m) do not run on every
                # bar. Showing an empty card would imply they failed; show the last
                # value they produced instead.
                matching = journal.query(
                    "SELECT * FROM agent_runs WHERE agent = ? ORDER BY id DESC LIMIT 3",
                    (key,),
                )
            cards = []
            for r in matching:
                try:
                    output = json.loads(r["output_json"])
                except (json.JSONDecodeError, TypeError):
                    output = {}
                cards.append(
                    {
                        "symbol": r["symbol"],
                        "status": r["status"],
                        "model": r["model"],
                        "latency_ms": r["latency_ms"],
                        "error": r["error"],
                        "output": output,
                    }
                )
            # Bull/Bear only exist when the debate is enabled; an empty card would
            # read as a failed agent rather than an absent one.
            if not cards and key in ("bull", "bear"):
                continue
            panel.append(
                {"key": key, "label": label, "model": agent_models.get(key, ""), "cards": cards}
            )

        proposals = journal.latest_proposals(8)
        for p in proposals:
            p["overrode"] = json.loads(p["overrode_json"] or "[]")

        risks = journal.latest_risk(10)
        for r in risks:
            r["checks"] = json.loads(r["checks_json"] or "{}")

        equity = journal.equity_series(200)
        positions = [p.model_dump(mode="json") for p in journal.load_protected()]

        # Heatmap: rows are symbols, columns are bars, oldest left.
        grid_rows = journal.disagreement_grid(24)
        bar_order: list[str] = []
        for r in grid_rows:
            if r["bar_id"] not in bar_order:
                bar_order.append(r["bar_id"])
        heat_symbols = sorted({r["symbol"] for r in grid_rows})
        cells = {(r["bar_id"], r["symbol"]): r for r in grid_rows}
        heatmap = {
            "bars": bar_order,
            "symbols": heat_symbols,
            "rows": [
                [cells.get((b, s)) for b in bar_order] for s in heat_symbols
            ],
        }

        state = journal.load_session()
        session = state.model_dump(mode="json") if state else None
        if state and equity:
            session["drawdown_pct"] = drawdown_from(state, equity[-1]["equity"])
            session["max_drawdown_pct"] = MAX_DAILY_DRAWDOWN_PCT

        return {
            "bar_id": bar_id,
            "panel": panel,
            "proposals": proposals,
            "risks": risks,
            "orders": journal.latest_orders(10),
            "equity": equity,
            "positions": positions,
            "lessons": [le.model_dump(mode="json") for le in journal.recent_lessons(5)],
            "degradations": journal.degradation_count(),
            "heatmap": heatmap,
            "session": session,
            "limits": {
                "max_position_pct": MAX_POSITION_PCT,
                "max_exposure_pct": MAX_TOTAL_EXPOSURE_PCT,
                "max_open": MAX_OPEN_POSITIONS,
            },
            "bar_count": getattr(loop, "bar_count", 0) if loop else 0,
            "provider": {"name": provider_name, "stubbed": agents_stubbed},
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "index.html", {"snapshot": snapshot(), "models": agent_models}
        )

    @app.get("/api/snapshot")
    async def api_snapshot() -> JSONResponse:
        return JSONResponse(snapshot())

    @app.get("/stream")
    async def stream(request: Request) -> EventSourceResponse:
        async def publisher():
            last: str | None = None
            while True:
                if await request.is_disconnected():
                    break
                data = snapshot()
                sess = data.get("session") or {}
                fingerprint = (
                    f"{data['bar_id']}-{len(data['proposals'])}-{len(data['orders'])}"
                    f"-{sess.get('halted')}-{sess.get('bar_index')}"
                )
                if fingerprint != last:
                    last = fingerprint
                    yield {"event": "update", "data": json.dumps(data)}
                await asyncio.sleep(1.0)

        return EventSourceResponse(publisher())

    @app.post("/api/deep/{symbol:path}")
    async def deep(symbol: str) -> JSONResponse:
        """A8 on demand. Opus at high effort, out of the hot path."""
        if loop is None:
            return JSONResponse({"error": "no loop attached"}, status_code=503)

        from budapilot.agents.deep import DeepContext
        from budapilot.features.indicators import compute_features

        bars = await loop.feed.get_bars(symbol, limit=200)
        if not bars:
            return JSONResponse({"error": f"no data for {symbol}"}, status_code=404)
        news = await loop.feed.get_news(symbol, limit=20)
        regime = getattr(loop.last_decision, "regime", None)

        result = await loop.bus.deep_analysis(
            DeepContext(features=compute_features(symbol, bars), news=news, regime=regime)
        )
        journal.log_agent_run(result, f"deep-{symbol}")
        return JSONResponse(
            {
                "symbol": symbol,
                "status": result.status.value,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "analysis": result.output.model_dump(mode="json"),
            }
        )

    return app
