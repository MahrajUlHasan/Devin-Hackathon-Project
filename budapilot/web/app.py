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
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MODELS,
)
from budapilot.journal.store import Journal

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

AGENT_ORDER = [
    ("regime", "A4 Regime"),
    ("scout", "A1 Scout"),
    ("technical", "A2 Technical"),
    ("news", "A3 News & Risk"),
    ("risk_analyst", "A5 Risk Analyst"),
    ("pm", "A6 Portfolio Manager"),
]


def create_app(journal: Journal, loop: Any = None) -> FastAPI:
    app = FastAPI(title="BudaPilot")
    app.state.journal = journal
    app.state.loop = loop

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
            panel.append({"key": key, "label": label, "model": MODELS.get(key, ""), "cards": cards})

        proposals = journal.latest_proposals(8)
        for p in proposals:
            p["overrode"] = json.loads(p["overrode_json"] or "[]")

        risks = journal.latest_risk(10)
        for r in risks:
            r["checks"] = json.loads(r["checks_json"] or "{}")

        equity = journal.equity_series(200)
        positions = [p.model_dump(mode="json") for p in journal.load_protected()]

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
            "limits": {
                "max_position_pct": MAX_POSITION_PCT,
                "max_exposure_pct": MAX_TOTAL_EXPOSURE_PCT,
                "max_open": MAX_OPEN_POSITIONS,
            },
            "bar_count": getattr(loop, "bar_count", 0) if loop else 0,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "index.html", {"snapshot": snapshot(), "models": MODELS}
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
                fingerprint = f"{data['bar_id']}-{len(data['proposals'])}-{len(data['orders'])}"
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
