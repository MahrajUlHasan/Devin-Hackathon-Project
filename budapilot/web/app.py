"""FastAPI dashboard. SSE for the journal feed, plain JSON for the live market tape.

The panel that matters is the agent grid: one card per specialist showing its latest
structured output, its model badge and its latency. A judge should be able to watch the
desk disagree -- Technical says BUY, News flags HIGH risk, Risk cuts size to 0.4x, and
the PM's rationale names who it overruled.

The dashboard reads the journal and the loop's last market frames. The one thing it may
*write* is the watchlist -- which symbols the desk looks at next bar. It cannot place an
order, size a position or touch the risk engine; those paths do not exist here.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from budapilot.agents.scout import rank_symbols
from budapilot.config import (
    MAX_CANDIDATES,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MODELS,
    UNIVERSE,
    WATCHLIST,
    settings,
)
from budapilot.contracts import FeatureBundle, utcnow
from budapilot.features.indicators import compute_features
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

# Agents that legitimately skip bars because their output is cached. For these, an
# empty card would imply failure, so the dashboard shows the last value they produced
# *this session*. Anything else that did not run this bar is shown as exactly that.
CACHED_AGENTS = {"regime": "cached 30m", "news": "cached 15m"}

MAX_WATCHLIST = 8
MARKET_TTL_S = 4.0
SUGGEST_TTL_S = 60.0


def _feature_summary(f: FeatureBundle) -> dict[str, Any]:
    return {
        "trend_score": f.trend_score,
        "rsi_14": f.rsi_14,
        "atr_pct": f.atr_pct,
        "ret_5": f.ret_5,
        "ret_20": f.ret_20,
        "ema_cross": f.ema_cross,
        "vol_zscore_20": f.vol_zscore_20,
        "dist_from_high_20": f.dist_from_high_20,
        "tradeable": f.is_tradeable(),
    }


def _why(f: FeatureBundle) -> str:
    """One deterministic sentence on why a symbol ranks where it does."""
    bits: list[str] = []
    if f.ema_cross == "GOLDEN":
        bits.append("golden cross")
    elif f.ema_cross == "DEATH":
        bits.append("death cross")
    if f.rsi_14 is not None:
        if 55 <= f.rsi_14 <= 70:
            bits.append(f"RSI {f.rsi_14:.0f} strong, not stretched")
        elif f.rsi_14 > 80:
            bits.append(f"RSI {f.rsi_14:.0f} overbought")
        elif f.rsi_14 < 30:
            bits.append(f"RSI {f.rsi_14:.0f} oversold")
    if f.ret_20 is not None:
        bits.append(f"{f.ret_20 * 100:+.1f}% over 20 bars")
    if f.vol_zscore_20 is not None and f.vol_zscore_20 > 1.5:
        bits.append("volume surging")
    if f.atr_pct is not None and f.atr_pct > 0.02:
        bits.append("high volatility")
    if not f.is_tradeable():
        bits.append("insufficient history")
    return "; ".join(bits) or "flat tape"


def create_app(journal: Journal, loop: Any = None) -> FastAPI:
    app = FastAPI(title="BudaPilot")
    app.state.journal = journal
    app.state.loop = loop

    # Everything the dashboard shows is scoped to this process's lifetime. The journal
    # is an audit trail that outlives sessions; a card must not show last week's error.
    started_at = utcnow().isoformat()

    # The header badges must name the models actually in use, not the Claude defaults.
    # A dashboard that says "claude-opus-5" while Gemini is answering is worse than no
    # badge at all -- it is a wrong answer to the first question a judge asks.
    runtime = getattr(getattr(loop, "bus", None), "runtime", None)
    agent_models: dict[str, str] = getattr(runtime, "models", None) or MODELS
    provider_name: str = getattr(runtime, "provider_name", "anthropic")
    fallback_name: str | None = getattr(runtime, "fallback_name", None)
    fallback_models: dict[str, str] = getattr(runtime, "fallback_models", None) or {}
    agents_stubbed: bool = bool(getattr(runtime, "offline", False))
    debate_on: bool = bool(getattr(getattr(loop, "bus", None), "enable_debate", False))

    market_cache: dict[str, Any] = {"at": 0.0, "data": None}
    suggest_cache: dict[str, Any] = {"at": 0.0, "data": None}

    def _authorise(request: Request) -> JSONResponse | None:
        """Gate the buttons when a dashboard token is configured.

        Reads are open -- the point of a hosted demo is that people can watch. Writes
        (pause, watchlist, an Opus call that costs money) need the token, sent as
        ``X-Dashboard-Token``. No token configured means no gate, which is fine on
        localhost and is what ``--demo-safe`` runs with.
        """
        token = settings.dashboard_token
        if not token:
            return None
        if hmac.compare_digest(request.headers.get("x-dashboard-token", ""), token):
            return None
        return JSONResponse({"error": "dashboard token required"}, status_code=401)

    def _card(r: dict[str, Any], *, cached: bool = False) -> dict[str, Any]:
        try:
            output = json.loads(r["output_json"])
        except (json.JSONDecodeError, TypeError):
            output = {}
        return {
            "symbol": r["symbol"],
            "status": r["status"],
            "model": r["model"],
            "latency_ms": r["latency_ms"],
            "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"],
            "error": r["error"],
            "output": output,
            "cached": cached,
            "served_by_fallback": bool(
                fallback_name and r["model"] in fallback_models.values()
            ),
        }

    def _panel(bar_id: str | None, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        had_candidates = any(r["agent"] == "technical" for r in runs)
        panel: list[dict[str, Any]] = []
        for key, label in AGENT_ORDER:
            if key in ("bull", "bear") and not debate_on:
                continue
            matching = [r for r in runs if r["agent"] == key]
            cards = [_card(r) for r in matching]
            note = ""
            if not cards and bar_id:
                if key in CACHED_AGENTS:
                    cards = [
                        _card(r, cached=True)
                        for r in journal.last_run_for_agent(key, started_at)
                    ]
                    note = CACHED_AGENTS[key]
                elif key in ("technical", "risk_analyst", "bull", "bear") and not had_candidates:
                    note = "no candidates this bar"
                else:
                    note = "did not run this bar"
            panel.append(
                {
                    "key": key,
                    "label": label,
                    "model": agent_models.get(key, ""),
                    "fallback_model": fallback_models.get(key, ""),
                    "cards": cards,
                    "note": note,
                }
            )
        return panel

    def snapshot() -> dict[str, Any]:
        bar_id = journal.latest_bar_id()
        runs = journal.runs_for_bar(bar_id) if bar_id else []
        # A bar from a previous process is history, not "the latest bar".
        if runs and runs[0]["ts"] < started_at:
            bar_id, runs = None, []

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
            "rows": [[cells.get((b, s)) for b in bar_order] for s in heat_symbols],
        }

        state = journal.load_session()
        session = state.model_dump(mode="json") if state else None
        if state and equity:
            session["drawdown_pct"] = drawdown_from(state, equity[-1]["equity"])
            session["max_drawdown_pct"] = MAX_DAILY_DRAWDOWN_PCT

        last_bar_at = getattr(loop, "last_bar_at", None) if loop else None
        interval_s = getattr(loop, "interval_s", None) if loop else None
        next_bar_in = None
        if last_bar_at is not None and interval_s:
            next_bar_in = max(0.0, interval_s - (utcnow() - last_bar_at).total_seconds())
        deadline = getattr(loop, "deadline", None) if loop else None
        budget_left = (
            max(0.0, (deadline - utcnow()).total_seconds()) if deadline is not None else None
        )

        return {
            "bar_id": bar_id,
            "panel": _panel(bar_id, runs),
            "proposals": proposals,
            "risks": risks,
            "orders": journal.latest_orders(10),
            "equity": equity,
            "positions": positions,
            "lessons": [le.model_dump(mode="json") for le in journal.recent_lessons(5)],
            "degradations": journal.degradation_count(since=started_at),
            "heatmap": heatmap,
            "session": session,
            "analytics": journal.analytics(started_at),
            "limits": {
                "max_position_pct": MAX_POSITION_PCT,
                "max_exposure_pct": MAX_TOTAL_EXPOSURE_PCT,
                "max_open": MAX_OPEN_POSITIONS,
                "max_candidates": MAX_CANDIDATES,
            },
            "bar_count": getattr(loop, "bar_count", 0) if loop else 0,
            "interval_s": interval_s,
            "next_bar_in_s": next_bar_in,
            "running": bool(getattr(loop, "running", False)) if loop else False,
            "paused": bool(getattr(loop, "paused", False)) if loop else False,
            "budget_left_s": budget_left,
            "watchlist": list(getattr(loop, "symbols", WATCHLIST)) if loop else list(WATCHLIST),
            "universe": UNIVERSE,
            "locked": bool(settings.dashboard_token),
            "provider": {
                "name": provider_name,
                "fallback": fallback_name,
                "stubbed": agents_stubbed,
                "debate": debate_on,
                "broker": type(getattr(loop, "broker", None)).__name__ if loop else "",
            },
        }

    # -- market tape ---------------------------------------------------------------------

    async def market() -> dict[str, Any]:
        now = time.monotonic()
        if market_cache["data"] is not None and now - market_cache["at"] < MARKET_TTL_S:
            return market_cache["data"]
        if loop is None:
            return {"symbols": [], "ts": utcnow().isoformat()}

        symbols = list(loop.symbols)
        frames: dict[str, list] = getattr(loop, "_frames", {})
        held = {p.symbol for p in journal.load_protected()}

        # Spot prices are the one fresh call; everything else comes from the bars the
        # loop already fetched for the last decision.
        spots = await asyncio.gather(
            *(loop.feed.latest_price(s) for s in symbols), return_exceptions=True
        )

        rows = []
        for symbol, spot in zip(symbols, spots, strict=True):
            bars = frames.get(symbol) or []
            closes = [b.close for b in bars]
            last = closes[-1] if closes else 0.0
            price = float(spot) if isinstance(spot, int | float) and spot > 0 else last
            first = closes[0] if closes else price
            summary = _feature_summary(compute_features(symbol, bars)) if bars else None
            rows.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "change_pct": ((price / first) - 1.0) * 100 if first else None,
                    "high": max((b.high for b in bars), default=None),
                    "low": min((b.low for b in bars), default=None),
                    "volume": sum(b.volume for b in bars) if bars else None,
                    "bars": len(bars),
                    "spark": closes[-60:],
                    "features": summary,
                    "held": symbol in held,
                }
            )
        data = {"symbols": rows, "ts": utcnow().isoformat()}
        market_cache.update(at=now, data=data)
        return data

    async def suggest() -> dict[str, Any]:
        """Rank the whole universe the same way the scout does, and say why."""
        now = time.monotonic()
        if suggest_cache["data"] is not None and now - suggest_cache["at"] < SUGGEST_TTL_S:
            return suggest_cache["data"]
        if loop is None:
            return {"ranked": [], "recommended": [], "ts": utcnow().isoformat()}

        bar_lists = await asyncio.gather(
            *(loop.feed.get_bars(s, limit=200) for s in UNIVERSE), return_exceptions=True
        )
        features = [
            compute_features(s, bars)
            for s, bars in zip(UNIVERSE, bar_lists, strict=True)
            if not isinstance(bars, Exception) and bars
        ]
        ranked_bundles = sorted(features, key=lambda f: f.trend_score, reverse=True)
        top = [f.symbol for f in rank_symbols(features, top_n=MAX_CANDIDATES) if f.trend_score > 0]
        ranked = [
            {
                "symbol": f.symbol,
                "close": f.close,
                "why": _why(f),
                "recommended": f.symbol in top,
                "in_watchlist": f.symbol in loop.symbols,
                **_feature_summary(f),
            }
            for f in ranked_bundles
        ]
        data = {"ranked": ranked, "recommended": top, "ts": utcnow().isoformat()}
        suggest_cache.update(at=now, data=data)
        return data

    # -- routes --------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "index.html", {"snapshot": snapshot(), "models": agent_models}
        )

    @app.get("/api/snapshot")
    async def api_snapshot() -> JSONResponse:
        return JSONResponse(snapshot())

    @app.get("/api/market")
    async def api_market() -> JSONResponse:
        return JSONResponse(await market())

    @app.get("/api/suggest")
    async def api_suggest() -> JSONResponse:
        return JSONResponse(await suggest())

    @app.get("/api/watchlist")
    async def get_watchlist() -> JSONResponse:
        symbols = list(loop.symbols) if loop else list(WATCHLIST)
        return JSONResponse({"symbols": symbols, "universe": UNIVERSE, "max": MAX_WATCHLIST})

    @app.post("/api/watchlist")
    async def set_watchlist(request: Request) -> JSONResponse:
        """The one write the dashboard is allowed: which symbols the desk looks at.

        Takes effect on the next bar. Open positions in a removed symbol keep their
        stops -- the StopManager tracks positions, not the watchlist.
        """
        if loop is None:
            return JSONResponse({"error": "no loop attached"}, status_code=503)
        if (denied := _authorise(request)) is not None:
            return denied
        try:
            body = await request.json()
            wanted = [str(s).upper().strip() for s in body.get("symbols", [])]
        except (ValueError, AttributeError):
            return JSONResponse({"error": "body must be {\"symbols\": [...]}"}, status_code=400)

        unknown = [s for s in wanted if s not in UNIVERSE]
        if unknown:
            return JSONResponse({"error": f"not in universe: {unknown}"}, status_code=400)
        deduped = list(dict.fromkeys(wanted))
        if not 1 <= len(deduped) <= MAX_WATCHLIST:
            return JSONResponse(
                {"error": f"pick between 1 and {MAX_WATCHLIST} symbols"}, status_code=400
            )
        loop.symbols = deduped
        market_cache["data"] = None
        return JSONResponse({"symbols": deduped, "applies": "next bar"})

    @app.post("/api/loop/{command}")
    async def loop_control(command: str, request: Request) -> JSONResponse:
        """Pause or resume decisions. Protective exits keep running either way.

        This is deliberately not a kill: a paused desk still watches its stops, and
        resuming does not need a restart. If the loop has *finished* (bar or time
        budget spent) and the process exposes a relauncher, ``start`` begins a fresh run
        with the same budget; otherwise it is a 409 and the process must be restarted.
        """
        if loop is None:
            return JSONResponse({"error": "no loop attached"}, status_code=503)
        if (denied := _authorise(request)) is not None:
            return denied
        relaunched = False
        if command == "stop":
            loop.pause()
        elif command == "start":
            if not loop.running:
                relaunch = getattr(request.app.state, "relaunch", None)
                if relaunch is None:
                    return JSONResponse(
                        {"error": "loop has finished (budget spent); restart the process"},
                        status_code=409,
                    )
                loop.paused = False
                relaunch()
                relaunched = True
                # run_forever sets running=True on its first tick; report the intent.
                return JSONResponse({"running": True, "paused": False, "relaunched": True})
            loop.resume()
        else:
            return JSONResponse({"error": "command must be start or stop"}, status_code=400)
        return JSONResponse(
            {"running": loop.running, "paused": loop.paused, "relaunched": relaunched}
        )

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
                    f"-{sess.get('halted')}-{sess.get('bar_index')}-{len(data['positions'])}"
                    f"-{data['degradations']}-{','.join(data['watchlist'])}"
                    f"-{data['running']}-{data['paused']}"
                )
                if fingerprint != last:
                    last = fingerprint
                    yield {"event": "update", "data": json.dumps(data)}
                await asyncio.sleep(1.0)

        return EventSourceResponse(publisher())

    @app.post("/api/deep/{symbol:path}")
    async def deep(symbol: str, request: Request) -> JSONResponse:
        """A8 on demand. Opus at high effort, out of the hot path."""
        if loop is None:
            return JSONResponse({"error": "no loop attached"}, status_code=503)
        if (denied := _authorise(request)) is not None:
            return denied

        from budapilot.agents.deep import DeepContext

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
