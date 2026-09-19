"""SQLite journal. stdlib sqlite3, WAL mode, no ORM.

WAL matters here: the dashboard reads while the trading loop writes, and without it the
reader blocks the writer. One file, one dependency-free store, and it is also the
system's audit trail.

``agent_runs`` is the table the demo is built on -- every agent's structured output,
model, latency and status, keyed by bar. Everything the dashboard shows is a query
against it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime
from typing import Any

from budapilot.contracts import (
    AgentResult,
    BarDecision,
    Disagreement,
    ExitReason,
    Lesson,
    Order,
    ProtectedPosition,
    RiskDecision,
    SessionState,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_id TEXT, ts TEXT, symbol TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_id TEXT, symbol TEXT, agent TEXT, model TEXT,
    output_json TEXT, latency_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER,
    status TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_id TEXT, symbol TEXT, action TEXT, conviction REAL,
    size_multiplier REAL, stop_atr REAL, rationale TEXT, overrode_json TEXT
);
CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_id TEXT, symbol TEXT, action TEXT, approved INTEGER,
    qty REAL, notional REAL, stop_px REAL, take_px REAL, reason TEXT, checks_json TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, order_id TEXT, client_order_id TEXT, symbol TEXT, side TEXT,
    qty REAL, type TEXT, status TEXT, filled_qty REAL, filled_avg_price REAL,
    stop_price REAL, intent TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, order_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    qty REAL, entry REAL, stop_px REAL, take_px REAL,
    opened_at TEXT, broker_stop_order_id TEXT, closing INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, symbol TEXT, outcome_pct REAL, exit_reason TEXT, lesson TEXT, tags_json TEXT
);
CREATE TABLE IF NOT EXISTS degradations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_id TEXT, agent TEXT, model TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, equity REAL, cash REAL, exposure_pct REAL
);
CREATE TABLE IF NOT EXISTS disagreements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, bar_id TEXT, symbol TEXT, technical_signed REAL, news_sentiment REAL,
    risk_multiplier REAL, pm_action TEXT, pm_conviction REAL, overrode_count INTEGER,
    score REAL
);
-- Single-row table. The kill-switch must not be resettable by a crash.
CREATE TABLE IF NOT EXISTS session_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_disagreements_bar ON disagreements(bar_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_bar ON agent_runs(bar_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_ts ON agent_runs(ts DESC);
CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_decisions(ts DESC);
"""


def _iso(ts: datetime | None = None) -> str:
    return (ts or utcnow()).isoformat()


_VOLATILE = [
    re.compile(r",?\s*'request_id':\s*'[^']*'"),
    re.compile(r"Please retry in [\d.]+s\.?"),
    re.compile(r"'retryDelay':\s*'[^']*'"),
    re.compile(r"req_[A-Za-z0-9]+"),
    re.compile(r"timeout after [\d.]+s"),
]


def _error_kind(error: str) -> str:
    """Collapse one error string to the part that identifies its *kind*."""
    kind = error
    for pat in _VOLATILE:
        kind = pat.sub(lambda m: "timeout" if m.group(0).startswith("timeout") else "", kind)
    return kind[:300]


class Journal:
    """Thread-safe enough for one writer (the loop) and many readers (the dashboard)."""

    def __init__(self, path: str = "budapilot.db") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _write(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # -- writers -----------------------------------------------------------------------

    def log_agent_run(self, result: AgentResult[Any], bar_id: str) -> None:
        self._write(
            "INSERT INTO agent_runs (ts, bar_id, symbol, agent, model, output_json, "
            "latency_ms, tokens_in, tokens_out, status, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                _iso(),
                bar_id,
                result.symbol,
                result.agent,
                result.model,
                result.output.model_dump_json(),
                result.latency_ms,
                result.tokens_in,
                result.tokens_out,
                result.status.value,
                result.error,
            ),
        )
        if result.error:
            self._write(
                "INSERT INTO degradations (ts, bar_id, agent, model, error) VALUES (?,?,?,?,?)",
                (_iso(), bar_id, result.agent, result.model, result.error),
            )

    def log_decision(self, decision: BarDecision) -> None:
        for result in decision.results:
            self.log_agent_run(result, decision.bar_id)
        if decision.disagreements:
            self.log_disagreements(decision.disagreements, decision.bar_id)
        if decision.proposal:
            p = decision.proposal
            self._write(
                "INSERT INTO proposals (ts, bar_id, symbol, action, conviction, "
                "size_multiplier, stop_atr, rationale, overrode_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    _iso(decision.ts),
                    decision.bar_id,
                    p.symbol,
                    p.action.value,
                    p.conviction,
                    p.size_multiplier,
                    p.stop_atr,
                    p.rationale,
                    json.dumps(p.overrode),
                ),
            )

    def log_disagreements(self, items: list[Disagreement], bar_id: str) -> None:
        for d in items:
            self._write(
                "INSERT INTO disagreements (ts, bar_id, symbol, technical_signed, "
                "news_sentiment, risk_multiplier, pm_action, pm_conviction, "
                "overrode_count, score) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    _iso(),
                    bar_id,
                    d.symbol,
                    d.technical_signed,
                    d.news_sentiment,
                    d.risk_multiplier,
                    d.pm_action.value,
                    d.pm_conviction,
                    d.overrode_count,
                    d.score,
                ),
            )

    # -- session state (kill-switch and cooldowns must survive a crash) ----------------

    def save_session(self, state: SessionState) -> None:
        self._write(
            "INSERT INTO session_state (id, state_json, updated_at) VALUES (1,?,?) "
            "ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json, "
            "updated_at=excluded.updated_at",
            (state.model_dump_json(), _iso()),
        )

    def load_session(self) -> SessionState | None:
        rows = self.query("SELECT state_json FROM session_state WHERE id = 1")
        if not rows:
            return None
        try:
            return SessionState.model_validate_json(rows[0]["state_json"])
        except Exception:  # noqa: BLE001 -- a corrupt row must not block startup
            return None

    def log_risk(self, decision: RiskDecision, bar_id: str) -> None:
        self._write(
            "INSERT INTO risk_decisions (ts, bar_id, symbol, action, approved, qty, "
            "notional, stop_px, take_px, reason, checks_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                _iso(),
                bar_id,
                decision.symbol,
                decision.action.value,
                int(decision.approved),
                decision.qty,
                decision.notional,
                decision.stop_px,
                decision.take_px,
                decision.reason,
                json.dumps(decision.checks),
            ),
        )

    def log_order(self, order: Order, intent: str = "entry") -> None:
        self._write(
            "INSERT INTO orders (ts, order_id, client_order_id, symbol, side, qty, type, "
            "status, filled_qty, filled_avg_price, stop_price, intent) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _iso(order.submitted_at),
                order.id,
                order.client_order_id,
                order.symbol,
                order.side.value,
                order.qty,
                order.type.value,
                order.status.value,
                order.filled_qty,
                order.filled_avg_price,
                order.stop_price,
                intent,
            ),
        )
        if order.filled_qty > 0 and order.filled_avg_price:
            self._write(
                "INSERT INTO fills (ts, order_id, symbol, side, qty, price) VALUES (?,?,?,?,?,?)",
                (
                    _iso(),
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.filled_qty,
                    order.filled_avg_price,
                ),
            )

    def log_equity(self, equity: float, cash: float, exposure_pct: float) -> None:
        self._write(
            "INSERT INTO equity_curve (ts, equity, cash, exposure_pct) VALUES (?,?,?,?)",
            (_iso(), equity, cash, exposure_pct),
        )

    def add_lesson(self, lesson: Lesson) -> None:
        self._write(
            "INSERT INTO lessons (ts, symbol, outcome_pct, exit_reason, lesson, tags_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                _iso(lesson.ts),
                lesson.symbol,
                lesson.outcome_pct,
                lesson.exit_reason.value,
                lesson.lesson,
                json.dumps(lesson.tags),
            ),
        )

    # -- protected positions (StopManager state) ---------------------------------------

    def upsert_protected(self, pos: ProtectedPosition) -> None:
        self._write(
            "INSERT INTO positions (symbol, qty, entry, stop_px, take_px, opened_at, "
            "broker_stop_order_id, closing) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, entry=excluded.entry, "
            "stop_px=excluded.stop_px, take_px=excluded.take_px, "
            "broker_stop_order_id=excluded.broker_stop_order_id, closing=excluded.closing",
            (
                pos.symbol,
                pos.qty,
                pos.entry,
                pos.stop_px,
                pos.take_px,
                _iso(pos.opened_at),
                pos.broker_stop_order_id,
                int(pos.closing),
            ),
        )

    def delete_protected(self, symbol: str) -> None:
        self._write("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def load_protected(self) -> list[ProtectedPosition]:
        return [
            ProtectedPosition(
                symbol=r["symbol"],
                qty=r["qty"],
                entry=r["entry"],
                stop_px=r["stop_px"],
                take_px=r["take_px"],
                opened_at=datetime.fromisoformat(r["opened_at"]),
                broker_stop_order_id=r["broker_stop_order_id"],
                closing=bool(r["closing"]),
            )
            for r in self.query("SELECT * FROM positions")
        ]

    # -- readers used by the dashboard --------------------------------------------------

    def recent_lessons(self, k: int = 5) -> list[Lesson]:
        return [
            Lesson(
                symbol=r["symbol"],
                ts=datetime.fromisoformat(r["ts"]),
                outcome_pct=r["outcome_pct"],
                exit_reason=ExitReason(r["exit_reason"]),
                lesson=r["lesson"],
                tags=json.loads(r["tags_json"]),
            )
            for r in self.query("SELECT * FROM lessons ORDER BY id DESC LIMIT ?", (k,))
        ]

    def latest_agent_runs(self, limit: int = 40) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,))

    def latest_bar_id(self) -> str | None:
        rows = self.query("SELECT bar_id FROM agent_runs ORDER BY id DESC LIMIT 1")
        return rows[0]["bar_id"] if rows else None

    def runs_for_bar(self, bar_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM agent_runs WHERE bar_id = ? ORDER BY id", (bar_id,))

    def latest_proposals(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM proposals ORDER BY id DESC LIMIT ?", (limit,))

    def latest_risk(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM risk_decisions ORDER BY id DESC LIMIT ?", (limit,))

    def latest_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))

    def equity_series(self, limit: int = 300) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM equity_curve ORDER BY id DESC LIMIT ?", (limit,))
        return list(reversed(rows))

    def disagreement_grid(self, bars: int = 24) -> list[dict[str, Any]]:
        """Recent disagreement scores, oldest first, for the dashboard heatmap."""
        rows = self.query(
            "SELECT * FROM disagreements WHERE bar_id IN "
            "(SELECT DISTINCT bar_id FROM disagreements ORDER BY id DESC LIMIT ?) "
            "ORDER BY id",
            (bars,),
        )
        return rows

    def degradation_count(self, since: str | None = None) -> int:
        rows = (
            self.query("SELECT COUNT(*) AS n FROM degradations WHERE ts >= ?", (since,))
            if since
            else self.query("SELECT COUNT(*) AS n FROM degradations")
        )
        return rows[0]["n"] if rows else 0

    def last_run_for_agent(self, agent: str, since: str, limit: int = 3) -> list[dict[str, Any]]:
        """Most recent rows for one agent from this process's session only.

        Scoped by ``since`` so a cached agent's card shows what it last said *today*,
        not an error from a run last week against a different vendor.
        """
        return self.query(
            "SELECT * FROM agent_runs WHERE agent = ? AND ts >= ? ORDER BY id DESC LIMIT ?",
            (agent, since, limit),
        )

    def _grouped_errors(self, since: str, limit: int = 5) -> list[dict[str, Any]]:
        """Degradations since ``since``, grouped by *kind* of error.

        Vendor errors carry a per-request id and a "retry in 23.5s" hint, so grouping on
        the raw string puts a hundred identical billing failures into a hundred groups
        of one. Strip the volatile parts first, then count.
        """
        rows = self.query(
            "SELECT agent, error, ts FROM degradations WHERE ts >= ? ORDER BY id",
            (since,),
        )
        groups: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = _error_kind(r["error"] or "")
            g = groups.setdefault(
                key, {"error": key, "n": 0, "agents": [], "last_ts": r["ts"]}
            )
            g["n"] += 1
            g["last_ts"] = max(g["last_ts"], r["ts"])
            if r["agent"] not in g["agents"]:
                g["agents"].append(r["agent"])
        out = sorted(groups.values(), key=lambda g: g["n"], reverse=True)[:limit]
        for g in out:
            g["agents"] = ",".join(g["agents"])
        return out

    def analytics(self, since: str) -> dict[str, Any]:
        """Session-scoped desk analytics for the dashboard. Reads only."""
        agents = self.query(
            "SELECT agent, model, COUNT(*) AS runs, "
            "AVG(latency_ms) AS avg_ms, MAX(latency_ms) AS max_ms, "
            "SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out, "
            "SUM(status = 'ok') AS ok, SUM(status = 'degraded') AS degraded, "
            "SUM(status = 'fallback') AS fallback, SUM(status = 'stub') AS stub "
            "FROM agent_runs WHERE ts >= ? GROUP BY agent, model ORDER BY agent, runs DESC",
            (since,),
        )
        actions = self.query(
            "SELECT action, COUNT(*) AS n, AVG(conviction) AS avg_conviction "
            "FROM proposals WHERE ts >= ? GROUP BY action",
            (since,),
        )
        risk = self.query(
            "SELECT SUM(approved) AS approved, COUNT(*) AS total FROM risk_decisions "
            "WHERE ts >= ?",
            (since,),
        )
        reasons = self.query(
            "SELECT reason, COUNT(*) AS n FROM risk_decisions WHERE ts >= ? AND approved = 0 "
            "GROUP BY reason ORDER BY n DESC LIMIT 6",
            (since,),
        )
        trades = self.query(
            "SELECT COUNT(*) AS n, SUM(outcome_pct > 0) AS wins, AVG(outcome_pct) AS avg_pct, "
            "MAX(outcome_pct) AS best, MIN(outcome_pct) AS worst, "
            "SUM(exit_reason = 'STOP') AS stops, SUM(exit_reason = 'TAKE') AS takes "
            "FROM lessons WHERE ts >= ?",
            (since,),
        )
        orders = self.query(
            "SELECT side, intent, COUNT(*) AS n FROM orders WHERE ts >= ? GROUP BY side, intent",
            (since,),
        )
        # Why agents failed, grouped. Without this the status bar says "degraded" and
        # nobody can tell an expired credit card from a timeout without opening a DB.
        errors = self._grouped_errors(since)
        r = risk[0] if risk else {}
        t = trades[0] if trades else {}
        return {
            "agents": agents,
            "actions": actions,
            "risk": {"approved": r.get("approved") or 0, "total": r.get("total") or 0},
            "reject_reasons": reasons,
            "trades": {
                "n": t.get("n") or 0,
                "wins": t.get("wins") or 0,
                "avg_pct": t.get("avg_pct"),
                "best": t.get("best"),
                "worst": t.get("worst"),
                "stops": t.get("stops") or 0,
                "takes": t.get("takes") or 0,
            },
            "orders": orders,
            "errors": errors,
        }
