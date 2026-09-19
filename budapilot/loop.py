"""The trading loop. One turn per closed bar.

    features -> agents -> proposal -> RISK VETO -> execution -> stop armed -> journal

The risk engine sits between the agents and the broker and cannot be bypassed. That is
the single most important structural property of this file: there is no code path from
a model's opinion to an order that does not pass through ``risk.evaluate``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from budapilot.agents.bus import AgentBus
from budapilot.agents.reflection import ReflectionContext
from budapilot.config import BAR_MINUTES, LESSON_INJECT_K, WATCHLIST
from budapilot.contracts import (
    Action,
    BrokerPort,
    ExitReason,
    FeatureBundle,
    NewsItem,
    NewsRisk,
    ProtectedPosition,
    utcnow,
)
from budapilot.execution.engine import ExecutionEngine
from budapilot.execution.stops import StopManager
from budapilot.features.indicators import bars_to_frame, compute_features, realized_vol_percentile
from budapilot.journal.store import Journal
from budapilot.risk import engine as risk

log = logging.getLogger("budapilot.loop")


class TradingLoop:
    def __init__(
        self,
        *,
        feed,  # MarketDataPort + NewsPort
        broker: BrokerPort,
        bus: AgentBus,
        journal: Journal,
        symbols: list[str] | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.feed = feed
        self.broker = broker
        self.bus = bus
        self.journal = journal
        self.symbols = symbols or WATCHLIST
        self.interval_s = interval_s if interval_s is not None else BAR_MINUTES * 60

        self.stops = StopManager(broker, on_exit=self._on_exit, journal=journal)
        self.execution = ExecutionEngine(broker, self.stops, journal)

        self.bar_count = 0
        self.running = False
        self.last_decision = None
        # Entry context kept so the reflection agent can review the actual reasoning.
        self._entry_context: dict[str, dict] = {}

    # -- startup ------------------------------------------------------------------------

    async def startup(self) -> None:
        """Rebuild protective state before the first bar. Never trade before this."""
        try:
            await self.stops.reconcile(self.journal.load_protected())
        except Exception as exc:  # noqa: BLE001
            log.warning("Reconciliation failed (continuing without it): %s", exc)

    # -- one bar -------------------------------------------------------------------------

    async def run_once(self) -> None:
        self.bar_count += 1
        bar_id = f"{utcnow():%Y%m%dT%H%M%S}-{self.bar_count:05d}"

        features, news = await self._gather(bar_id)
        if not features:
            log.warning("No features this bar; skipping.")
            return

        portfolio = await self.broker.get_portfolio()
        self.journal.log_equity(portfolio.equity, portfolio.cash, portfolio.exposure_pct)

        decision = await self.bus.run_bar(
            bar_id=bar_id,
            features=features,
            news_by_symbol=news,
            portfolio=portfolio,
            lessons=self.journal.recent_lessons(LESSON_INJECT_K),
            realized_vol_pct=self._realized_vol(features),
        )
        self.journal.log_decision(decision)
        self.last_decision = decision

        proposal = decision.proposal
        if proposal is None or proposal.action is Action.HOLD:
            return

        feature = next((f for f in features if f.symbol == proposal.symbol), None)
        if feature is None:
            log.warning("PM proposed %s, which is not in the feature set.", proposal.symbol)
            return

        news_risk = next(
            (op.news.news_risk for op in decision.opinions if op.symbol == proposal.symbol),
            NewsRisk.LOW,
        )
        asset = await self.feed.get_asset(proposal.symbol)

        # THE VETO. Nothing reaches the broker without passing here.
        verdict = risk.evaluate(proposal, feature, portfolio, asset, news_risk=news_risk)
        self.journal.log_risk(verdict, bar_id)

        if not verdict.approved:
            log.info("Risk rejected %s: %s", verdict.symbol, verdict.reason)
            return

        order = await self.execution.execute(verdict)
        if order is not None:
            log.info(
                "Order %s %s %s -> %s", order.side.value, order.qty, order.symbol,
                order.status.value,
            )
            if verdict.action is Action.BUY:
                self._entry_context[proposal.symbol] = {
                    "proposal": proposal,
                    "technical": next(
                        (
                            op.technical
                            for op in decision.opinions
                            if op.symbol == proposal.symbol
                        ),
                        None,
                    ),
                    "news": next(
                        (op.news for op in decision.opinions if op.symbol == proposal.symbol),
                        None,
                    ),
                    "bar": self.bar_count,
                }

    async def _gather(
        self, bar_id: str
    ) -> tuple[list[FeatureBundle], dict[str, list[NewsItem]]]:
        """Fetch bars and news for the watchlist concurrently, tolerating failures."""
        bar_lists = await asyncio.gather(
            *(self.feed.get_bars(s, limit=200) for s in self.symbols),
            return_exceptions=True,
        )
        news_lists = await asyncio.gather(
            *(self.feed.get_news(s, limit=20) for s in self.symbols),
            return_exceptions=True,
        )

        features: list[FeatureBundle] = []
        news: dict[str, list[NewsItem]] = {}
        self._frames = {}

        for symbol, bars, items in zip(self.symbols, bar_lists, news_lists, strict=True):
            if isinstance(bars, Exception) or not bars:
                why = bars if isinstance(bars, Exception) else "empty"
                log.warning("No bars for %s (%s)", symbol, why)
                continue
            features.append(compute_features(symbol, bars))
            self._frames[symbol] = bars
            news[symbol] = [] if isinstance(items, Exception) else items

            # Keep the simulated broker marked to the latest price.
            if hasattr(self.broker, "set_price"):
                self.broker.set_price(symbol, bars[-1].close)

        # Feed the software stops on every bar close. In live trading this is also
        # driven by the trade websocket at a much higher rate.
        for symbol in list(self.stops.positions):
            price = await self.feed.latest_price(symbol)
            if price > 0:
                await self.stops.on_price(symbol, price)

        return features, news

    def _realized_vol(self, features: list[FeatureBundle]) -> float:
        btc = getattr(self, "_frames", {}).get("BTC/USD")
        if not btc:
            return 50.0
        return realized_vol_percentile(bars_to_frame(btc)["close"])

    # -- exits ---------------------------------------------------------------------------

    async def _on_exit(
        self, pos: ProtectedPosition, fill: float, reason: ExitReason
    ) -> None:
        """A position closed. Journal it, then let A7 write the lesson."""
        outcome_pct = (fill / pos.entry - 1.0) * 100 if pos.entry else 0.0
        log.info("%s exited via %s at %.4f (%+.2f%%)", pos.symbol, reason.value, fill, outcome_pct)

        ctx = self._entry_context.pop(pos.symbol, None)
        if ctx is None or ctx.get("proposal") is None:
            return

        result = await self.bus.reflect(
            ReflectionContext(
                symbol=pos.symbol,
                proposal=ctx["proposal"],
                technical=ctx.get("technical"),
                news=ctx.get("news"),
                entry=pos.entry,
                exit_price=fill,
                outcome_pct=outcome_pct,
                exit_reason=reason,
                bars_held=self.bar_count - ctx.get("bar", self.bar_count),
            )
        )
        self.journal.log_agent_run(result, f"exit-{pos.symbol}")
        self.journal.add_lesson(result.output)

    # -- driver ---------------------------------------------------------------------------

    async def run_forever(self, max_bars: int | None = None) -> None:
        self.running = True
        await self.startup()
        try:
            while self.running:
                started = datetime.now()
                try:
                    await self.run_once()
                except Exception:  # noqa: BLE001 -- one bad bar must not end the session
                    log.exception("Bar failed; continuing.")

                if max_bars and self.bar_count >= max_bars:
                    break
                if hasattr(self.feed, "advance") and not self.feed.advance():
                    log.info("Fixture replay exhausted after %d bars.", self.bar_count)
                    break

                elapsed = (datetime.now() - started).total_seconds()
                await asyncio.sleep(max(self.interval_s - elapsed, 0))
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False
