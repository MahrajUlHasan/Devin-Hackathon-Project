"""Entrypoint. Cold start to a running dashboard in one command. (NFR6)

    python -m budapilot --demo-safe     zero network calls, fixtures + stub agents
    python -m budapilot                 live crypto data, real Claude agents, SimBroker
    python -m budapilot --live          ... and real orders to an Alpaca paper account

The safety default is deliberate: you have to ask for the real broker. Everything else
degrades toward simulation rather than toward surprise.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import uvicorn

from budapilot.agents.base import AgentRuntime
from budapilot.agents.bus import AgentBus
from budapilot.config import (
    BAR_MINUTES,
    DEBATE_ROUNDS,
    ENABLE_DEBATE,
    PROVIDER_MODELS,
    WATCHLIST,
    settings,
)
from budapilot.journal.store import Journal
from budapilot.loop import TradingLoop
from budapilot.web.app import create_app

log = logging.getLogger("budapilot")

# A --live run with no explicit budget stops after this long. Long enough for a demo
# and a coffee; short enough that a forgotten deploy is not a week of paper trades.
DEFAULT_LIVE_MINUTES = 120


def build(args: argparse.Namespace) -> tuple[TradingLoop, Journal]:
    journal = Journal(args.db)

    # --- data ---------------------------------------------------------------------
    if args.demo_safe:
        from budapilot.data.fixtures import FixtureFeed

        feed = FixtureFeed()
        log.info(
            "demo-safe: fixtures only, no network. %s",
            "synthetic data (no fixtures on disk)" if feed.synthetic else "frozen fixtures",
        )
    else:
        from budapilot.data.alpaca_crypto import AlpacaCryptoFeed

        feed = AlpacaCryptoFeed()
        log.info("Live Alpaca crypto data for %d symbols.", len(WATCHLIST))

    # --- broker -------------------------------------------------------------------
    if args.live and not args.demo_safe:
        if not settings.has_alpaca:
            log.error("--live needs ALPACA_API_KEY and ALPACA_API_SECRET. Copy .env.example.")
            sys.exit(2)
        from budapilot.execution.broker_alpaca import AlpacaPaperBroker

        broker = AlpacaPaperBroker(settings)
        log.warning("LIVE: submitting real orders to your Alpaca PAPER account.")
    else:
        from budapilot.execution.broker_sim import SimBroker

        broker = SimBroker(cash=args.cash)
        log.info("Simulated broker, $%s starting equity.", f"{args.cash:,.0f}")

    # --- agents -------------------------------------------------------------------
    runtime = AgentRuntime(
        offline=args.demo_safe or args.stub_agents,
        provider=args.provider,
        fallback=None if args.no_fallback else "auto",
    )
    if runtime.offline:
        reason = (
            "requested"
            if (args.demo_safe or args.stub_agents)
            else f"no API key for provider '{runtime.provider_name}'"
        )
        log.info("Agents are stubbed (deterministic, no API calls) -- %s.", reason)
        if reason != "requested":
            log.warning(
                "Set %s in .env, or pass --provider to pick the other one.",
                "GEMINI_API_KEY" if runtime.provider_name == "gemini" else "ANTHROPIC_API_KEY",
            )
    else:
        tiers = sorted(set(runtime.models.values()))
        log.info("LLM provider: %s  |  models: %s", runtime.provider_name, ", ".join(tiers))
        if runtime.fallback_name:
            log.info(
                "Fallback provider: %s  |  models: %s",
                runtime.fallback_name,
                ", ".join(sorted(set(runtime.fallback_models.values()))),
            )
        else:
            log.info("No fallback provider (only one vendor key configured).")

    debate = args.debate or ENABLE_DEBATE
    if debate:
        log.info("Bull/Bear debate enabled (%d round(s)).", DEBATE_ROUNDS)

    loop = TradingLoop(
        feed=feed,
        broker=broker,
        bus=AgentBus(runtime, enable_debate=debate),
        journal=journal,
        interval_s=args.interval if args.interval is not None else BAR_MINUTES * 60,
    )
    return loop, journal


async def serve(args: argparse.Namespace) -> None:
    loop, journal = build(args)
    app = create_app(journal, loop)

    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)

    max_minutes = args.max_minutes
    if args.live and max_minutes is None:
        max_minutes = DEFAULT_LIVE_MINUTES
        log.warning(
            "--live without --max-minutes: stopping after %d min. Pass --max-minutes 0 "
            "to run until interrupted.",
            max_minutes,
        )
    if max_minutes:
        log.info("Time budget: %.0f minutes.", max_minutes)

    print(f"\n  BudaPilot dashboard  ->  http://{args.host}:{args.port}\n")

    trading: asyncio.Task | None = None

    def launch() -> asyncio.Task:
        """Start (or restart) the trading loop with the same bar and time budget."""
        nonlocal trading
        trading = asyncio.create_task(
            loop.run_forever(max_bars=args.max_bars, max_minutes=max_minutes or None)
        )
        return trading

    launch()
    # The dashboard's Start button can bring a finished loop back with a fresh budget.
    app.state.relaunch = launch
    serving = asyncio.create_task(server.serve())

    try:
        if args.serve_after_done:
            # Hosted: the URL must stay alive after the budget is spent so the results
            # remain visible. Only a SIGTERM/Ctrl+C ends the process.
            await serving
        else:
            await asyncio.wait({trading, serving}, return_when=asyncio.FIRST_COMPLETED)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.stop()
        # Let uvicorn unwind its own lifespan before cancelling, otherwise it logs a
        # CancelledError traceback that looks like a crash but is just shutdown.
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, timeout=3)
        except (TimeoutError, asyncio.CancelledError):
            serving.cancel()
        if trading is not None:
            trading.cancel()
            await asyncio.gather(trading, return_exceptions=True)
        await asyncio.gather(serving, return_exceptions=True)
        journal.close()
        print("Stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="budapilot", description=__doc__)
    parser.add_argument(
        "--demo-safe",
        action="store_true",
        help="No network calls at all: fixture data, stub agents, simulated broker.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit real orders to an Alpaca PAPER account.",
    )
    parser.add_argument(
        "--stub-agents", action="store_true", help="Real data, deterministic agents."
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_MODELS),
        default=None,
        help="Which LLM vendor runs the agents. Default: LLM_PROVIDER from .env, else "
        "Claude when ANTHROPIC_API_KEY is set, else whichever key is present.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not retry a failed agent call on the other vendor. By default a "
        "Claude failure is retried once on Gemini (and vice versa) before stubbing.",
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Run the A9/A10 bull-vs-bear round before the PM. Roughly doubles tokens "
        "and latency per candidate.",
    )
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument(
        "--interval", type=float, default=None, help="Seconds between bars (default 300)."
    )
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help=f"Stop the loop after this many minutes of wall-clock time. Defaults to "
        f"{DEFAULT_LIVE_MINUTES} with --live so a hosted demo cannot run unattended "
        "forever; 0 disables the budget.",
    )
    parser.add_argument(
        "--serve-after-done",
        action="store_true",
        default=os.getenv("BUDAPILOT_SERVE_AFTER_DONE", "").lower() in ("1", "true", "yes"),
        help="Keep the dashboard up after the loop's bar/time budget is spent instead of "
        "exiting. For hosting. Also enabled by BUDAPILOT_SERVE_AFTER_DONE=true.",
    )
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
