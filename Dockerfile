# BudaPilot on a container host (Railway, Fly, Render, any VPS with Docker).
#
# One process: the trading loop and the dashboard share an asyncio event loop. The
# journal is SQLite, so mount a volume at /data and point BUDAPILOT_DB there or the
# audit trail (and the kill-switch state) vanishes on every redeploy.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so a code change does not reinstall pandas.
COPY pyproject.toml README.md ./
COPY budapilot ./budapilot
# /data is world-writable because Hugging Face Spaces run the container as uid 1000,
# not root, and SQLite needs to create its -wal/-shm files next to the database.
RUN pip install --no-cache-dir . && mkdir -p /data && chmod 777 /data

# Frozen fixtures (if any were generated with scripts/fetch_fixtures.py) make
# --demo-safe real. The trailing glob keeps the build working when the directory is
# absent: the feed then synthesises a seeded random walk instead.
COPY fixture[s] ./fixtures/

# Runtime defaults for hosting. Every one of these is overridable from the host's
# Variables tab; PORT is injected by the platform and picked up by config.Settings.
ENV BUDAPILOT_HOST=0.0.0.0 \
    BUDAPILOT_DB=/data/budapilot.db \
    BUDAPILOT_SERVE_AFTER_DONE=true \
    ALPACA_PAPER=true

EXPOSE 8000

# Paper-only by construction: --live means "real orders to the PAPER account".
# --max-minutes bounds each run; the dashboard's Start button begins the next one.
CMD ["python", "-m", "budapilot", "--live", "--interval", "300", "--max-minutes", "20"]
