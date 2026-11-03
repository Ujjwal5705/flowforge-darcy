FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    FLOWFORGE_THREADS=2 \
    FLOWFORGE_ARTIFACT_DIR=/app/artifacts/darcy-fno \
    FLOWFORGE_FRONTEND_DIR=/app/frontend/dist
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /usr/local/bin/uv
# Install the CPU wheel first. The ==2.8.0 requirement also accepts 2.8.0+cpu,
# avoiding CUDA dependencies on the Linux deployment target.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r requirements.lock
RUN useradd --create-home --uid 1000 flowforge

FROM dependencies AS runtime
COPY --chown=flowforge:flowforge ml/ ./ml/
COPY --chown=flowforge:flowforge backend/ ./backend/
COPY --chown=flowforge:flowforge artifacts/darcy-fno/config.json artifacts/darcy-fno/best.pt artifacts/darcy-fno/dataset.json artifacts/darcy-fno/evaluation.json artifacts/darcy-fno/examples.json artifacts/darcy-fno/history.json ./artifacts/darcy-fno/
COPY --from=frontend --chown=flowforge:flowforge /build/frontend/dist ./frontend/dist
USER flowforge
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/health',timeout=5)); assert r['model_loaded']"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
