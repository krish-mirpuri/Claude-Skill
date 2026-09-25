# Serving image. The model artifacts are built during the image build so the
# container starts with a ready model and needs no volume mount.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM needs libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so edits to source do not invalidate the install layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve,explain]"

COPY conf ./conf
COPY scripts ./scripts

# ---- build stage: train the model into the image -------------------------
FROM base AS trained
RUN overbook download \
    && overbook prepare \
    && overbook train \
    && overbook backtest \
    && rm -rf data/raw data/interim

# ---- runtime -------------------------------------------------------------
FROM base AS serve
COPY --from=trained /app/models ./models
COPY --from=trained /app/reports ./reports

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "overbook.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
