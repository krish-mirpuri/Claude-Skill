# Serving image. The model is trained during the build so the container starts
# ready and needs no mounted volume.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"
COPY conf ./conf

FROM base AS trained
RUN nuance download && nuance train && rm -rf data/raw data/processed

FROM base AS serve
COPY --from=trained /app/models ./models
COPY --from=trained /app/reports ./reports

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "nuance.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
