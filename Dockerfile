# Multi-stage Dockerfile for vaak
# Stage 1: Dependency builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .python-version ./

RUN uv venv /app/.venv && uv pip install --no-cache -r pyproject.toml

# Stage 2: Minimal runtime image
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY vaak /app/vaak
COPY config.example.yaml pyproject.toml ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV VAAK_HOST=0.0.0.0
ENV VAAK_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "-m", "vaak.cli", "--server-name", "0.0.0.0"]
