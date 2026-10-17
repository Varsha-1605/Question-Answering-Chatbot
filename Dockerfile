# BUILD STAGE
FROM python:3.13-slim AS builder

# Copy UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

WORKDIR /app

# UV Docker optimizations
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

# Install dependencies first (cached if unchanged)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Copy app code and install project (.dockerignore keeps .env, tests and data out)
COPY . ./
RUN uv sync --locked --no-dev

# PRODUCTION STAGE
FROM python:3.13-slim

WORKDIR /app

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Copy app and dependencies from builder stage
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Bake the reranker model into the image so the first start does not download it
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health').status==200 else 1)"

# exec replaces shell so uvicorn receives SIGTERM for clean shutdown
CMD ["/bin/sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"$PORT\" --ws-max-size 1048576 --proxy-headers --forwarded-allow-ips '*'"]
