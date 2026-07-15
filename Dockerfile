# =============================================================================
# Dockerfile -- audible-frames-azure
#
# Packages the FastAPI app into a Docker image that runs identically on your
# laptop, in CI, and in Azure Container Apps.
#
# HOW TO BUILD LOCALLY (for testing):
#   docker build -t audible-frames .
#   docker run -p 8000:8000 --env-file .env audible-frames
#   Open http://localhost:8000
#
# The GitHub Actions workflow (deploy.yml) builds and pushes this automatically
# on every push to main.
# =============================================================================

# ── Stage 1: build ────────────────────────────────────────────────────────────
# We use a two-stage build:
#   Stage 1 (builder): installs all packages including build tools
#   Stage 2 (runtime): copies only the installed packages -- no build tools
# Result: smaller final image (build tools like gcc can be ~200MB)
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed to compile some Python packages
# (azure-cognitiveservices-speech needs ssl headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache this layer.
# If requirements.txt hasn't changed, pip install is skipped on next build.
COPY requirements.txt .

# Install into a local folder (/install) so we can copy just this folder
# to the runtime stage -- leaves build tools behind.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies for Azure Speech SDK
# libasound2: audio library (Speech SDK needs it even for in-memory synthesis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
# .dockerignore excludes: venv/, .env, __pycache__, evals/dataset/images/
COPY src/ ./src/

# ── Security: non-root user ────────────────────────────────────────────────────
# Running as root inside a container is a security risk.
# If the app is compromised, an attacker would have root inside the container.
# This creates a minimal user with no login shell and no home directory.
RUN useradd --system --no-create-home --shell /bin/false appuser
USER appuser

# ── Health check ───────────────────────────────────────────────────────────────
# Docker and Azure Container Apps both use this to decide if the container
# is healthy. If /health returns non-200 three times in a row, the container
# is restarted automatically.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ──────────────────────────────────────────────────────────────
# --workers 2: two parallel worker processes (handles concurrent requests better)
# --host 0.0.0.0: listen on all interfaces (required inside containers)
# --port 8000: match EXPOSE
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
