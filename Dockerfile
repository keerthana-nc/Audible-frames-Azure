# =============================================================================
# Dockerfile — audible-frames-azure
#
# What this does:
#   Packages the entire app into a self-contained image that can run anywhere
#   (your laptop, Azure Container Apps, any cloud) without needing Python or
#   packages installed separately.
#
# STATUS: Phase 1 skeleton — fully fleshed out in Phase 6 (CI/CD).
#   Phase 6 will add: non-root user (security), health checks, build args.
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is the official Python 3.11 image on a minimal Debian base.
# "slim" means it has Python but strips out docs, compilers, and test files
# to keep the image small (~50MB vs ~300MB for the full image).
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
# All commands below run relative to /app inside the container.
# This is the conventional location for app code in Docker images.
WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────────────
# Copy requirements.txt FIRST (before the rest of the code).
# Why: Docker caches each layer. If requirements.txt hasn't changed, Docker
# skips the pip install step entirely — saves minutes on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
# Copy everything else after pip install so code changes don't invalidate
# the pip cache layer above.
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
# FastAPI will listen on port 8000. EXPOSE documents this — it doesn't
# actually open the port; that's done by Azure Container Apps or docker run.
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# uvicorn runs the FastAPI app.
#   src.api:app  → the `app` variable inside src/api.py
#   --host 0.0.0.0  → listen on all network interfaces (required in containers)
#   --port 8000     → match the EXPOSE above
#
# Phase 6 will add: --workers, proper signal handling, non-root user.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
