# =============================================================================
# Backend Dockerfile — Ultrasonic Metal Welding Virtual Simulation Platform
# Multi-stage build: builder (install deps) -> runtime (slim image)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies and gmsh
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgmsh-dev \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency specification first for layer caching
COPY pyproject.toml ./

# Install Python dependencies into a virtual-env so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="weldsim-team"
LABEL description="WeldSim backend API server"

# System packages required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgmsh-dev \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# NOTE: FEniCS (dolfinx), Elmer, and CalculiX are optional solver backends.
# They are complex to install and significantly increase image size.
# To add FEniCS, consider using dolfinx/dolfinx as a base image instead.
# To add Elmer/CalculiX, install from source or distribution packages.
# Users who need these solvers can extend this image:
#
#   FROM weldsim-backend:latest
#   RUN apt-get update && apt-get install -y elmer calculix-ccx ...

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
RUN groupadd --gid 1000 weldsim \
    && useradd --uid 1000 --gid weldsim --shell /bin/bash --create-home weldsim

# Application directory
WORKDIR /app

# Copy backend source code
# The project uses `from backend.app.xxx import ...` style imports,
# so we place the source under /app/backend/app/
COPY --chown=weldsim:weldsim ./app /app/backend/app/

# Create storage directory
RUN mkdir -p /app/storage && chown weldsim:weldsim /app/storage

# Set PYTHONPATH so "from backend.app.xxx" imports resolve correctly
ENV PYTHONPATH="/app"

# Switch to non-root user
USER weldsim

EXPOSE 8001

# Health check — hit the health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/api/v2/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
