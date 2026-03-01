# Deployment Guide

Deployment guide for the Ultrasonic Metal Welding Virtual Simulation Platform.

## Prerequisites

- **Docker** >= 24.0 ([install](https://docs.docker.com/get-docker/))
- **Docker Compose** >= 2.20 (bundled with Docker Desktop, or [install separately](https://docs.docker.com/compose/install/))
- At least 4 GB of free RAM (8 GB recommended for solver workloads)
- At least 10 GB of free disk space

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Start all services in detached mode
docker compose up -d

# Verify all services are running
docker compose ps

# Follow logs
docker compose logs -f
```

Once all services are healthy, the platform is available at:

| Service   | URL                              |
|-----------|----------------------------------|
| Frontend  | http://localhost:3000             |
| Backend API | http://localhost:8001/api/v2/docs |
| PostgreSQL | localhost:5432                   |
| Redis     | localhost:6379                    |

## Environment Variables

All backend configuration is managed through environment variables. Defaults are suitable for development; override them for production.

| Variable        | Default                                                          | Description                                |
|-----------------|------------------------------------------------------------------|--------------------------------------------|
| `DATABASE_URL`  | `postgresql+asyncpg://postgres:postgres@localhost:5432/weldsim`  | Async PostgreSQL connection string         |
| `REDIS_URL`     | `redis://localhost:6379/0`                                       | Redis connection string (broker + backend) |
| `STORAGE_PATH`  | `storage`                                                        | Path for simulation result file storage    |
| `CORS_ORIGINS`  | `["*"]`                                                          | Allowed CORS origins (JSON array string)   |
| `SECRET_KEY`    | `dev-secret-key`                                                 | Secret key for token signing               |
| `DEBUG`         | `true`                                                           | Enable debug mode and verbose SQL logging  |

### Production Overrides

Create a `.env` file at the project root (it is git-ignored):

```env
SECRET_KEY=your-secure-random-key-here
DEBUG=false
CORS_ORIGINS=["https://yourdomain.com"]
```

Docker Compose automatically reads `.env` and substitutes `${SECRET_KEY}` and `${DEBUG}` variables in the compose file.

## Services Architecture

```
                  +-------------------+
                  |    Frontend       |
                  |  (nginx :3000)    |
                  +--------+----------+
                           |
                   /api/   |   /ws/
                           v
                  +-------------------+
                  |    Backend        |
                  | (FastAPI :8001)   |
                  +--------+----------+
                     |           |
              +------+    +-----+------+
              v           v            v
        +---------+  +---------+  +----------+
        |PostgreSQL|  |  Redis  |  | Celery   |
        | (:5432) |  | (:6379) |  | Worker   |
        +---------+  +---------+  +----------+
```

- **Frontend (nginx)**: Serves the Vue SPA, proxies `/api/` and `/ws/` to the backend.
- **Backend (FastAPI)**: REST API, WebSocket server, database migrations.
- **Celery Worker**: Executes long-running simulation and optimization tasks asynchronously.
- **PostgreSQL**: Primary data store for projects, simulations, results, materials.
- **Redis**: Celery message broker and result backend; also used for WebSocket pub/sub.

## Development Setup (Without Docker)

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ".[dev]"

# Start PostgreSQL and Redis locally (or use Docker for just these)
docker compose up -d postgres redis

# Run the API server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8001
```

### Frontend

```bash
cd frontend

# Install dependencies
npm ci

# Start the dev server (proxies /api to localhost:8001)
npm run dev
```

The Vite dev server runs on http://localhost:5173 with hot module replacement.

### Celery Worker (development)

```bash
cd backend
celery -A backend.app.dependencies:celery_app worker --loglevel=info
```

## Solver Installation Notes

The Docker image includes **gmsh** for mesh generation. The following solvers are optional and not included in the default image due to their complexity:

### FEniCS (DOLFINx)

FEniCS is used for finite element analysis. Installation options:

1. **Docker base image** (recommended): Use `dolfinx/dolfinx:stable` as the base image instead of `python:3.11-slim` in `backend/Dockerfile`.
2. **Conda/Mamba**: `conda install -c conda-forge fenics-dolfinx`
3. **From source**: See [DOLFINx documentation](https://github.com/FEniCS/dolfinx)

### Elmer FEM

Elmer is used for multiphysics simulations:

```bash
# Ubuntu/Debian
sudo apt-get install elmer

# From source
git clone https://github.com/ElmerCSC/elmerfem.git
cd elmerfem && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

### CalculiX

CalculiX is used for structural analysis:

```bash
# Ubuntu/Debian
sudo apt-get install calculix-ccx

# Verify installation
ccx --version
```

### Extending the Docker Image

To add solvers to the Docker image, create a custom Dockerfile:

```dockerfile
FROM weldsim-backend:latest

USER root
RUN apt-get update && apt-get install -y \
    elmer \
    calculix-ccx \
    && rm -rf /var/lib/apt/lists/*
USER weldsim
```

## Scaling Considerations

### Celery Workers

Scale Celery workers horizontally for higher simulation throughput:

```bash
# Scale to 4 worker containers
docker compose up -d --scale celery-worker=4
```

Each worker defaults to `--concurrency=2` (2 processes per container). Adjust in `docker-compose.yml` based on available CPU cores.

### PostgreSQL

For production workloads:

- Use a managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
- Configure connection pooling with PgBouncer
- Set up automated backups and point-in-time recovery
- Update `DATABASE_URL` to point to the managed instance

### Redis

For production workloads:

- Use a managed Redis service (AWS ElastiCache, Redis Cloud, etc.)
- Enable Redis persistence (AOF or RDB) for task durability
- Update `REDIS_URL` to point to the managed instance

### Reverse Proxy / Load Balancer

For production, place a reverse proxy (Traefik, Caddy, or cloud LB) in front:

- TLS termination
- Rate limiting
- Request buffering
- Multiple backend replicas

## Common Commands

```bash
# Rebuild images after code changes
docker compose build

# Rebuild a specific service
docker compose build backend

# Restart a single service
docker compose restart backend

# View logs for a specific service
docker compose logs -f celery-worker

# Execute a command inside a running container
docker compose exec backend bash

# Run database migrations
docker compose exec backend alembic upgrade head

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

## Troubleshooting

### Backend cannot connect to PostgreSQL

- Verify PostgreSQL is healthy: `docker compose ps postgres`
- Check logs: `docker compose logs postgres`
- Ensure `DATABASE_URL` uses the service name `postgres` (not `localhost`) when running in Docker

### Backend cannot connect to Redis

- Verify Redis is healthy: `docker compose ps redis`
- Check logs: `docker compose logs redis`
- Ensure `REDIS_URL` uses the service name `redis` (not `localhost`) when running in Docker

### Frontend shows 502 Bad Gateway

- The backend may still be starting. Wait 10-15 seconds and refresh.
- Check backend health: `docker compose logs backend`
- Verify the backend is listening on port 8001: `docker compose exec backend curl http://localhost:8001/api/v2/health`

### Celery worker not processing tasks

- Check worker logs: `docker compose logs celery-worker`
- Verify Redis connectivity: `docker compose exec celery-worker redis-cli -h redis ping`
- Ensure tasks are properly registered: `docker compose exec celery-worker celery -A backend.app.dependencies:celery_app inspect registered`

### Port conflicts

If ports 3000, 5432, 6379, or 8001 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - "3001:80"    # Change 3000 to another port
```

### Out of disk space

Docker images and volumes can consume significant disk space:

```bash
# View disk usage
docker system df

# Remove unused images and containers
docker system prune

# Remove unused volumes (WARNING: deletes data)
docker volume prune
```

### Build failures

- Ensure Docker has at least 4 GB of memory allocated (Docker Desktop settings)
- Try building with `--no-cache`: `docker compose build --no-cache`
- Check network connectivity for package downloads
