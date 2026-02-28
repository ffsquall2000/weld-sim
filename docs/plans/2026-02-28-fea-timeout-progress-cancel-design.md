# FEA Timeout Protection, Progress Tracking & Cancellation Design

**Date**: 2026-02-28
**Status**: Approved

## Problem Statement

1. FEA computations can run forever (35-core zombie consuming 18GB RAM for 38+ minutes)
2. No way to cancel a running simulation from the frontend
3. No real-time progress feedback during simulation (only a boolean loading spinner)
4. PM2 `embedding-service` port conflict with `weld-sim` on port 8001
5. Stale `systemd weld-sim.service` file could cause future conflicts

## Solution: Subprocess Isolation + Hybrid Progress + Cancel

### Architecture Overview

```
Frontend (Vue)                    Backend (FastAPI)                    FEA Subprocess
┌────────────┐   WebSocket    ┌──────────────────┐  mp.Process    ┌──────────────┐
│ Progress UI │◄─────────────►│ analysis_manager  │◄─────────────►│ FEA Worker   │
│ Cancel Btn  │   /ws/analysis│ FEAProcessRunner  │  mp.Queue     │ Gmsh+SolverA │
│ % + phase   │   /{task_id}  │                   │  mp.Event     │              │
└────────────┘               └──────────────────┘  (cancel)      └──────────────┘
```

### Component Design

#### 1. FEAProcessRunner (`web/services/fea_process_runner.py`)

New class that wraps FEA execution in a `multiprocessing.Process`:

- **run_with_progress()**: Async generator yielding progress events
- **Timeout**: `process.join(timeout)` + `process.kill()` if exceeded
- **Cancel**: `multiprocessing.Event` checked at each FEA phase boundary
- **Progress**: `multiprocessing.Queue` for phase-level + time-estimated updates
- **Resource limits**: `gmsh.option.setNumber("General.NumThreads", N)` caps Gmsh threads
- **Mesh pre-validation**: Estimate node count before meshing; reject if > 50,000

#### 2. Hybrid Progress Estimation

Fixed phase nodes with time-based smooth interpolation within phases:

| Phase | Fixed % Start | Fixed % End | Estimation Method |
|-------|--------------|-------------|-------------------|
| init_gmsh | 0% | 5% | Instant |
| import_step | 5% | 15% | File size based |
| mesh_generation | 15% | 35% | mesh_size + geometry size |
| matrix_assembly | 35% | 55% | Node count linear |
| eigenvalue_solve | 55% | 85% | Historical time estimate |
| mode_classification | 85% | 95% | Instant |
| result_packaging | 95% | 100% | Instant |

Within each phase: `progress = phase_start + (phase_end - phase_start) * min(elapsed / estimated, 0.95)`

#### 3. Cancellation Flow

```
User clicks Cancel → WebSocket {action: "cancel", task_id}
  → analysis_manager sets cancel_event
  → FEA subprocess checks cancel_event at each phase boundary
  → If set: gmsh.finalize(), cleanup, sys.exit(0)
  → If subprocess ignores (stuck in Gmsh): timeout kills it with SIGKILL
```

#### 4. Route Refactoring

All 5 FEA POST endpoints refactored to use FEAProcessRunner:
- `POST /geometry/fea/run`
- `POST /geometry/fea/run-step`
- `POST /acoustic/analyze`
- `POST /assembly/analyze`
- `POST /assembly/modal`

Each endpoint:
1. Creates task via `analysis_manager.create_task()`
2. Returns `task_id` immediately (HTTP 202 Accepted)
3. Spawns FEA subprocess
4. Client connects to WebSocket for progress
5. On completion: result available via `GET /analysis/{task_id}/result`

**OR** (simpler approach): Keep synchronous HTTP response but stream progress via parallel WebSocket.

#### 5. Frontend Progress Component

Shared `<FEAProgress>` component used in GeometryView, AcousticView, SimulationView:

```
┌─────────────────────────────────────────┐
│  🔄 模态分析中...         [取消]         │
│  ████████████░░░░░░░░  55%              │
│  当前阶段: 矩阵组装 (3/7)               │
│  已用时: 0:08 / 预估: 0:15              │
└─────────────────────────────────────────┘
```

#### 6. PM2/System Cleanup

- Change `embedding-service` from port 8001 to 8005
- Remove `/etc/systemd/system/weld-sim.service`
- `pm2 save` to persist correct configuration

## Files to Create/Modify

### New Files
- `web/services/fea_process_runner.py` - Subprocess runner with progress/cancel
- `frontend/src/components/FEAProgress.vue` - Shared progress UI component

### Modified Files
- `web/routers/geometry.py` - Use FEAProcessRunner
- `web/routers/acoustic.py` - Use FEAProcessRunner
- `web/routers/assembly.py` - Use FEAProcessRunner
- `web/routers/ws.py` - Enhanced cancel support
- `web/services/analysis_manager.py` - Integration with subprocess cancel
- `web/services/fea_service.py` - Add progress callback hooks
- `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py` - Progress callbacks + thread limit
- `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` - Progress callbacks
- `frontend/src/views/GeometryView.vue` - Add FEAProgress component
- `frontend/src/views/AcousticView.vue` - Add FEAProgress component
- `frontend/src/views/SimulationView.vue` - Add FEAProgress component
- `frontend/src/i18n/zh-CN.json` - Progress-related translations
- `frontend/src/i18n/en.json` - Progress-related translations
- Server: PM2 embedding-service port config
- Server: Remove systemd service file

## Timeout Configuration

- Default: 300 seconds (5 minutes)
- Configurable via env var: `UWM_FEA_TIMEOUT=300`
- Gmsh thread limit: 8 (configurable via `UWM_GMSH_THREADS=8`)
