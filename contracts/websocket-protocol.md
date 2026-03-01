# WebSocket Protocol Specification

## Endpoints

### Run Progress: `WS /api/v2/ws/runs/{run_id}`

Client connects to receive real-time solver progress updates.

### Optimization Progress: `WS /api/v2/ws/optimizations/{optimization_id}`

Client connects to receive optimization iteration updates.

## Message Types (Server → Client)

### Progress Update
```json
{
  "type": "progress",
  "run_id": "uuid",
  "percent": 45.0,
  "phase": "solving",
  "message": "Harmonic response sweep: 9/21 frequencies",
  "elapsed_s": 12.3,
  "timestamp": "2026-03-01T10:30:00Z"
}
```

Phases: `meshing`, `solving`, `postprocessing`

### Metric Update (streamed during solve)
```json
{
  "type": "metric_update",
  "run_id": "uuid",
  "metric_name": "natural_frequency_hz",
  "value": 19850.3,
  "unit": "Hz",
  "timestamp": "2026-03-01T10:30:05Z"
}
```

### Run Completed
```json
{
  "type": "completed",
  "run_id": "uuid",
  "status": "completed",
  "compute_time_s": 45.2,
  "metrics_summary": {
    "natural_frequency_hz": 19850.3,
    "amplitude_uniformity": 0.92,
    "max_von_mises_stress_mpa": 312.5,
    "stress_safety_factor": 2.82,
    "frequency_deviation_pct": 0.75
  },
  "timestamp": "2026-03-01T10:30:50Z"
}
```

### Run Error
```json
{
  "type": "error",
  "run_id": "uuid",
  "error": "Solver failed to converge after 500 iterations",
  "solver_log_tail": "... last 10 lines of solver log ...",
  "timestamp": "2026-03-01T10:31:00Z"
}
```

### Run Cancelled
```json
{
  "type": "cancelled",
  "run_id": "uuid",
  "timestamp": "2026-03-01T10:31:00Z"
}
```

### Optimization Iteration Complete
```json
{
  "type": "optimization_iteration",
  "optimization_id": "uuid",
  "iteration": 5,
  "run_id": "uuid",
  "parameters": {"width_mm": 28.5, "height_mm": 85.0},
  "metrics": {"amplitude_uniformity": 0.91, "stress_safety_factor": 2.5},
  "feasible": true,
  "pareto_updated": true,
  "timestamp": "2026-03-01T10:35:00Z"
}
```

## Message Types (Client → Server)

### Heartbeat/Ping
```json
{
  "type": "ping"
}
```

Server responds with:
```json
{
  "type": "pong"
}
```

## Connection Lifecycle

1. Client connects to WebSocket endpoint
2. Server sends initial status message
3. Server streams progress/metric updates
4. On completion/error, server sends final message
5. Connection remains open for subsequent runs (client can reconnect)

## Reconnection

- Client should implement exponential backoff reconnection
- On reconnect, server sends current state (last known status)
- Missed messages are not replayed (client should poll REST API for missed data)
