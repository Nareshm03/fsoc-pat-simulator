# FSOC PAT Simulator - Frontend

Next.js dashboard for the Free Space Optical Communication Pointing,
Acquisition, and Tracking simulator.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The FastAPI backend must
be running (default `http://localhost:8000`, see backend `main.py`).

## Configuration

Backend URLs come from `src/app/config.ts` and can be overridden with
environment variables (see `.env.example`):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Features

- **Real-time Dashboard**: PAT state, beacon position, tracking error, FPS,
  and gimbal angles streamed over WebSocket (auto-reconnect with backoff).
- **Virtual Camera View**: Beacon marker, crosshair, and error vector over
  the live base64-JPEG frame messages (streamed every 3rd tick).
- **Disturbance Controls**: Debounced sliders plus presets for turbulence,
  vibration, camera motion, and sensor noise.
- **Control Buttons**: Start, pause, resume, and reset the simulation.
- **Detector Switch**: Classical CV or YOLO11n; availability is probed from
  `GET /detector` and synced over the socket.
- **Dataset Generator**: On-demand YOLO training data via `WS /ws/dataset`,
  with progress, cancel, and training next-steps.
- **Experiment History**: Run headless experiments (`POST /experiments/run`),
  browse saved runs, inspect exact metrics, compare two runs side-by-side,
  and plot error-vs-time convergence (dependency-free SVG).
- **Mission Mode**: Run the scripted 7-phase mission (`POST /mission/run`)
  and inspect the phase timeline plus final PAT/link report.
- **Pause/Resume**: `pause` acks enter the `PAUSED` state immediately (backend
  halts telemetry while paused); `resume` acks return to running. Ticks use
  sim-time `dt`, so pauses and reconnects never corrupt timing.

## Scripts

```bash
npm run dev    # dev server (Turbopack)
npm run build  # production build
npm run start  # serve production build
npm test       # vitest unit tests (pure helpers)
npm run typecheck  # tsc --noEmit
npm run lint   # next lint
```

## Architecture

```
Python Backend (FastAPI: REST + WebSocket)
        ↓  ws://…/ws/simulation (telemetry at the nominal 30 Hz target rate + base64-JPEG frames every 3rd tick)
        ↓  http://…/detector, /experiments/*, /mission/run
Next.js Frontend (src/app)
        ↓
React Dashboard Components (page + PresetButtons, DetectorSelector,
  DatasetGenerator, ExperimentHistory, ConvergenceGraph, MissionPanel)
```

## Troubleshooting

- **○ DISCONNECTED / CAMERA OFFLINE**: backend is down or unreachable —
  check the URL in `.env.local`, then press RETRY in the header.
- **YOLO shows Unavailable**: backend runs but `models/best.pt` is missing
  (see backend training guide) — Classical CV keeps working.
- **Experiment run fails**: check backend logs; large tick counts with YOLO
  take minutes (server limit: `EXPERIMENT_RUN_TIMEOUT_S`).
