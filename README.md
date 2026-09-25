# FSOC PAT Simulator

Free Space Optical Communication Pointing, Acquisition, and Tracking
simulator: a closed-loop gimbal simulation (Python) with a real-time
dashboard (Next.js), dual beacon detectors (Classical CV + YOLO11n),
synthetic YOLO dataset generation, and a headless experiment workflow
(run → persist → compare → convergence graphs).

## Quickstart

Backend (from repo root):

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend (new terminal, from repo root):

```bash
cd frontend
npm install
npm run dev     # http://localhost:3000
```

Headless smoke test (from `backend/`, server running for the WS demo):

```bash
python test_frontend_integration.py   # WS closed-loop demo
python test_camera.py                 # offline camera smoke
python test_detector.py               # offline detector smoke
python test_target.py                 # offline target smoke
```

## Architecture

```
Target(angular traj) + DisturbanceModel(turb/vib/motion/noise)
  → pixel_from_angles(gimbal feedback) → VirtualCamera 1280×720
  → BeaconDetector | YOLO11Detector → Tracker → PATStateMachine
  → PATController (PID) → PanTilt ─┘ (loop closes next tick)

backend/main.py ── WS /ws/simulation (telemetry at the nominal 30 Hz target rate + JPEG every 3rd tick)
                ── WS /ws/dataset (YOLO dataset generation with progress)
                ── REST /, /health, /detector, /experiments[/run, /{id}], /mission/run
frontend/src/app ── page.tsx + PresetButtons, DetectorSelector,
                    DatasetGenerator, ExperimentHistory, ConvergenceGraph,
                    MissionPanel
backend/analytics ── experiment_metrics → runner → store → API
                  ── mission_runner → mission_api (7-phase scripted mission)
```

## API reference

REST (`http://localhost:8000` by default):

| Method | Path | Body / notes |
|---|---|---|
| GET | `/` | service info |
| GET | `/health` | `{status, simulation_running, connected_clients}` |
| GET | `/detector` | `{active, classical_available, yolo_available, …}` — works even before startup |
| POST | `/experiments/run` | `{detector: "classical"\|"yolo", scenario: name\|dict, seed: int, ticks: 1–2000, dt: float}` → saved result (limit: `EXPERIMENT_RUN_TIMEOUT_S`, default 1800 s) |
| GET | `/experiments` | saved experiments, oldest first |
| GET | `/experiments/{id}` | one saved experiment incl. per-frame `records` |
| POST | `/mission/run` | `{detector: "classical"\|"yolo", seed: int, dt: float}` → scripted 7-phase report: `acquisition → stationary_lock → moving → disturbed → loss → search_reacquire → final_lock` (limit: `MISSION_RUN_TIMEOUT_S`, default 1800 s) |

WebSocket `/ws/simulation` commands (JSON `{command, …}` → `ack`/`error`):

`start`, `pause`, `resume`, `reset`, `stop`,
`set_disturbance {name: turbulence\|vibration\|camera_motion\|sensor_noise, value: 0–10}`,
`set_detector {detector_type: classical\|yolo}`,
`get_detector`. Server streams `{type: telemetry}`, `{type: frame}` (base64 JPEG).

WebSocket `/ws/dataset`: `{command: generate, num_images}` → `ack/start/progress/complete`
(or `warning`/`error`) messages.

CORS origins are explicit via `FSOC_CORS_ORIGINS`
(default `http://localhost:3000,http://127.0.0.1:3000`); frontend URLs via
`NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_WS_URL` (see `frontend/.env.example`).

## Demo sequence

1. `START` — loop runs, PAT moves `SEARCH → ACQUIRE → FINE_TRACK → LOCKED`.
2. `PAUSE` — dashboard shows `PAUSED` immediately (backend halts telemetry while paused); `RESUME` is enabled.
3. `RESUME` — returns to running without needing `RESET`.
4. Detector switch to YOLO11 — `Warming…` banner until the first detections flow, then nominal tracking.
5. Raise turbulence — error grows; forced loss drives `REACQUIRE → SEARCH`, then reacquisition.
6. `RUN MISSION` (MissionPanel) — scripted 7-phase run ending in `COMPLETE` with final PAT/link report.

## Runtime notes

- PAT thresholds: `ACQUIRE → FINE_TRACK` below 50 px error, `LOCKED` after error stays below 5 px for 2 s, back to `FINE_TRACK` above 10 px, `REACQUIRE → SEARCH` after 3 s with no detection.
- YOLO runs off the control tick (newest-frame-only async worker, timestamped observations, 0.4 s observation-age bound); the tick uses sim-time `dt`, so runs stay deterministic.
- Optical link state derives from link margin only (`NOMINAL ≥ 3 dB`, `MARGINAL ≥ 0 dB`, else `OUTAGE`); under typical tracking residuals the link reads `OUTAGE` — expected from the configured 1000 km budget, not a tracking fault.
- Telemetry targets 30 Hz wall-clock; camera frames arrive as base64-JPEG `frame` messages every 3rd tick.
- Extra manual scripts live in repo-root `test_*.py` (need a running server where noted); `dataset_test/` is the 10-image smoke fixture — e.g. `cd backend && $env:PYTHONUTF8='1'; python test_dataset_integrity.py ../dataset_test` (UTF-8 mode avoids a Windows-console encoding crash in that script's checkmarks).

## Testing

```bash
cd backend && python -m pytest tests/   # full suite (YOLO tests skip if model/deps missing)
cd frontend && npm test                 # vitest unit tests
cd frontend && npm run typecheck && npm run build
```

Manual/live scripts live in `backend/test_*.py` (need a running server where
noted). `backend/dataset/` is the committed 10k-image training set;
`dataset_test/` is the 10-image smoke fixture.

## Docs

- `frontend/README.md` — dashboard guide + troubleshooting
- `backend/DATASET_GENERATION_GUIDE.md` — dataset CLI + Colab training flow
- `backend/dataset/COLAB_TRAINING_GUIDE.md` — training notebook walkthrough
