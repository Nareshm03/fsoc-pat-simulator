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

Headless smoke test (backend dir, server running):

```bash
python test_frontend_integration.py   # WS closed-loop demo
python test_camera.py test_detector.py test_target.py  # offline component smoke
```

## Architecture

```
Target(angular traj) + DisturbanceModel(turb/vib/motion/noise)
  → pixel_from_angles(gimbal feedback) → VirtualCamera 1280×720
  → BeaconDetector | YOLO11Detector → Tracker → PATStateMachine
  → PATController (PID) → PanTilt ─┘ (loop closes next tick)

backend/main.py ── WS /ws/simulation (telemetry @30 Hz + JPEG every 3rd frame)
                ── WS /ws/dataset (YOLO dataset generation with progress)
                ── REST /, /health, /detector, /experiments[/run, /{id}]
frontend/src/app ── page.tsx + PresetButtons, DetectorSelector,
                    DatasetGenerator, ExperimentHistory, ConvergenceGraph
backend/analytics ── experiment_metrics → runner → store → API
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
