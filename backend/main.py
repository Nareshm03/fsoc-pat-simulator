"""
FastAPI backend for FSOC PAT Simulation.
Integrates existing simulation modules with WebSocket streaming.
"""
import asyncio
import logging
import os
import time
import base64
import cv2
import math
from typing import Set, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


def sanitize_json_numbers(value):
    """Recursively replace non-finite floats with None for strict JSON.

    Browsers reject bare Infinity/-Infinity/NaN tokens, which Python's
    json emits for out-of-range floats (e.g. unbounded pointing loss).
    Finite values pass through untouched; strings (incl. base64 images)
    are never altered. Idempotent.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_numbers(item) for item in value]
    return value

# Import existing simulation modules
from simulation.target import Target
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.yolo_detector import YOLO11Detector, AVAILABLE_DETECTORS
from simulation.tracker import Tracker
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATStateMachine
from simulation.search_pattern import RasterSearch
from simulation.sim_step import simulation_step
from simulation.async_detector import AsyncDetector
from simulation.link_budget import LinkBudgetConfig
from simulation.logger import setup_logging
from dataset_generator import DatasetGenerator, DatasetConfig
from analytics.experiment_api import router as experiments_router
from analytics.mission_api import router as mission_router


class SimulationManager:
    """
    Manages the authoritative simulation state and lifecycle.
    This is a singleton - all WebSocket clients share the same simulation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.running = False
        self.paused = False
        self.simulation_time = 0.0
        self.frame_number = 0
        
        # Timing
        self.target_fps = 30
        self.dt = 1.0 / self.target_fps
        self.last_frame_time = 0.0
        self.actual_fps = 0.0
        
        # Initialize existing simulation components
        self.target = Target(width=1280, height=720)
        self.camera = VirtualCamera(width=1280, height=720)
        # Detector selection: Classical CV is default and always available.
        # YOLO11 is lazy-loaded on demand so missing deps/model never
        # breaks the existing PAT pipeline.
        self.detector_type: str = "classical"
        self.classical_detector = BeaconDetector()
        self.yolo_detector: Optional[YOLO11Detector] = None
        self.detector = self.classical_detector
        self.detection_bbox = None
        self.detector_error: Optional[str] = None
        self.tracker = Tracker()
        self.disturbances = DisturbanceModel(
            turbulence=1.0,
            vibration=1.0,
            camera_motion=1.0,
            sensor_noise=1.0
        )
        self.pan_tilt = PanTilt()
        self.pat_controller = PATController()
        self.pat_state_machine = PATStateMachine()
        # Lost-beacon raster sweep, active only in SEARCH/REACQUIRE modes.
        self.search_pattern = RasterSearch()
        # Optical link budget: default configuration, reused every tick.
        # Read-only w.r.t. PAT state; link_state derives from margin only.
        self.link_config = LinkBudgetConfig()
        self.pointing_error_rad = 0.0
        self.link_result = None

        # Current state data
        self.ground_truth_x = 0.0
        self.ground_truth_y = 0.0
        self.detected_x = 0.0
        self.detected_y = 0.0
        self.tracked_x = 0.0
        self.tracked_y = 0.0
        self.detected = False
        self.detection_confidence = 0.0
        self.tracking_vx = 0.0
        self.tracking_vy = 0.0
        self.tracking_error_px = 0.0
        
        # Connected clients
        self.clients: Set[WebSocket] = set()
        
        # Simulation task
        self.simulation_task: Optional[asyncio.Task] = None

        # Background YOLO warmup task (kept referenced so it cannot be
        # garbage-collected mid-flight; see _schedule_yolo_warmup).
        self._yolo_warmup_task: Optional[asyncio.Task] = None

        # Bounded async YOLO pipeline (created lazily on first YOLO tick;
        # Classical ticks never touch it). Holds the newest-frame worker
        # plus the explicit observation-age bookkeeping surfaced below.
        self._async_yolo: Optional[AsyncDetector] = None
        self.obs_age_ticks: Optional[int] = None
        self.obs_fresh: bool = False

        self.logger.info("SimulationManager initialized")
    
    def reset(self):
        """Reset simulation to initial state."""
        self.logger.info("Resetting simulation")
        
        self.simulation_time = 0.0
        self.frame_number = 0
        self.last_frame_time = 0.0
        
        # Reset components
        self.tracker = Tracker()
        self.pan_tilt = PanTilt()
        self.pat_controller.reset()
        self.pat_state_machine.reset()
        
        # Reset state (detector selection is preserved across reset)
        self.ground_truth_x = 0.0
        self.ground_truth_y = 0.0
        self.detected_x = 0.0
        self.detected_y = 0.0
        self.tracked_x = 0.0
        self.tracked_y = 0.0
        self.detected = False
        self.detection_confidence = 0.0
        self.detection_bbox = None
        self.tracking_vx = 0.0
        self.tracking_vy = 0.0
        self.tracking_error_px = 0.0
        self.pointing_error_rad = 0.0
        self.link_result = None
        # Drop any in-flight/completed async observations: sim_time
        # restarts at 0, so old timestamps are meaningless afterwards.
        if self._async_yolo is not None:
            self._async_yolo.reset_generation()
        self.obs_age_ticks = None
        self.obs_fresh = False

    async def start(self):
        """Start the simulation."""
        if self.running:
            self.logger.warning("Simulation already running")
            return
        
        self.logger.info("Starting simulation")
        self.running = True
        self.paused = False
        self.last_frame_time = time.time()
        
        if self.simulation_task is None or self.simulation_task.done():
            self.simulation_task = asyncio.create_task(self._simulation_loop())
    
    async def pause(self):
        """Pause the simulation."""
        self.logger.info("Pausing simulation")
        self.paused = True
    
    async def resume(self):
        """Resume the simulation."""
        self.logger.info("Resuming simulation")
        self.paused = False
        self.last_frame_time = time.time()
    
    async def stop(self):
        """Stop the simulation."""
        self.logger.info("Stopping simulation")
        self.running = False
        self.paused = False
        
        if self.simulation_task and not self.simulation_task.done():
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass

        # Never leave the async detection worker running past shutdown.
        if self._async_yolo is not None:
            await self._async_yolo.shutdown()
    
    def set_disturbance(self, name: str, value: float):
        """
        Set disturbance parameter.
        
        This works regardless of simulation state (running, paused, or stopped).
        Changes apply immediately to the configuration and will affect
        the simulation when it's running or resumed.
        """
        if name == "turbulence":
            self.disturbances.turbulence = value
        elif name == "vibration":
            self.disturbances.vibration = value
        elif name == "camera_motion":
            self.disturbances.camera_motion = value
        elif name == "sensor_noise":
            self.disturbances.sensor_noise = value
        else:
            raise ValueError(f"Unknown disturbance: {name}")
        
        self.logger.info(f"Set {name} = {value}")

    def set_detector(self, detector_type: str) -> dict:
        """Switch active detector without touching PAT/tracker/gimbal/physics.

        Args:
            detector_type: "classical" | "yolo"

        Returns:
            Detector info dict for the newly active detector.

        Raises:
            ValueError: on unknown detector_type.
            RuntimeError: if requested detector is unavailable (e.g. YOLO
                model/deps missing). Active detector is left unchanged.
        """
        if detector_type not in AVAILABLE_DETECTORS:
            raise ValueError(
                f"Unknown detector: {detector_type}. Use {list(AVAILABLE_DETECTORS)}"
            )
        if detector_type == "classical":
            self.detector = self.classical_detector
            self.detector_type = "classical"
            self.detector_error = None
            # Retire async observations: a stale YOLO result must never
            # be applied after switching away (or back).
            if self._async_yolo is not None:
                self._async_yolo.reset_generation()
            self.logger.info("Detector switched to classical")
            return self.get_detector_info()
        # yolo: construct cheaply (model load deferred), switch
        # synchronously, then warm the model in the background so the
        # first live tick does not pay initialization/inference cost on
        # the event loop. Reuses the cached instance across switches.
        if self.yolo_detector is None:
            self.yolo_detector = YOLO11Detector(defer_load=True)
        if self.yolo_detector.model_path is None:
            raise RuntimeError(
                f"YOLO detector unavailable: {self.yolo_detector.load_error}"
            )
        try:
            import importlib.util as _ilu

            _has_ultralytics = _ilu.find_spec("ultralytics") is not None
        except Exception:
            _has_ultralytics = False
        if not _has_ultralytics:
            raise RuntimeError(
                "YOLO detector unavailable: "
                "ultralytics not installed (pip install ultralytics)"
            )
        self.detector = self.yolo_detector
        self.detector_type = "yolo"
        self.detector_error = None
        # Fresh epoch for the async pipeline: anything inferred under a
        # previous selection is stale by definition. Harvest every ~0.1 s
        # of simulation time (10 Hz regular at any dt) keeps correction
        # jumps periodic instead of jittered so the 2 s lock timer can
        # sustain; fresh negatives still apply immediately and expiry is
        # unchanged.
        if self._async_yolo is None or \
                self._async_yolo.detector is not self.yolo_detector:
            self._async_yolo = AsyncDetector(self.yolo_detector,
                                             self.target,
                                             harvest_interval_s=0.1)
        else:
            self._async_yolo.reset_generation()
        self.logger.info("Detector switched to yolo")
        self._schedule_yolo_warmup()
        return self.get_detector_info()

    def _schedule_yolo_warmup(self) -> None:
        """Warm the cached YOLO model in a worker thread (stale-safe).

        Warmup mutates only the cached YOLO11Detector instance (model
        load is idempotent), never manager detector-selection state, so
        a stale completion after rapid switching cannot corrupt anything.
        With no running loop (scripts/tests), this is a silent no-op and
        the existing lazy load inside detect() remains the fallback.
        """
        detector = self.yolo_detector
        if detector is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _warm() -> bool:
            try:
                return await asyncio.to_thread(detector.warmup)
            except Exception:
                self.logger.warning(
                    "YOLO background warmup failed", exc_info=True
                )
                return False

        self._yolo_warmup_task = loop.create_task(_warm())

    def get_detector_info(self) -> dict:
        """Return active detector + availability without side effects."""
        yolo_available = False
        yolo_error = None
        yolo_path = None
        if self.yolo_detector is not None:
            yolo_available = self.yolo_detector.model is not None
            yolo_error = self.yolo_detector.load_error
            yolo_path = (
                str(self.yolo_detector.model_path)
                if self.yolo_detector.model_path
                else None
            )
        else:
            # Probe availability cheaply without loading torch:
            # model file present + ultralytics importable.
            from pathlib import Path as _Path

            default_pt = (
                _Path(__file__).resolve().parent / "models" / "best.pt"
            )
            default_zip = (
                _Path(__file__).resolve().parent / "models" / "best.pt.zip"
            )
            yolo_path = str(default_pt)
            try:
                import importlib.util as _ilu

                has_ultralytics = _ilu.find_spec("ultralytics") is not None
            except Exception:
                has_ultralytics = False
            yolo_available = bool(
                (default_pt.exists() or default_zip.exists()) and has_ultralytics
            )
            if not yolo_available:
                if not (default_pt.exists() or default_zip.exists()):
                    yolo_error = f"Model file not found: {default_pt}"
                elif not has_ultralytics:
                    yolo_error = "ultralytics not installed (pip install ultralytics)"
        return {
            "active": self.detector_type,
            "available": list(AVAILABLE_DETECTORS),
            "classical_available": True,
            "yolo_available": yolo_available,
            "yolo_model": yolo_path,
            "yolo_error": yolo_error,
            "error": self.detector_error,
        }

    async def _simulation_loop(self):
        """Main simulation loop."""
        self.logger.info("Simulation loop started")
        
        try:
            while self.running:
                if not self.paused:
                    await self._simulation_tick()
                else:
                    await asyncio.sleep(0.01)  # Small sleep when paused
                
                # Control frame rate
                await asyncio.sleep(max(0, self.dt - 0.001))
                
        except asyncio.CancelledError:
            self.logger.info("Simulation loop cancelled")
        except Exception as e:
            self.logger.error(f"Simulation loop error: {e}", exc_info=True)
            self.running = False
    
    async def _simulation_tick(self):
        """Execute one simulation tick with closed-loop PAT control."""
        frame_start_time = time.time()
        
        try:
            # 1. Advance simulation time
            self.simulation_time += self.dt
            self.frame_number += 1
            
            # 2-11. Shared closed-loop tick (target geometry ->
            # disturbance -> render -> detect -> error -> PAT state machine
            # -> search/PID control -> gimbal update -> link budget).
            # Live-loop semantics: raster search, feedforward, tracker,
            # link budget, out-of-FOV render through the real detector.
            use_async_yolo = (
                self.detector_type == "yolo"
                and self.yolo_detector is not None
                and self.detector is self.yolo_detector
            )
            if use_async_yolo:
                # Bounded async YOLO path: the tick runs at control
                # cadence while inference runs newest-frame-only in the
                # background. simulation_step is identical; only the
                # detector argument is the async shim, which never blocks
                # and never invents detections (see AsyncDetector).
                if (self._async_yolo is None or
                        self._async_yolo.detector is not self.yolo_detector):
                    self._async_yolo = AsyncDetector(self.yolo_detector,
                                                     self.target,
                                                     harvest_interval_s=0.1)
                self._async_yolo.begin_tick(
                    self.simulation_time, self.frame_number,
                    math.radians(self.pan_tilt.azimuth),
                    math.radians(self.pan_tilt.elevation),
                )
                self._async_yolo.pump()
                out = simulation_step(
                    target=self.target,
                    target_override=None,
                    disturbances=self.disturbances,
                    camera=self.camera,
                    detector=self._async_yolo,
                    tracker=self.tracker,
                    pan_tilt=self.pan_tilt,
                    controller=self.pat_controller,
                    state_machine=self.pat_state_machine,
                    search=self.search_pattern,
                    link_config=self.link_config,
                    sim_time=self.simulation_time,
                    dt=self.dt,
                    simulation_running=self.running and not self.paused,
                    blank=False,
                    detect_when_out_of_fov=True,
                    use_raster_search=True,
                    feedforward=True,
                )
                self.obs_age_ticks = self._async_yolo.last_age_ticks
                self.obs_fresh = self._async_yolo.last_fresh
                # Fairness checkpoint: this branch performs no awaits of
                # its own (the worker thread does the inference), so yield
                # once to keep the loop responsive even with zero clients
                # attached and no other yield points in the tick.
                await asyncio.sleep(0)
            else:
                # Classical (and any non-YOLO) path, byte-identical to
                # before: the whole CPU-bound tick runs in a worker
                # thread so the event loop keeps serving WS/HTTP
                # clients. Same function, same order, same dt.
                out = await asyncio.to_thread(
                    simulation_step,
                    target=self.target,
                    target_override=None,
                    disturbances=self.disturbances,
                    camera=self.camera,
                    detector=self.detector,
                    tracker=self.tracker,
                    pan_tilt=self.pan_tilt,
                    controller=self.pat_controller,
                    state_machine=self.pat_state_machine,
                    search=self.search_pattern,
                    link_config=self.link_config,
                    sim_time=self.simulation_time,
                    dt=self.dt,
                    simulation_running=self.running and not self.paused,
                    blank=False,
                    detect_when_out_of_fov=True,
                    use_raster_search=True,
                    feedforward=True,
                )
                # Synchronous detection of the current frame: age 0, fresh.
                self.obs_age_ticks = 0
                self.obs_fresh = True
            detection = out["detection"]
            frame = out["frame"]
            pat_state = out["pat_state"]

            # Ground truth is where target WOULD be without disturbances.
            if out["ground_truth_xy"] is not None:
                self.ground_truth_x, self.ground_truth_y = out["ground_truth_xy"]
            else:
                # Target is out of FOV
                self.ground_truth_x, self.ground_truth_y = -1, -1

            self.pointing_error_rad = out["pointing_error_rad"]
            self.link_result = out["link"]

            if detection:
                self.detected = True
                self.detected_x = out["dx"]
                self.detected_y = out["dy"]
                self.detection_confidence = out["confidence"]
                # bbox is optional (YOLO provides it, Classical does not).
                # PAT/tracker only use x/y/confidence, so this is additive.
                self.detection_bbox = detection.get("bbox")
            else:
                self.detected = False
                self.detected_x = 0.0
                self.detected_y = 0.0
                self.detection_confidence = 0.0
                self.detection_bbox = None

            tracking_result = out["tracking"]

            if tracking_result:
                self.tracked_x = tracking_result["x"]
                self.tracked_y = tracking_result["y"]
                self.tracking_vx = tracking_result["vx"]
                self.tracking_vy = tracking_result["vy"]
            else:
                self.tracked_x = 0.0
                self.tracked_y = 0.0
                self.tracking_vx = 0.0
                self.tracking_vy = 0.0

            self.tracking_error_px = out["error_px"]
            
            # 12. Calculate FSOC status
            fsoc_status = self._calculate_fsoc_status()
            
            # 13. Build telemetry
            telemetry = self._build_telemetry(pat_state.value, fsoc_status)
            
            # 14. Broadcast telemetry
            await self._broadcast(telemetry)
            
            # 15. Broadcast camera frame (every 3rd frame to reduce bandwidth)
            if self.frame_number % 3 == 0:
                frame_message = self._build_frame_message(frame)
                if frame_message:  # Only broadcast if encoding succeeded
                    await self._broadcast(frame_message)
            
            # Calculate actual FPS
            frame_time = time.time() - frame_start_time
            if frame_time > 0:
                self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / frame_time)
            
        except Exception as e:
            self.logger.error(f"Simulation tick error: {e}", exc_info=True)
    
    def _calculate_fsoc_status(self) -> str:
        """Calculate FSOC link status based on tracking error."""
        if not self.detected:
            return "LOST"
        elif self.tracking_error_px < 3.0:
            return "ACTIVE"
        elif self.tracking_error_px < 10.0:
            return "DEGRADED"
        else:
            return "LOST"
    
    def _build_telemetry(self, pat_state: str, fsoc_status: str) -> dict:
        """Build telemetry message."""
        telemetry = {
            "type": "telemetry",
            "simulation_time": round(self.simulation_time, 3),
            "frame": self.frame_number,
            "simulation": {
                "running": self.running,
                "paused": self.paused
            },
            "target": {
                "x": round(self.ground_truth_x, 2),
                "y": round(self.ground_truth_y, 2)
            },
            "detection": {
                "detected": self.detected,
                "x": round(self.detected_x, 2) if self.detected else 0.0,
                "y": round(self.detected_y, 2) if self.detected else 0.0,
                "confidence": round(self.detection_confidence, 3),
                "bbox": (
                    [round(float(v), 2) for v in self.detection_bbox]
                    if self.detected and self.detection_bbox
                    else None
                ),
                "detector_type": self.detector_type,
                # Async-pipeline provenance (additive): age in ticks of
                # the observation backing this tick (0/fresh for the
                # synchronous classical path; null when undetected).
                "observation_age_ticks": self.obs_age_ticks,
                "observation_fresh": self.obs_fresh,
            },
            "detector": {
                "type": self.detector_type,
            },
            "tracking": {
                "x": round(self.tracked_x, 2),
                "y": round(self.tracked_y, 2),
                "vx": round(self.tracking_vx, 2),
                "vy": round(self.tracking_vy, 2)
            },
            "error": {
                "x": round(self.detected_x - self.ground_truth_x, 2) if self.detected else 0.0,
                "y": round(self.detected_y - self.ground_truth_y, 2) if self.detected else 0.0,
                "pixel": round(self.tracking_error_px, 2),
                "angular_deg": round(self.tracking_error_px * 0.05, 3)  # Approximate
            },
            "pat": {
                "state": pat_state
            },
            "gimbal": {
                "azimuth": round(self.pan_tilt.azimuth, 2),
                "elevation": round(self.pan_tilt.elevation, 2),
                "azimuth_velocity": round(self.pan_tilt.azimuth_velocity, 2),
                "elevation_velocity": round(self.pan_tilt.elevation_velocity, 2)
            },
            "disturbances": {
                "turbulence": self.disturbances.turbulence,
                "vibration": self.disturbances.vibration,
                "camera_motion": self.disturbances.camera_motion,
                "sensor_noise": self.disturbances.sensor_noise
            },
            "fsoc": {
                "status": fsoc_status
            },
            "link": (
                {
                    "received_power_w": self.link_result["received_power_w"],
                    "pointing_loss_db": round(self.link_result["pointing_loss_db"], 3),
                    "atmospheric_loss_db": round(self.link_result["atmospheric_loss_db"], 3),
                    "link_margin_db": round(self.link_result["link_margin_db"], 3),
                    "snr_db": round(self.link_result["snr_db"], 3),
                    "link_state": self.link_result["link_state"],
                }
                if self.link_result is not None
                else None
            ),
            "performance": {
                "simulation_fps": self.target_fps,  # Configured simulation rate (Hz)
                "processing_fps": round(self.actual_fps, 1)  # Actual backend throughput
            }
        }
        # Strict-JSON guarantee for direct sends (broadcast sanitizes again).
        return sanitize_json_numbers(telemetry)

    def _build_frame_message(self, frame) -> dict:
        """Build camera frame message with base64-encoded JPEG."""
        # Encode frame as JPEG with lower quality to reduce size
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        
        if not success:
            self.logger.error("Failed to encode frame as JPEG")
            return None
        
        jpg_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "type": "frame",
            "frame": self.frame_number,
            "simulation_time": round(self.simulation_time, 3),
            "image": jpg_base64
        }
    
    async def _broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        message_type = message.get("type", "unknown")
        
        if not self.clients:
            # self.logger.debug(f"No clients to broadcast {message_type}")
            return
        
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send_json(sanitize_json_numbers(message))
                # self.logger.debug(f"Sent {message_type} to client")
            except Exception as e:
                self.logger.error(f"Error sending {message_type} to client: {e}", exc_info=True)
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    def add_client(self, websocket: WebSocket):
        """Register a new client."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    def remove_client(self, websocket: WebSocket):
        """Unregister a client."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")


# Global simulation manager instance
simulation_manager: Optional[SimulationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global simulation_manager
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting FSOC PAT Simulation Backend")
    
    # Initialize simulation manager
    simulation_manager = SimulationManager()
    
    yield
    
    # Cleanup
    if simulation_manager:
        await simulation_manager.stop()
    
    logger.info("FSOC PAT Simulation Backend shut down")


# Create FastAPI application
app = FastAPI(
    title="FSOC PAT Simulation",
    description="Free Space Optical Communication Point, Acquisition, and Tracking Simulator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS origins are explicit (never "*") so allow_credentials stays valid.
# Configure via FSOC_CORS_ORIGINS as a comma-separated list; defaults
# preserve local development (Next.js dev server on port 3000).
def _get_cors_origins() -> list:
    raw = os.environ.get(
        "FSOC_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Experiment + Mission REST APIs (analytics only; WebSocket endpoints unchanged below).
app.include_router(experiments_router)
app.include_router(mission_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FSOC PAT Simulation",
        "status": "online"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "simulation_running": simulation_manager.running if simulation_manager else False,
        "connected_clients": len(simulation_manager.clients) if simulation_manager else 0
    }


@app.get("/detector")
async def detector_info():
    """Detector availability for frontend (no model load, no PAT side effects)."""
    if simulation_manager is not None:
        return simulation_manager.get_detector_info()
    # Lifespan not run (e.g. fresh import): lightweight probe only.
    from pathlib import Path as _Path

    default_pt = _Path(__file__).resolve().parent / "models" / "best.pt"
    default_zip = _Path(__file__).resolve().parent / "models" / "best.pt.zip"
    try:
        import importlib.util as _ilu

        has_ultralytics = _ilu.find_spec("ultralytics") is not None
    except Exception:
        has_ultralytics = False
    yolo_available = bool(
        (default_pt.exists() or default_zip.exists()) and has_ultralytics
    )
    yolo_error = None
    if not yolo_available:
        if not (default_pt.exists() or default_zip.exists()):
            yolo_error = f"Model file not found: {default_pt}"
        elif not has_ultralytics:
            yolo_error = "ultralytics not installed (pip install ultralytics)"
    return {
        "active": "classical",
        "available": list(AVAILABLE_DETECTORS),
        "classical_available": True,
        "yolo_available": yolo_available,
        "yolo_model": str(default_pt),
        "yolo_error": yolo_error,
        "error": None,
    }


@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation streaming."""
    logger = logging.getLogger(__name__)
    
    if not simulation_manager:
        logger.error("Simulation manager not initialized")
        await websocket.close(code=1011, reason="Simulation manager not available")
        return
    
    # Accept connection
    await websocket.accept()
    simulation_manager.add_client(websocket)
    
    try:
        # Send initial state
        if simulation_manager.running:
            initial_state = simulation_manager._build_telemetry(
                "SEARCH" if not simulation_manager.detected else "FINE_TRACK",
                simulation_manager._calculate_fsoc_status()
            )
            await websocket.send_json(initial_state)
        
        # Listen for commands
        while True:
            data = await websocket.receive_json()
            
            command = data.get("command")
            
            if command == "start":
                await simulation_manager.start()
                await websocket.send_json({"type": "ack", "command": "start"})
            
            elif command == "pause":
                await simulation_manager.pause()
                await websocket.send_json({"type": "ack", "command": "pause"})
            
            elif command == "resume":
                await simulation_manager.resume()
                await websocket.send_json({"type": "ack", "command": "resume"})
            
            elif command == "reset":
                simulation_manager.reset()
                await websocket.send_json({"type": "ack", "command": "reset"})
            
            elif command == "stop":
                await simulation_manager.stop()
                await websocket.send_json({"type": "ack", "command": "stop"})
            
            elif command == "set_disturbance":
                # Disturbance changes work regardless of paused state
                name = data.get("name")
                value = data.get("value")
                
                # Validate input
                if not isinstance(name, str) or name not in ["turbulence", "vibration", "camera_motion", "sensor_noise"]:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid disturbance name"
                    })
                    continue
                
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid disturbance value"
                    })
                    continue
                
                if value < 0 or value > 10:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Disturbance value out of range (0-10)"
                    })
                    continue
                
                simulation_manager.set_disturbance(name, value)
                logger.info(f"Disturbance {name} set to {value} (paused={simulation_manager.paused})")
                await websocket.send_json({"type": "ack", "command": "set_disturbance"})
            
            elif command == "set_target_speed":
                # Placeholder for future implementation
                await websocket.send_json({"type": "ack", "command": "set_target_speed"})

            elif command == "set_detector":
                # Frontend sends {command:'set_detector', detector_type:'yolo'|'classical'}
                # Accept detector_type / detector / value aliases.
                requested = (
                    data.get("detector_type")
                    or data.get("detector")
                    or data.get("value")
                )
                if not isinstance(requested, str) or requested not in AVAILABLE_DETECTORS:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid detector. Use {list(AVAILABLE_DETECTORS)}"
                    })
                    continue
                try:
                    info = simulation_manager.set_detector(requested)
                    await websocket.send_json({
                        "type": "ack",
                        "command": "set_detector",
                        "detector_type": requested,
                        "detector": info,
                    })
                except RuntimeError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif command == "get_detector":
                await websocket.send_json({
                    "type": "detector",
                    "detector": simulation_manager.get_detector_info(),
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown command: {command}"
                })
    
    except WebSocketDisconnect:
        simulation_manager.remove_client(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        simulation_manager.remove_client(websocket)


@app.websocket("/ws/dataset")
async def dataset_generation_websocket(websocket: WebSocket):
    """WebSocket endpoint for dataset generation with progress updates."""
    logger = logging.getLogger(__name__)
    
    await websocket.accept()
    logger.info("Dataset generation WebSocket connected")
    
    try:
        # Wait for generation command
        data = await websocket.receive_json()
        command = data.get("command")
        
        if command == "generate":
            num_images = data.get("num_images", 1000)
            
            logger.info(f"Starting dataset generation: {num_images} images")
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "command": "generate",
                "num_images": num_images
            })
            
            # Create dataset generator
            config = DatasetConfig(num_images=num_images)
            generator = DatasetGenerator(
                camera=simulation_manager.camera,
                detector=simulation_manager.detector,
                disturbances=simulation_manager.disturbances,
                config=config
            )
            
            # Setup directories
            base_path = generator.setup_directories()
            
            # Pre-assign splits
            split_assignments = generator._prepare_split_assignments()
            
            # Send start message
            await websocket.send_json({
                "type": "start",
                "output_dir": str(base_path.absolute()),
                "splits": {
                    "train": sum(1 for _, split in split_assignments if split == 'train'),
                    "val": sum(1 for _, split in split_assignments if split == 'val'),
                    "test": sum(1 for _, split in split_assignments if split == 'test')
                }
            })
            
            # Generate dataset with progress updates
            successful = 0
            for sample_id, split in split_assignments:
                try:
                    # Generate sample with explicit split
                    sample_info = generator.generate_sample(sample_id, split, base_path)
                    
                    if sample_info is None:
                        # Sample failed, send warning
                        await websocket.send_json({
                            "type": "warning",
                            "message": f"Sample {sample_id} ({split}) failed validation after retries"
                        })
                        continue
                    
                    successful += 1
                    
                    # Send progress update every image
                    await websocket.send_json(sanitize_json_numbers({
                        "type": "progress",
                        "current": successful,
                        "total": num_images,
                        "percentage": successful / num_images * 100,
                        "split": sample_info['split'],
                        "filename": sample_info['filename'],
                        "detected": sample_info['detected'],
                        "confidence": sample_info['confidence']
                    }))
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Error generating sample {sample_id} ({split}): {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error generating sample {sample_id}: {str(e)}"
                    })
            
            # Validate dataset
            validation_results = generator._validate_dataset(base_path)
            generator.stats['validation'] = validation_results
            
            # Generate YAML and README
            generator._generate_dataset_yaml(base_path)
            generator._generate_readme(base_path)
            
            # Generate contact sheet
            generator._generate_contact_sheet(base_path)
            
            # Send completion message
            await websocket.send_json(sanitize_json_numbers({
                "type": "complete",
                "stats": generator.stats,
                "validation": validation_results,
                "output_dir": str(base_path.absolute())
            }))
            
            logger.info(f"Dataset generation complete: {generator.stats}")
        
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown command: {command}"
            })
    
    except WebSocketDisconnect:
        logger.info("Dataset generation WebSocket disconnected")
    except Exception as e:
        logger.error(f"Dataset generation error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
