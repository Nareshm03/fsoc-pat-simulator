"""
FastAPI backend for FSOC PAT Simulation.
Integrates existing simulation modules with WebSocket streaming.
"""
import asyncio
import logging
import time
import base64
import cv2
import math
from typing import Set, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import existing simulation modules
from simulation.target import Target
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.tracker import Tracker
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATStateMachine
from simulation.logger import setup_logging
from dataset_generator import DatasetGenerator, DatasetConfig


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
        self.detector = BeaconDetector()
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
        
        # Reset state
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
            
            # 2. Get target ANGULAR position (absolute in space)
            target_az_rad, target_el_rad = self.target.angular_position(self.simulation_time)
            
            # 3. Calculate disturbances (applied to gimbal or relative angle)
            turb_x, turb_y = self.disturbances.turbulence_offset(self.simulation_time)
            vib_x, vib_y = self.disturbances.vibration_offset(self.simulation_time)
            cam_x, cam_y = self.disturbances.camera_offset(self.simulation_time)
            noise_x, noise_y = self.disturbances.sensor_offset()
            
            # 4. Calculate where target appears in camera frame given current gimbal angle
            # CLOSED-LOOP: Gimbal angle affects what the camera sees
            gimbal_az_rad = math.radians(self.pan_tilt.azimuth)
            gimbal_el_rad = math.radians(self.pan_tilt.elevation)
            
            # Calculate pixel position based on relative angle
            pixel_pos = self.target.pixel_from_angles(
                target_az_rad, target_el_rad,
                gimbal_az_rad, gimbal_el_rad
            )
            
            if pixel_pos is not None:
                beacon_x, beacon_y = pixel_pos
                
                # Apply disturbances to pixel position
                beacon_x += turb_x + vib_x + cam_x + noise_x
                beacon_y += turb_y + vib_y + cam_y + noise_y
                
                # Ground truth is where target WOULD be without disturbances
                self.ground_truth_x, self.ground_truth_y = pixel_pos
            else:
                # Target is out of FOV
                beacon_x, beacon_y = -1, -1  # Off-screen
                self.ground_truth_x, self.ground_truth_y = -1, -1
            
            # 5. Render virtual camera frame at the calculated position
            frame = self.camera.render(beacon_x, beacon_y)
            
            # 6. Run detector on the generated frame
            detection = self.detector.detect(frame)
            
            if detection:
                self.detected = True
                self.detected_x = detection["x"]
                self.detected_y = detection["y"]
                self.detection_confidence = detection["confidence"]
            else:
                self.detected = False
                self.detected_x = 0.0
                self.detected_y = 0.0
                self.detection_confidence = 0.0
            
            # 7. Update tracker
            tracking_result = self.tracker.update(detection)
            
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
            
            # 8. Calculate tracking error (from camera center)
            camera_center_x = self.camera.width / 2
            camera_center_y = self.camera.height / 2
            
            if self.detected:
                error_x = self.detected_x - camera_center_x
                error_y = self.detected_y - camera_center_y
                self.tracking_error_px = math.sqrt(error_x**2 + error_y**2)
            else:
                self.tracking_error_px = 0.0
            
            # 9. PAT State Machine Update
            pat_state = self.pat_state_machine.update(
                detected=self.detected,
                error_px=self.tracking_error_px,
                current_time=self.simulation_time,
                simulation_running=self.running and not self.paused
            )
            
            # Get control mode from state machine
            control_mode = self.pat_state_machine.get_control_mode()
            
            # 10. PAT Control Logic - compute gimbal commands
            az_rate_command, el_rate_command = self.pat_controller.compute_command(
                detected=self.detected,
                beacon_x=self.detected_x,
                beacon_y=self.detected_y,
                control_mode=control_mode,
                dt=self.dt
            )
            
            # 11. Update gimbal with velocity commands
            # CLOSED-LOOP: Gimbal movement will change camera view in next tick
            self.pan_tilt.update(az_rate_command, el_rate_command, self.dt)
            
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
        return {
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
                "confidence": round(self.detection_confidence, 3)
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
            "performance": {
                "simulation_fps": self.target_fps,  # Configured simulation rate (Hz)
                "processing_fps": round(self.actual_fps, 1)  # Actual backend throughput
            }
        }
    
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
                await client.send_json(message)
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
                    await websocket.send_json({
                        "type": "progress",
                        "current": successful,
                        "total": num_images,
                        "percentage": successful / num_images * 100,
                        "split": sample_info['split'],
                        "filename": sample_info['filename'],
                        "detected": sample_info['detected'],
                        "confidence": sample_info['confidence']
                    })
                    
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
            await websocket.send_json({
                "type": "complete",
                "stats": generator.stats,
                "validation": validation_results,
                "output_dir": str(base_path.absolute())
            })
            
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
