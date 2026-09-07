"""
Main simulation engine that orchestrates all components.
"""
import asyncio
import logging
from typing import Optional

from .target import Target
from .camera import VirtualCamera
from .detector import BeaconDetector
from .tracker import Tracker
from .pat_controller import PATController
from .pan_tilt import PanTilt
from .fsoc import FSOCommunication
from .disturbances import DisturbanceModel
from .telemetry import TelemetryCollector
from models.schemas import SimulationConfig


class SimulationEngine:
    """Main simulation engine for FSO satellite tracking."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or SimulationConfig()
        
        # Initialize components
        self.target = Target()
        self.camera = VirtualCamera()
        self.detector = BeaconDetector()
        self.tracker = Tracker()
        self.pat_controller = PATController()
        self.pan_tilt = PanTilt()
        self.fsoc = FSOCommunication()
        self.disturbances = DisturbanceModel()
        self.telemetry = TelemetryCollector()
        
        self.running = False
        self.simulation_time = 0.0
        self.dt = 0.01  # Time step in seconds
        
    async def initialize(self):
        """Initialize all simulation components."""
        self.logger.info("Initializing simulation components")
        
        # Initialize each component
        await self.target.initialize()
        await self.camera.initialize()
        await self.detector.initialize()
        await self.tracker.initialize()
        await self.pat_controller.initialize()
        await self.pan_tilt.initialize()
        await self.fsoc.initialize()
        
        self.logger.info("All components initialized")
        
    async def step(self):
        """Execute one simulation step."""
        # Update target position
        target_position = self.target.update(self.dt)
        
        # Apply disturbances
        disturbances = self.disturbances.generate(self.simulation_time)
        
        # Capture camera frame
        frame = await self.camera.capture()
        
        # Detect target in frame
        detection = await self.detector.detect(frame)
        
        # Update tracker
        tracking_state = await self.tracker.update(detection, self.dt)
        
        # PAT controller computes corrections
        control_signal = await self.pat_controller.compute(
            tracking_state, 
            disturbances
        )
        
        # Apply control to pan-tilt mechanism
        await self.pan_tilt.apply_control(control_signal, self.dt)
        
        # Update FSO communication link
        link_quality = await self.fsoc.update(
            tracking_state,
            self.pan_tilt.get_state()
        )
        
        # Collect telemetry
        await self.telemetry.record(
            time=self.simulation_time,
            target_position=target_position,
            tracking_state=tracking_state,
            control_signal=control_signal,
            link_quality=link_quality
        )
        
        self.simulation_time += self.dt
        
    async def run(self, duration: Optional[float] = None):
        """
        Run the simulation.
        
        Args:
            duration: Simulation duration in seconds (None for infinite)
        """
        await self.initialize()
        
        self.running = True
        self.logger.info("Starting simulation loop")
        
        try:
            while self.running:
                await self.step()
                
                if duration and self.simulation_time >= duration:
                    self.logger.info(f"Simulation completed: {duration}s")
                    break
                    
                # Small sleep to prevent CPU overload
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.logger.error(f"Simulation error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown simulation and cleanup resources."""
        self.logger.info("Shutting down simulation")
        self.running = False
        
        # Save telemetry
        await self.telemetry.save()
        
        self.logger.info("Simulation shutdown complete")
