"""
WebSocket server for streaming simulation data to frontend.
"""
import asyncio
import json
import websockets
from typing import Set
import logging

from simulation.target import Target
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.tracker import Tracker
from simulation.disturbances import DisturbanceModel


class SimulationWebSocketServer:
    """WebSocket server for real-time simulation streaming."""
    
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        
        # Initialize simulation components
        self.target = Target()
        self.camera = VirtualCamera()
        self.detector = BeaconDetector()
        self.tracker = Tracker()
        self.disturbances = DisturbanceModel()
        
        self.time = 0.0
        self.dt = 0.033  # ~30 FPS
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def register(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """Unregister a client connection."""
        self.clients.remove(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_state(self, state_data: dict):
        """Broadcast state to all connected clients."""
        if self.clients:
            message = json.dumps(state_data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def simulation_loop(self):
        """Main simulation loop."""
        self.logger.info("Starting simulation loop")
        fps_counter = 0
        last_fps_time = 0.0
        current_fps = 0.0
        
        while self.running:
            # Update target position
            true_x, true_y = self.target.position(self.time)
            
            # Apply disturbances
            turb_x, turb_y = self.disturbances.turbulence_offset(self.time)
            vib_x, vib_y = self.disturbances.vibration_offset(self.time)
            cam_x, cam_y = self.disturbances.camera_offset(self.time)
            noise_x, noise_y = self.disturbances.sensor_offset()
            
            # Observed position with disturbances
            observed_x = true_x + turb_x + vib_x + cam_x + noise_x
            observed_y = true_y + turb_y + vib_y + cam_y + noise_y
            
            # Render camera frame
            frame = self.camera.render(observed_x, observed_y)
            
            # Detect beacon
            detection = self.detector.detect(frame)
            
            # Update tracker
            tracking = self.tracker.update(detection)
            
            # Calculate tracking error
            if detection:
                error = ((detection['x'] - true_x)**2 + (detection['y'] - true_y)**2)**0.5
            else:
                error = 999.9
            
            # Calculate FPS
            fps_counter += 1
            if self.time - last_fps_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                last_fps_time = self.time
            
            # Prepare state data
            state_data = {
                'state': 'LOCKED' if detection else 'SEARCHING',
                'beacon_x': detection['x'] if detection else 0,
                'beacon_y': detection['y'] if detection else 0,
                'error': error,
                'fps': current_fps,
                'azimuth': 0.0,  # TODO: Add pan/tilt integration
                'elevation': 0.0,
                'confidence': detection['confidence'] if detection else 0,
                'link_status': 'ACTIVE' if detection and error < 5 else 'DEGRADED',
                'time': self.time
            }
            
            # Send to all clients
            await self.send_state(state_data)
            
            # Increment time
            self.time += self.dt
            
            # Control frame rate
            await asyncio.sleep(self.dt)
    
    async def handle_client(self, websocket, path):
        """Handle individual client connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Handle client commands (start, pause, reset, etc.)
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    
                    if command == 'start':
                        if not self.running:
                            self.running = True
                            asyncio.create_task(self.simulation_loop())
                            self.logger.info("Simulation started")
                    
                    elif command == 'pause':
                        self.running = False
                        self.logger.info("Simulation paused")
                    
                    elif command == 'reset':
                        self.running = False
                        self.time = 0.0
                        self.tracker = Tracker()
                        self.logger.info("Simulation reset")
                    
                    elif command == 'set_disturbances':
                        params = data.get('params', {})
                        self.disturbances.turbulence = params.get('turbulence', 1.0)
                        self.disturbances.vibration = params.get('vibration', 1.0)
                        self.disturbances.camera_motion = params.get('camera_motion', 1.0)
                        self.disturbances.sensor_noise = params.get('sensor_noise', 1.0)
                        self.logger.info(f"Disturbances updated: {params}")
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON from client: {message}")
                    
        finally:
            await self.unregister(websocket)
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point."""
    server = SimulationWebSocketServer(host='0.0.0.0', port=8000)
    await server.start_server()


if __name__ == '__main__':
    asyncio.run(main())
