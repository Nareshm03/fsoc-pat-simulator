"""
Integration tests for FastAPI backend with existing simulation modules.
"""
import pytest
from fastapi.testclient import TestClient

from main import app, SimulationManager


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test GET / returns service information."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "FSOC PAT Simulation"
        assert data["status"] == "online"


class TestHealthEndpoint:
    """Test health endpoint."""
    
    def test_health_endpoint(self):
        """Test GET /health returns health status."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "simulation_running" in data
        assert "connected_clients" in data


class TestSimulationManager:
    """Test SimulationManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a SimulationManager instance."""
        return SimulationManager()
    
    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.running == False
        assert manager.paused == False
        assert manager.simulation_time == 0.0
        assert manager.frame_number == 0
        assert manager.target_fps == 30
        assert len(manager.clients) == 0
    
    def test_reset(self, manager):
        """Test reset functionality."""
        # Advance state
        manager.simulation_time = 10.0
        manager.frame_number = 300
        manager.detected = True
        manager.tracking_error_px = 5.0
        
        # Reset
        manager.reset()
        
        # Verify reset
        assert manager.simulation_time == 0.0
        assert manager.frame_number == 0
        assert manager.detected == False
        assert manager.tracking_error_px == 0.0
    
    def test_set_disturbance(self, manager):
        """Test setting disturbance parameters."""
        manager.set_disturbance("turbulence", 2.5)
        assert manager.disturbances.turbulence == 2.5
        
        manager.set_disturbance("vibration", 1.5)
        assert manager.disturbances.vibration == 1.5
        
        manager.set_disturbance("camera_motion", 0.5)
        assert manager.disturbances.camera_motion == 0.5
        
        manager.set_disturbance("sensor_noise", 3.0)
        assert manager.disturbances.sensor_noise == 3.0
    
    def test_set_invalid_disturbance(self, manager):
        """Test invalid disturbance name raises error."""
        with pytest.raises(ValueError):
            manager.set_disturbance("invalid", 1.0)
    
    def test_calculate_fsoc_status(self, manager):
        """Test FSOC status calculation."""
        # Not detected
        manager.detected = False
        assert manager._calculate_fsoc_status() == "LOST"
        
        # Small error - ACTIVE
        manager.detected = True
        manager.tracking_error_px = 2.0
        assert manager._calculate_fsoc_status() == "ACTIVE"
        
        # Medium error - DEGRADED
        manager.tracking_error_px = 5.0
        assert manager._calculate_fsoc_status() == "DEGRADED"
        
        # Large error - LOST
        manager.tracking_error_px = 15.0
        assert manager._calculate_fsoc_status() == "LOST"


class TestWebSocket:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishes."""
        from main import SimulationManager
        import main
        
        # Initialize manager for testing
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            # Connection successful
            assert websocket is not None
    
    @pytest.mark.asyncio
    async def test_start_command(self):
        """Test start command."""
        from main import SimulationManager
        import main
        
        # Initialize manager for testing
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "start"})
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "start"
    
    @pytest.mark.asyncio
    async def test_pause_command(self):
        """Test pause command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        # Stop any running simulation first
        await main.simulation_manager.stop()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "pause"})
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "pause"
    
    @pytest.mark.asyncio
    async def test_resume_command(self):
        """Test resume command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        # Stop any running simulation first
        await main.simulation_manager.stop()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "resume"})
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "resume"
    
    @pytest.mark.asyncio
    async def test_reset_command(self):
        """Test reset command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        # Stop any running simulation first
        await main.simulation_manager.stop()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "reset"})
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "reset"
    
    @pytest.mark.asyncio
    async def test_stop_command(self):
        """Test stop command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "stop"})
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "stop"
    
    @pytest.mark.asyncio
    async def test_invalid_command(self):
        """Test invalid command returns error."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({"command": "invalid_command"})
            response = websocket.receive_json()
            
            assert response["type"] == "error"
            assert "Unknown command" in response["message"]
    
    @pytest.mark.asyncio
    async def test_disturbance_command(self):
        """Test set disturbance command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({
                "command": "set_disturbance",
                "name": "turbulence",
                "value": 2.5
            })
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "set_disturbance"
    
    @pytest.mark.asyncio
    async def test_invalid_disturbance_name(self):
        """Test invalid disturbance name returns error."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({
                "command": "set_disturbance",
                "name": "invalid",
                "value": 1.0
            })
            response = websocket.receive_json()
            
            assert response["type"] == "error"
            assert "Invalid disturbance name" in response["message"]
    
    @pytest.mark.asyncio
    async def test_invalid_disturbance_value(self):
        """Test invalid disturbance value returns error."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            # Test NaN
            websocket.send_json({
                "command": "set_disturbance",
                "name": "turbulence",
                "value": "NaN"
            })
            response = websocket.receive_json()
            assert response["type"] == "error"
            
            # Test out of range
            websocket.send_json({
                "command": "set_disturbance",
                "name": "turbulence",
                "value": 100.0
            })
            response = websocket.receive_json()
            assert response["type"] == "error"
    
    @pytest.mark.asyncio
    async def test_target_speed_command(self):
        """Test set target speed command."""
        from main import SimulationManager
        import main
        
        if main.simulation_manager is None:
            main.simulation_manager = SimulationManager()
        
        client = TestClient(app)
        
        with client.websocket_connect("/ws/simulation") as websocket:
            websocket.send_json({
                "command": "set_target_speed",
                "value": 1.5
            })
            response = websocket.receive_json()
            
            assert response["type"] == "ack"
            assert response["command"] == "set_target_speed"
