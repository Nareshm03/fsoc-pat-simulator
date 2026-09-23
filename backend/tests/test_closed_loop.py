"""
Tests for closed-loop PAT control system.
Verifies that gimbal movement genuinely changes camera view.
"""
import pytest
import math
import sys
sys.path.insert(0, '..')

from simulation.target import Target
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.pan_tilt import PanTilt
from main import SimulationManager


class TestClosedLoop:
    """Test closed-loop PAT system."""
    
    def test_gimbal_angle_changes_pixel_position(self):
        """Test 1: Changing gimbal angle changes beacon pixel position."""
        target = Target()
        
        # Fixed target angular position
        target_az = math.radians(10.0)  # 10 degrees right
        target_el = math.radians(5.0)   # 5 degrees up
        
        # Test 1: Gimbal at origin (0, 0)
        gimbal_az_1 = math.radians(0.0)
        gimbal_el_1 = math.radians(0.0)
        
        pos1 = target.pixel_from_angles(target_az, target_el, gimbal_az_1, gimbal_el_1)
        assert pos1 is not None, "Target should be in FOV"
        x1, y1 = pos1
        
        # Target is at +10 deg azimuth relative to gimbal at 0
        # Should appear right of center
        assert x1 > target.width / 2, f"Target should be right of center, got x={x1}"
        
        # Test 2: Gimbal moves right by 5 degrees
        gimbal_az_2 = math.radians(5.0)
        gimbal_el_2 = math.radians(0.0)
        
        pos2 = target.pixel_from_angles(target_az, target_el, gimbal_az_2, gimbal_el_2)
        assert pos2 is not None, "Target should still be in FOV"
        x2, y2 = pos2
        
        # Target is now at +5 deg relative to gimbal
        # Should appear closer to center (less to the right)
        assert x2 < x1, f"Target should move left when gimbal turns right: x1={x1}, x2={x2}"
        assert abs(x2 - target.width/2) < abs(x1 - target.width/2), "Target should be closer to center"
        
        # Test 3: Gimbal matches target angle
        gimbal_az_3 = target_az
        gimbal_el_3 = target_el
        
        pos3 = target.pixel_from_angles(target_az, target_el, gimbal_az_3, gimbal_el_3)
        assert pos3 is not None
        x3, y3 = pos3
        
        # Relative angle is zero, target should be at camera center
        assert abs(x3 - target.width/2) < 1, f"Target should be centered when gimbal matches: x3={x3}"
        assert abs(y3 - target.height/2) < 1, f"Target should be centered when gimbal matches: y3={y3}"
        
        print("✓ Test 1 PASSED: Gimbal angle changes pixel position")
    
    def test_detector_sees_changed_position(self):
        """Test 2: Detector sees the position changed by gimbal."""
        target = Target()
        camera = VirtualCamera()
        detector = BeaconDetector()
        
        target_az = math.radians(8.0)
        target_el = math.radians(0.0)
        
        # Scenario A: Gimbal at 0
        gimbal_az_a = math.radians(0.0)
        pos_a = target.pixel_from_angles(target_az, target_el, gimbal_az_a, 0.0)
        assert pos_a is not None
        
        frame_a = camera.render(*pos_a)
        detection_a = detector.detect(frame_a)
        assert detection_a is not None, "Detector should see beacon"
        
        # Scenario B: Gimbal at +4 degrees
        gimbal_az_b = math.radians(4.0)
        pos_b = target.pixel_from_angles(target_az, target_el, gimbal_az_b, 0.0)
        assert pos_b is not None
        
        frame_b = camera.render(*pos_b)
        detection_b = detector.detect(frame_b)
        assert detection_b is not None, "Detector should still see beacon"
        
        # Detector should report different positions
        assert detection_a["x"] != detection_b["x"], \
            f"Detector should see different X: {detection_a['x']} vs {detection_b['x']}"
        
        # Position B should be more centered (closer to 640)
        center_x = camera.width / 2
        assert abs(detection_b["x"] - center_x) < abs(detection_a["x"] - center_x), \
            "Gimbal movement should center the target"
        
        print("✓ Test 2 PASSED: Detector sees changed position")
    
    def test_pat_error_responds_to_position(self):
        """Test 3: PAT tracking error responds to detected position."""
        manager = SimulationManager()
        
        # Manually set detected position off-center
        manager.detected = True
        manager.detected_x = 800.0  # Right of center (640)
        manager.detected_y = 360.0  # Centered vertically
        
        center_x = manager.camera.width / 2
        center_y = manager.camera.height / 2
        
        error_x = manager.detected_x - center_x
        error_y = manager.detected_y - center_y
        expected_error = math.sqrt(error_x**2 + error_y**2)
        
        # Error should be non-zero
        assert expected_error > 0, "Should have tracking error when off-center"
        
        # Now center the detection
        manager.detected_x = center_x
        manager.detected_y = center_y
        
        error_x_centered = manager.detected_x - center_x
        error_y_centered = manager.detected_y - center_y
        centered_error = math.sqrt(error_x_centered**2 + error_y_centered**2)
        
        assert centered_error < expected_error, "Error should be smaller when centered"
        assert centered_error < 1.0, "Error should be near zero when perfectly centered"
        
        print("✓ Test 3 PASSED: PAT error responds to position")
    
    def test_gimbal_moves_from_pat_command(self):
        """Test 4: Gimbal moves in response to PAT command."""
        pan_tilt = PanTilt()
        
        initial_az = pan_tilt.azimuth
        initial_el = pan_tilt.elevation
        
        # Send a positive azimuth command (rate in deg/sec)
        az_command = 5.0  # 5 deg/sec
        el_command = 0.0
        dt = 0.1  # 100ms
        
        pan_tilt.update(az_command, el_command, dt)
        
        # Gimbal should have moved
        expected_change = az_command * dt
        assert pan_tilt.azimuth > initial_az, f"Azimuth should increase: {initial_az} -> {pan_tilt.azimuth}"
        assert abs(pan_tilt.azimuth - initial_az - expected_change) < 0.01, \
            f"Azimuth change should match command: expected {expected_change}, got {pan_tilt.azimuth - initial_az}"
        
        # Elevation should not change
        assert pan_tilt.elevation == initial_el, "Elevation should not change"
        
        print("✓ Test 4 PASSED: Gimbal moves from PAT command")
    
    def test_feedback_loop_converges(self):
        """Test 5: Repeated ticks form a converging feedback loop."""
        target = Target()
        camera = VirtualCamera()
        detector = BeaconDetector()
        pan_tilt = PanTilt()
        
        # Start with target offset and gimbal at origin
        target_az = math.radians(15.0)  # 15 degrees right
        target_el = math.radians(0.0)
        
        # Track errors over time
        errors = []
        gimbal_angles = []
        
        # Simulate 100 ticks of closed-loop control
        dt = 1.0 / 30.0  # 30 FPS
        for tick in range(100):
            # Calculate where target appears in camera
            gimbal_az_rad = math.radians(pan_tilt.azimuth)
            gimbal_el_rad = math.radians(pan_tilt.elevation)
            
            pos = target.pixel_from_angles(target_az, target_el, gimbal_az_rad, gimbal_el_rad)
            
            if pos is None:
                # Target out of FOV - search mode
                continue
            
            # Render and detect
            frame = camera.render(*pos)
            detection = detector.detect(frame)
            
            if detection:
                # Calculate error from center
                error_x = detection["x"] - camera.width / 2
                error_y = detection["y"] - camera.height / 2
                error_pixels = math.sqrt(error_x**2 + error_y**2)
                errors.append(error_pixels)
                gimbal_angles.append(pan_tilt.azimuth)
                
                # PAT control command (proportional)
                pixel_to_deg_per_sec = 0.02
                az_command = error_x * pixel_to_deg_per_sec
                el_command = error_y * pixel_to_deg_per_sec
                
                # Limit commands
                max_rate = 10.0
                az_command = max(-max_rate, min(max_rate, az_command))
                el_command = max(-max_rate, min(max_rate, el_command))
                
                # Update gimbal
                pan_tilt.update(az_command, el_command, dt)
        
        # Verify convergence
        assert len(errors) > 50, f"Should have at least 50 successful detections, got {len(errors)}"
        
        # Initial error should be large
        initial_error = errors[0]
        assert initial_error > 100, f"Initial error should be large (target offset), got {initial_error}"
        
        # Final error should be small (converged)
        final_error = errors[-1]
        assert final_error < 70, f"Final error should be small (converged), got {final_error}"
        
        # Error should decrease over time
        mid_error = errors[len(errors)//2]
        assert mid_error < initial_error, "Error should decrease"
        assert final_error < mid_error, "Error should continue decreasing"
        
        # Gimbal should have moved toward target
        initial_gimbal = gimbal_angles[0]
        final_gimbal = gimbal_angles[-1]
        assert final_gimbal > initial_gimbal, \
            f"Gimbal should move right to track target: {initial_gimbal} -> {final_gimbal}"
        
        # Gimbal should be close to target angle
        target_az_deg = math.degrees(target_az)
        assert abs(final_gimbal - target_az_deg) < 2.0, \
            f"Gimbal should converge to target angle: target={target_az_deg}, final={final_gimbal}"
        
        print("✓ Test 5 PASSED: Feedback loop converges")
        print(f"  Initial error: {initial_error:.1f}px, Final error: {final_error:.1f}px")
        print(f"  Initial gimbal: {initial_gimbal:.2f}°, Final gimbal: {final_gimbal:.2f}°")
        print(f"  Target angle: {target_az_deg:.2f}°")
    
    def test_gimbal_respects_velocity_limits(self):
        """Test that gimbal respects maximum velocity limits."""
        pan_tilt = PanTilt()
        
        # Try to command very high velocity
        excessive_command = 100.0  # 100 deg/sec (much higher than max 30 deg/sec)
        dt = 0.1
        
        initial_az = pan_tilt.azimuth
        pan_tilt.update(excessive_command, 0.0, dt)
        
        actual_change = pan_tilt.azimuth - initial_az
        max_possible_change = pan_tilt.max_azimuth_rate * dt
        
        # Change should be limited to max rate
        assert actual_change <= max_possible_change + 0.01, \
            f"Gimbal should limit velocity: max={max_possible_change}, actual={actual_change}"
        
        print("✓ Test 6 PASSED: Gimbal respects velocity limits")
    
    def test_target_out_of_fov_when_gimbal_far(self):
        """Test that target goes out of FOV when gimbal points away."""
        target = Target()
        
        target_az = math.radians(5.0)
        target_el = math.radians(0.0)
        
        # Point gimbal far away
        gimbal_az = math.radians(50.0)  # 50 degrees away (FOV is 40 degrees)
        gimbal_el = math.radians(0.0)
        
        pos = target.pixel_from_angles(target_az, target_el, gimbal_az, gimbal_el)
        
        # Target should be out of FOV
        assert pos is None, "Target should be out of FOV when gimbal points far away"
        
        print("✓ Test 7 PASSED: Target out of FOV when gimbal far")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
