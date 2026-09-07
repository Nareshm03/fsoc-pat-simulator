import math


class Target:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        
        # Field of view (FOV) parameters
        # Total FOV in degrees
        self.fov_horizontal_deg = 40.0
        self.fov_vertical_deg = 30.0
        
        # Convert to radians
        self.fov_horizontal_rad = math.radians(self.fov_horizontal_deg)
        self.fov_vertical_rad = math.radians(self.fov_vertical_deg)
        
        # Pixels per radian conversion factors
        self.pixels_per_rad_x = self.width / self.fov_horizontal_rad
        self.pixels_per_rad_y = self.height / self.fov_vertical_rad

    def position(self, t: float):
        """
        Get target position in pixel coordinates (for ground truth visualization).
        This represents where the target would appear if gimbal was at (0, 0).
        """
        x = (
            self.width / 2
            + 250 * math.sin(t * 0.5)
        )

        y = (
            self.height / 2
            + 120 * math.cos(t * 0.35)
        )

        return x, y
    
    def angular_position(self, t: float):
        """
        Get target angular position in radians (azimuth, elevation).
        This is the absolute angular position of the target in space.
        """
        # Convert the sinusoidal motion to angular coordinates
        # Scale the pixel motion to angular motion
        azimuth_rad = math.radians(20.0 * math.sin(t * 0.5))  # ±20 degrees
        elevation_rad = math.radians(10.0 * math.cos(t * 0.35))  # ±10 degrees
        
        return azimuth_rad, elevation_rad
    
    def pixel_from_angles(self, target_az, target_el, gimbal_az, gimbal_el):
        """
        Calculate where the target appears in camera pixels given:
        - target_az, target_el: Target's absolute angular position (radians)
        - gimbal_az, gimbal_el: Gimbal's current angle (radians)
        
        Returns: (pixel_x, pixel_y) or None if out of FOV
        """
        # Relative angle = target angle - gimbal angle
        rel_az = target_az - gimbal_az
        rel_el = target_el - gimbal_el
        
        # Check if target is within camera FOV
        if abs(rel_az) > self.fov_horizontal_rad / 2:
            return None
        if abs(rel_el) > self.fov_vertical_rad / 2:
            return None
        
        # Convert relative angle to pixel coordinates
        # Center of image is (width/2, height/2)
        # Positive azimuth -> right, Positive elevation -> down (image coordinates)
        pixel_x = self.width / 2 + rel_az * self.pixels_per_rad_x
        pixel_y = self.height / 2 - rel_el * self.pixels_per_rad_y  # Negative because image Y is down
        
        return pixel_x, pixel_y