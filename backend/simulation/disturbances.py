import math
import random


class DisturbanceModel:

    def __init__(
        self,
        turbulence=1.0,
        vibration=1.0,
        camera_motion=1.0,
        sensor_noise=1.0
    ):
        self.turbulence = turbulence
        self.vibration = vibration
        self.camera_motion = camera_motion
        self.sensor_noise = sensor_noise

    def turbulence_offset(self, t):
        x = self.turbulence * (
            5 * math.sin(t * 1.7)
            + 2 * math.sin(t * 4.3)
        )

        y = self.turbulence * (
            4 * math.cos(t * 1.4)
            + 2 * math.sin(t * 3.8)
        )

        return x, y

    def vibration_offset(self, t):
        x = self.vibration * (
            3 * math.sin(2 * math.pi * 6 * t)
        )

        y = self.vibration * (
            3 * math.cos(2 * math.pi * 7 * t)
        )

        return x, y

    def camera_offset(self, t):
        x = self.camera_motion * (
            6 * math.sin(t * 0.8)
        )

        y = self.camera_motion * (
            4 * math.cos(t * 0.6)
        )

        return x, y

    def sensor_offset(self):
        x = random.uniform(
            -self.sensor_noise,
            self.sensor_noise
        )

        y = random.uniform(
            -self.sensor_noise,
            self.sensor_noise
        )

        return x, y
