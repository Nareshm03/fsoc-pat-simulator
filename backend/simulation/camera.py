import cv2
import numpy as np


class VirtualCamera:

    def __init__(
        self,
        width=1280,
        height=720
    ):
        self.width = width
        self.height = height

    def render(self, beacon_x, beacon_y, beacon_size=8):
        """
        Render camera frame with beacon.
        
        Args:
            beacon_x: Beacon x position
            beacon_y: Beacon y position
            beacon_size: Beacon radius in pixels (default: 8)
        
        Returns:
            Rendered frame
        """
        frame = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        if (
            0 <= beacon_x < self.width
            and
            0 <= beacon_y < self.height
        ):
            # Inner bright circle (beacon core)
            cv2.circle(
                frame,
                (int(beacon_x), int(beacon_y)),
                int(beacon_size),
                (255, 255, 255),
                -1
            )

            # Outer dim circle (beacon glow/halo)
            cv2.circle(
                frame,
                (int(beacon_x), int(beacon_y)),
                int(beacon_size * 2.5),
                (100, 100, 100),
                2
            )

        return frame
