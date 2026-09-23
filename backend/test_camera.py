import os
import sys

import cv2

from simulation.camera import VirtualCamera

camera = VirtualCamera()

frame = camera.render(640, 360)

# Headless-safe: only open a window with --show (or when a display exists).
if "--show" in sys.argv or os.environ.get("DISPLAY"):
    cv2.imshow("Virtual Camera", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Headless mode: rendering only (pass --show to display window)")
    assert frame is not None and frame.shape == (720, 1280, 3)
    print(f"Rendered frame shape: {frame.shape}")
