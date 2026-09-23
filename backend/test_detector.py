import os
import sys

import cv2

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector


camera = VirtualCamera()
detector = BeaconDetector()

frame = camera.render(640, 360)

result = detector.detect(frame)

print(result)

assert result is not None, "Detector should find the centered beacon"
assert abs(result["x"] - 640) < 5, result
assert abs(result["y"] - 360) < 5, result

# Headless-safe: only open a window with --show (or when a display exists).
if "--show" in sys.argv or os.environ.get("DISPLAY"):
    cv2.imshow("Camera", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Headless mode: detection only (pass --show to display window)")
