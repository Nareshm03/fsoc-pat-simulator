import cv2

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector


camera = VirtualCamera()
detector = BeaconDetector()

frame = camera.render(640, 360)

result = detector.detect(frame)

print(result)

cv2.imshow("Camera", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
