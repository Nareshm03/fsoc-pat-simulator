import cv2

from simulation.camera import VirtualCamera

camera = VirtualCamera()

frame = camera.render(640, 360)

cv2.imshow("Virtual Camera", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
