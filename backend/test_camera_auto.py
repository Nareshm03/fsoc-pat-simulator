import cv2

from simulation.camera import VirtualCamera

camera = VirtualCamera()

frame = camera.render(640, 360)

print(f"Frame shape: {frame.shape}")
print(f"Frame type: {frame.dtype}")
print(f"Beacon should be at (640, 360)")

# Check if the beacon is drawn (white pixel at center)
if frame[360, 640].sum() > 0:
    print("✓ Beacon detected in frame!")
else:
    print("✗ No beacon found")

# Save the image instead of displaying
cv2.imwrite("test_output.png", frame)
print("✓ Frame saved to test_output.png")
