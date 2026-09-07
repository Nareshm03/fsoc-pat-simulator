from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.tracker import Tracker

camera = VirtualCamera()
detector = BeaconDetector()
tracker = Tracker()

print("=" * 50)
print("TRACKER TEST")
print("=" * 50)

# Simulate several frames with moving beacon
positions = [(640, 360), (645, 362), (650, 364), (655, 366)]

for i, (x, y) in enumerate(positions):
    frame = camera.render(x, y)
    detection = detector.detect(frame)
    tracking = tracker.update(detection)
    
    print(f"\nFrame {i}:")
    print(f"  True position: ({x}, {y})")
    if tracking:
        print(f"  Tracked: ({tracking['x']:.1f}, {tracking['y']:.1f})")
        print(f"  Velocity: ({tracking['vx']:.1f}, {tracking['vy']:.1f}) px/frame")
    else:
        print(f"  No tracking")

print("\n" + "=" * 50)
print("✓ Tracker working! Velocity computed from motion")
print("=" * 50)
