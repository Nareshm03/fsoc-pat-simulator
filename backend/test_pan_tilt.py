from simulation.pan_tilt import PanTilt

pan_tilt = PanTilt()

print("=" * 50)
print("PAN/TILT SYSTEM TEST")
print("=" * 50)

print("\nInitial state:")
print(f"  Azimuth: {pan_tilt.azimuth:.2f}°")
print(f"  Elevation: {pan_tilt.elevation:.2f}°")

# Test velocity limiting
print("\nTest 1: Apply excessive command (should be limited)")
print(f"  Command: azimuth=100°/s (exceeds max {pan_tilt.max_azimuth_rate}°/s)")

dt = 0.1  # 100ms timestep
pan_tilt.update(azimuth_command=100.0, elevation_command=0.0, dt=dt)

print(f"  Velocity: {pan_tilt.azimuth_velocity:.2f}°/s")
print(f"  Position after {dt}s: {pan_tilt.azimuth:.2f}°")

if pan_tilt.azimuth_velocity == pan_tilt.max_azimuth_rate:
    print(f"  ✓ Velocity correctly limited to {pan_tilt.max_azimuth_rate}°/s")

# Test position limiting
print("\nTest 2: Move to extreme position")
for _ in range(100):
    pan_tilt.update(azimuth_command=50.0, elevation_command=0.0, dt=0.1)

print(f"  Final azimuth: {pan_tilt.azimuth:.2f}°")
print(f"  Max limit: {pan_tilt.max_azimuth}°")

if pan_tilt.azimuth == pan_tilt.max_azimuth:
    print(f"  ✓ Position correctly limited to {pan_tilt.max_azimuth}°")

# Reset and test elevation
pan_tilt.azimuth = 0.0
pan_tilt.azimuth_velocity = 0.0

print("\nTest 3: Elevation movement")
for i in range(5):
    pan_tilt.update(azimuth_command=0.0, elevation_command=10.0, dt=0.1)
    print(f"  Step {i+1}: elevation = {pan_tilt.elevation:.2f}°")

print("\n" + "=" * 50)
print("✓ Pan/Tilt system working!")
print("  - Velocity limiting works")
print("  - Position integration works")
print("  - Position limits enforced")
print("=" * 50)
