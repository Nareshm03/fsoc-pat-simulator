import math

from simulation.target import Target

target = Target()

print("=== Legacy pixel motion (position) ===")
for i in range(10):
    t = i * 0.5

    x, y = target.position(t)

    print(
        f"t={t:.2f}s  "
        f"x={x:.2f}  "
        f"y={y:.2f}"
    )

print("=== Live angular motion + camera projection ===")
for i in range(5):
    t = i * 0.5

    target_az, target_el = target.angular_position(t)
    pos = target.pixel_from_angles(
        target_az, target_el,
        math.radians(0.0), math.radians(0.0),
    )

    assert pos is not None, "Target should be in FOV at gimbal origin"
    x, y = pos
    assert 0 <= x < target.width and 0 <= y < target.height, (x, y)

    print(
        f"t={t:.2f}s  "
        f"az={math.degrees(target_az):+.2f}deg  "
        f"el={math.degrees(target_el):+.2f}deg  "
        f"px=({x:.1f}, {y:.1f})"
    )

print("Target projection checks passed")
