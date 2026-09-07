from simulation.target import Target

target = Target()

for i in range(10):
    t = i * 0.5

    x, y = target.position(t)

    print(
        f"t={t:.2f}s  "
        f"x={x:.2f}  "
        f"y={y:.2f}"
    )
