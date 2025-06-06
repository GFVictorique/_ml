import random

def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 初始點
x, y, z = random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)
step_size = 0.01
tolerance = 1e-6
max_iterations = 10000

for _ in range(max_iterations):
    current = f(x, y, z)

    # 嘗試微調
    candidates = []
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            for dz in [-step_size, 0, step_size]:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                candidates.append((f(nx, ny, nz), nx, ny, nz))
    
    best = min(candidates, key=lambda t: t[0])
    
    if best[0] < current:
        x, y, z = best[1], best[2], best[3]
    else:
        break  # 沒有更小的方向了

print(f"Minimum at x={x:.5f}, y={y:.5f}, z={z:.5f}, f={f(x,y,z):.5f}")
