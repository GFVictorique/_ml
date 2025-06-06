import random
import math

# 生成隨機城市
def generate_cities(n, width=100, height=100):
    return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]

# 計算兩城市間距離
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# 計算整條路徑總距離
def total_distance(path, cities):
    return sum(distance(cities[path[i]], cities[path[(i + 1) % len(path)]]) for i in range(len(path)))

# 2-opt 交換
def two_opt_swap(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
    return new_route

# 爬山演算法解 TSP
def hill_climb_tsp(cities, max_iter=10000):
    n = len(cities)
    current_path = list(range(n))
    random.shuffle(current_path)
    current_distance = total_distance(current_path, cities)

    for i in range(max_iter):
        new_path = two_opt_swap(current_path)
        new_distance = total_distance(new_path, cities)
        if new_distance < current_distance:
            current_path = new_path
            current_distance = new_distance

    return current_path, current_distance

# 範例使用
cities = generate_cities(10)  # 10 個城市
best_path, best_dist = hill_climb_tsp(cities)

# 輸出結果
print("Best Path:", best_path)
print("Total Distance:", best_dist)
