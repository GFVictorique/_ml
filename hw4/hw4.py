import math
import numpy as np
from numpy.linalg import norm
import numpy as np

# 七段顯示器的真值表（輸入：7 位元，輸出：4 位元二進制）
seven_segment_truth_table = {
    0:  (1, 1, 1, 1, 1, 1, 0),
    1:  (0, 1, 1, 0, 0, 0, 0),
    2:  (1, 1, 0, 1, 1, 0, 1),
    3:  (1, 1, 1, 1, 0, 0, 1),
    4:  (0, 1, 1, 0, 0, 1, 1),
    5:  (1, 0, 1, 1, 0, 1, 1),
    6:  (1, 0, 1, 1, 1, 1, 1),
    7:  (1, 1, 1, 0, 0, 0, 0),
    8:  (1, 1, 1, 1, 1, 1, 1),
    9:  (1, 1, 1, 1, 0, 1, 1)
}

# 目標輸出 (數字的 4 位元二進位表示)
binary_outputs = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 0, 1, 1),
    4: (0, 1, 0, 0),
    5: (0, 1, 0, 1),
    6: (0, 1, 1, 0),
    7: (0, 1, 1, 1),
    8: (1, 0, 0, 0),
    9: (1, 0, 0, 1)
}

# 建立訓練資料
input_vectors = np.array([seven_segment_truth_table[i] for i in range(10)])  # 10 x 7
target_outputs = np.array([binary_outputs[i] for i in range(10)])  # 10 x 4

# 權重矩陣 (7x4)，初始值為隨機數
weights = np.random.rand(7, 4)

# 誤差函數：均方誤差 (MSE)
def loss_function(w):
    w = np.array(w).reshape(7, 4)  # 確保 w 是 7x4 矩陣
    predictions = input_vectors @ w  # 預測輸出
    return np.mean((predictions - target_outputs) ** 2)  # MSE

# 函數 f 對變數 k 的偏微分: df / dk
def df(f, p, k, h=0.01):
    p1 = p.copy()
    p1[k] = p[k]+h
    return (f(p1) - f(p)) / h

# 函數 f 在點 p 上的梯度
def grad(f, p, h=0.01):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, h)
    return gp

# 使用梯度下降法尋找函數最低點
def gradientDescendent(f, p0, h=0.01, max_loops=10000, dump_period=1000):
    p = p0.copy()
    for i in range(max_loops):
        fp = f(p)
        gp = grad(f, p) # 計算梯度 gp
        glen = norm(gp) # norm = 梯度的長度 (步伐大小)
        if i%dump_period == 0: 
            print('{:05d}:  f(p)={:.3f} glen={:.5f}\n'.format(i, fp, glen))
        p =  [pi - gi * h for pi, gi in zip(p, gp)] # 向 gh 方向走一小步
    print('{:05d}:  f(p)={:.3f} glen={:.5f}\n'.format(i, fp, glen))
    return p # 傳回最低點！

# 使用梯度下降法來調整權重
p = gradientDescendent(loss_function, weights.flatten().tolist())

# 訓練後的權重
trained_weights = np.array(p).reshape(7, 4)

# 預測函數
def predict(segment_input):
    prediction = np.round(segment_input @ trained_weights).astype(int)  # 轉換為 0/1
    return prediction

# 測試預測
for num, segment in seven_segment_truth_table.items():
    binary_prediction = predict(np.array(segment))
    binary_str = "".join(map(str, binary_prediction))
    print(f"Input: {segment} -> Predicted Binary: {binary_prediction}")