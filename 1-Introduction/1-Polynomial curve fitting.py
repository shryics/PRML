import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 10 # データ点の数
x = np.linspace(0, 1, N) # 0 から 1 まで N個の点で刻む
y = np.sin(2 * np.pi * x) # sin(2πx) 正解データ

t = np.random.rand(N) / 1
t += y # 人工データ(PRMLで言う所のt)

w_0, w_1, w_2, w_3 = 0.0, 0, 0, 0 # 重み


E_w_min = 999999
w_0_min, w_1_min, w_2_min = 999999, 999999, 999999


w_2 = -(1 / N)
for i_2 in tqdm(range(N)):
    w_2 += (2 / N)

    w_1 = -(1 / N)
    for i_1 in range(N):
        w_1 += (2 / N)

        w_0 = -(1 / N)
        for i_0 in range(N):
            w_0 += (2 / N)


            # 最小化問題の式
            E_w = 0
            for j in range(N):
                E_w += ( ( w_0 + w_1 * x[j] + w_2 * x[j]**2 ) - t[j]) ** 2
            E_w = E_w / 2

            # 最小化問題の重み更新
            if E_w < E_w_min:
                E_w_min = E_w
                w_0_min = w_0
                w_1_min = w_1
                w_2_min = w_2
                print(w_0,w_1,w_2)


print()
print(w_0_min, w_1_min, w_2_min)

# 回帰式
Predict = w_0_min + w_1_min * x + w_2_min * x**2

plt.plot(x, y) # sin(2πx)
plt.plot(x, t, "o") # sin(2πx)に乱数を付与した点
plt.plot(x, Predict) # 回帰式
plt.show()
