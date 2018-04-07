import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 100 # データ点の数
M = 4 # 項数
x = np.linspace(0, 1, N) # 0 から 1 まで N個の点で刻む
y = np.sin(2 * np.pi * x) # sin(2πx) 正解データ

t = np.random.rand(N) / 10
t += y # 人工データ(PRMLで言う所のt)

E_w_min = 999999
w_opt = [999999 for i in range(M)]
w = [0 for i in range(M)]


w[3] = - N/2
for i_3 in tqdm(range(N)):
    w[3] += 100/N

    w[2] = - N/2
    for i_2 in range(N):
        w[2] += 100/N

        w[1] = - N/2
        for i_1 in range(N):
            w[1] += 100/N

            w[0] = - N/2
            for i_0 in range(N):
                w[0] += 100/N


                # 最小化問題の式
                E_w = 0
                for j in range(N):
                    sum = 0
                    for k in range(M):
                        sum += w[k] * x[j]**k
                    E_w += (sum - t[j])**2
                E_w = E_w / 2
                # 最小化問題の重み更新
                if E_w < E_w_min:
                    E_w_min = E_w
                    for l in range(M):
                        w_opt[l] = w[l]
                    #print(w_opt)

#print(w_opt)

# 回帰式
for k in range(M):
    Predict_part = w_opt[k] * x**k
    if k == 0:
        Predict = Predict_part
    Predict += Predict_part
print(Predict)
plt.plot(x, y) # sin(2πx)
plt.plot(x, t, "o") # sin(2πx)に乱数を付与した点
plt.plot(x, Predict) # 回帰式
plt.show()
