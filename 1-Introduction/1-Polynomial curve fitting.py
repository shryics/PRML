import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

S = 50 # 刻みの回数
P = 10 # データ点の数
L = 3 # 項数

x = np.linspace(0, 1, P) # 0 から 1 まで P個の点で刻む
y = np.sin(2 * np.pi * x) # sin(2πx) 正解データ
t = np.random.rand(P) / 10
t += y # 人工データ(PRLLで言う所のt)

E_w_min = 99999999
w_opt = [0 for i in range(L)] # 最適時の重み
w = [0 for i in range(L)] # 重み

left_val = - S/2 # stepの左端値
step = 1 # step幅

Num = len(w)
def iterations():
    global Num, E_w_min, w_opt

    w[Num-1] = left_val
    for i in tqdm(range(S)):
        w[Num-1] += step
        if Num != 0:
            Num -= 1
            iterations()
        else:
            # optimize
            E_w = 0
            for j in range(P):
                sum = 0
                for k in range(L):
                    sum += w[k] * x[j]**k
                E_w += (sum - t[j])**2
            E_w = E_w / 2.0
            # 最小化問題の重み更新
            if E_w < E_w_min:
                E_w_min = E_w
                for l in range(L):
                    w_opt[l] = w[l]
    Num += 1
    return 0

iterations()

# 回帰式
for k in range(L):
    Predict_part = w_opt[k] * x**k
    if k == 0:
        Predict = Predict_part
    else:
        Predict += Predict_part

plt.plot(x, y) # sin(2πx)
plt.plot(x, t, "o") # sin(2πx)に乱数を付与した点
plt.plot(x, Predict) # 回帰式
plt.show()
