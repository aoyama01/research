#%%
# ↓↓↓ これのRスクリプトをChatGPTでPythonに変換したやつ
# https://chaos-kiyono.hatenablog.com/entry/2022/06/28/150608

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 時系列の長さ
N = 1000

# サンプルの時系列（白色ノイズ）
x = np.random.normal(size=N)
x = x - np.mean(x)  # 平均0に

# ランダムウォーク解析
# 積分
y = np.cumsum(x)

# 解析するスケールs
s = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(N/4), num=20)))).astype(int)
n_s = len(s)

# ゆらぎ関数F(s)を計算
F_s = []
for i in range(n_s):
    D_y = y[:-s[i]] - y[s[i]:]
    F_s.append(np.sqrt(np.mean(D_y ** 2)))

# log-logをとって直線をあてはめ
log_s = np.log10(s)
log_F_s = np.log10(F_s)
model = LinearRegression().fit(log_s.reshape(-1, 1), log_F_s)
slope = model.coef_[0]

# log-logプロット
plt.plot(log_s, log_F_s, 'o', color='blue', label='log F(s)')
plt.plot(log_s, model.predict(log_s.reshape(-1, 1)), 'r--', label=f'Fit (slope={slope:.3f})')
plt.xlabel("log10 s")
plt.ylabel("log10 F(s)")
plt.title(f"slope = {slope:.3f}")
plt.legend()
plt.show()

# %%
# プロットしてみる

#サンプル時系列
plt.figure()
plt.plot(np.arange(1000), x)

# 積分時系列
plt.figure()
plt.plot(np.arange(1000), y)

# ゆらぎ関数
plt.figure()
plt.plot(s, F_s)

# %%
