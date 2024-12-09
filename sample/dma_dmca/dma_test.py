# %%
# 必要なパッケージのインストール
# pip install fbm scipy matplotlib numpy

import matplotlib.pyplot as plt
import numpy as np
from fbm import FBM
from scipy.signal import savgol_filter

# DMAの次数
q = 0

# 時系列の長さ
n = 10000
# Hurst指数の指定
H = 0.8
# 時系列の分散
StDev = 1

# 時系列生成
f = FBM(n=n, hurst=H, length=1, method="daviesharte")
x = f.fgn() * StDev

# スケールを何点とるか
n_s = 30
# スケールの決定 (DMAで解析するスケールは奇数のみ)
s = np.unique(
    np.round(np.exp(np.linspace(np.log(q + 3), np.log(n / 4), n_s)) / 2).astype(int) * 2
    + 1
)
# スケール数の再計算 (重複があるので減る)
n_s = len(s)

# F2の計算
F2 = []
# 【STEP1】時系列の積分
y = np.cumsum(x)

# q次DMA
for scale in s:
    # 【STEP2】Detrending
    y_detrend = y - savgol_filter(y, window_length=scale, polyorder=q, deriv=0)
    # 【STEP3】2乗偏差の計算
    F2.append(np.mean(y_detrend**2))

# スケーリング
log10F_s = np.log10(F2) / 2
log10F_min = np.min(log10F_s)
log10F_max = np.max(log10F_s)

# グラフの描画
plt.figure(figsize=(12, 8))

# サンプル時系列
plt.subplot(2, 2, 1)
plt.plot(np.arange(n), x, "b-")
plt.xlabel("i")
plt.ylabel("x[i]")
plt.title("Sample time series")

# 積分した系列
plt.subplot(2, 2, 2)
plt.plot(np.arange(n), y, "b-")
plt.xlabel("i")
plt.ylabel("x[i]")
plt.title("Integrated series")

# 【STEP4】両対数プロットで傾きを計算
plt.subplot(2, 2, 3)
plt.plot(np.log10(s), log10F_s, "bo", label="log10 F(s)")
plt.xlabel("log10 s")
plt.ylabel("log10 F(s)")
plt.ylim(log10F_min, log10F_max)
plt.title("DMA")

# 傾きの推定
log10s_min = 1  # 区間の下限
log10s_max = 3  # 区間の上限
mask = (np.log10(s) >= log10s_min) & (np.log10(s) <= log10s_max)
slope, intercept = np.polyfit(np.log10(s)[mask], log10F_s[mask], 1)
plt.plot(
    np.log10(s),
    intercept + slope * np.log10(s),
    "r--",
    label=f"Estimated slope: {slope:.2f}",
)
plt.axvline(log10s_min, color="gray", linestyle="--")
plt.axvline(log10s_max, color="gray", linestyle="--")
plt.legend(loc="lower right")

# 局所傾斜のプロット
plt.subplot(2, 2, 4)
local_slope = np.diff(log10F_s) / np.diff(np.log10(s))
plt.plot(np.log10(s[:-1]), local_slope, "bo", label="local slope")
plt.axhline(H, color="gray", linestyle="--", label="Theory")
plt.xlabel("log10 s")
plt.ylabel("local slope")
plt.ylim(0, 2)
plt.title("local slope estimation")
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.signal import savgol_filter

# DMAの次数
q = 0

# 時系列の長さ
n = 10000
# Hurst指数の指定
H = 0.8
# 時系列の分散
StDev = 1


# フラクショナルガウスノイズの生成
def simFGN(n, H):
    # フラクショナルガウスノイズ生成用のシンプルな関数
    # ここでは代替として単純な正規分布乱数を使います
    return np.random.normal(0, 1, n)


# scaleは平均0，標準偏差1にする関数
x = (simFGN(n, H) - np.mean(simFGN(n, H))) / np.std(simFGN(n, H)) * StDev

# スケールを何点とるか
n_s = 30
# スケールの決定 (DMAで解析するスケールは奇数のみ)
s = np.unique(
    np.round(np.exp(np.linspace(np.log(q + 3), np.log(n / 4), n_s)) / 2) * 2 + 1
).astype(int)
# スケール数の再計算 (重複があるので減る)
n_s = len(s)

F2 = []
# 【STEP1】時系列の積分
y = np.cumsum(x)

# q次DMA
for i in range(n_s):
    # 【STEP2】Detrending
    y_detrend = y - savgol_filter(y, window_length=s[i], polyorder=q)
    # 【STEP3】2乗偏差の計算
    F2.append(np.mean(y_detrend**2))

# スケーリング
log10F_s = np.log10(F2) / 2
log10F_min = np.min(log10F_s)
log10F_max = np.max(log10F_s)

# プロット設定
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].plot(range(n), x, color="blue")
axs[0, 0].set_title("Sample time series")
axs[0, 0].set_xlabel("i")
axs[0, 0].set_ylabel("x[i]")

axs[0, 1].plot(range(n), y, color="blue")
axs[0, 1].set_title("Integrated series")
axs[0, 1].set_xlabel("i")
axs[0, 1].set_ylabel("x[i]")

# 【STEP4】両対数プロットで傾きを計算
axs[1, 0].plot(np.log10(s), log10F_s, "o", color="blue")
axs[1, 0].set_title("DMA")
axs[1, 0].set_xlabel("log10 s")
axs[1, 0].set_ylabel("log10 F(s)")
axs[1, 0].set_ylim([log10F_min, log10F_max])

# 傾きの推定
log10s_min = 1  # 区間の下限
log10s_max = 3  # 区間の上限
log10_s = np.log10(s)
mask = (log10_s >= log10s_min) & (log10_s <= log10s_max)
slope, intercept = np.polyfit(log10_s[mask], log10F_s[mask], 1)
axs[1, 0].plot(
    log10_s, slope * log10_s + intercept, color="red", linestyle="--", linewidth=2
)
axs[1, 0].axvline(log10s_min, color="gray", linestyle="--")
axs[1, 0].axvline(log10s_max, color="gray", linestyle="--")
axs[1, 0].legend([f"Estimated slope: {slope:.2f}"], loc="lower right")

# 傾きの差分プロット
local_slope = np.diff(log10F_s) / np.diff(log10_s)
axs[1, 1].plot(log10_s[1:], local_slope, "o", color="blue")
axs[1, 1].axhline(H, color="gray", linestyle="--")
axs[1, 1].set_title("local slope estimation")
axs[1, 1].set_xlabel("log10 s")
axs[1, 1].set_ylabel("local slope")
axs[1, 1].legend(["Theory", "Slope between adjacent points"], loc="upper left")

plt.tight_layout()
plt.show()

# %%
