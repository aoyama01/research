#%%
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
from scipy.signal import savgol_filter

# パラメータの設定
n = 5000  # 時系列の長さ
H_common = 0.8  # 共通成分のHurst指数
H1 = 0.5  # 時系列1のHurst指数
H2 = 0.7  # 時系列2のHurst指数
SD_common = 0.5  # 共通成分の標準偏差

# Fractional Gaussian Noiseの生成と標準化
def generate_fgn(n, H):
    fbm = FBM(n=n, hurst=H)
    return (fbm.fgn() - np.mean(fbm.fgn())) / np.std(fbm.fgn())

# 時系列の生成
eps_1 = generate_fgn(n, H1)
eps_2 = generate_fgn(n, H2)
eps_common = generate_fgn(n, H_common) * SD_common

x1 = eps_1 + eps_common
x2 = eps_2 + eps_common

# サンプル時系列のプロット
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, color='green', label='x1')
plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("i")
plt.title("Sample time series (x1)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x2, color='blue', label='x2')
plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("i")
plt.title("Sample time series (x2)")
plt.legend()

# DMA解析で使用するスケール
n_s = 20
s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

# DMAの解析
F1, F2, F12_sq = [], [], []
y1, y2 = np.cumsum(x1), np.cumsum(x2)  # 累積和（積分）

# 0次のDMCAを適用
for scale in s:
    y1_detrend = y1 - savgol_filter(y1, window_length=scale, polyorder=0)
    y2_detrend = y2 - savgol_filter(y2, window_length=scale, polyorder=0)
    F1.append(np.sqrt(np.mean(y1_detrend ** 2)))
    F2.append(np.sqrt(np.mean(y2_detrend ** 2)))
    F12_sq.append(np.mean(y1_detrend * y2_detrend))

rho = np.array(F12_sq) / (np.array(F1) * np.array(F2))

# クロス相関のプロット
plt.subplot(2, 2, 3)
plt.plot(np.log10(s), rho, color='red', marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='--')
plt.axhline(y=-1, color='gray', linestyle='--')
plt.ylim(-1, 1)
plt.xlabel("log10(s)")
plt.ylabel("Cross-correlation (rho)")
plt.title("Cross-correlation")

# スケーリングプロット
log10F1 = np.log10(F1)
log10F2 = np.log10(F2)
log10F12 = np.log10(np.abs(F12_sq)) / 2
y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
y_max = max(log10F1.max(), log10F2.max(), log10F12.max())

plt.subplot(2, 2, 4)
plt.plot(np.log10(s), log10F1, color='green', marker='^', linestyle='-', label='log10(F1)')
plt.plot(np.log10(s), log10F2, color='blue', marker='s', linestyle='-', label='log10(F2)')
plt.plot(np.log10(s), log10F12, color='red', marker='o', linestyle='-', label='log10(F12)')
plt.ylim(y_min, y_max)
plt.xlabel("log10(s)")
plt.ylabel("log10(F(s))")
plt.title("Scaling")
plt.legend()

plt.tight_layout()
plt.show()

# %%
