#%%
# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.linear_model import LinearRegression

# ハースト指数 (0 < H < 1)
H = 0.8

# 時系列の長さ
N = 10000

# 解析するスケール s
s = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(N/10), 20)))).astype(int)
n_s = len(s)

# Fractional Gaussian Noise (FGN) の生成
# fractional_brownian_motion は、Fractional Brownian Motionのライブラリです
# !pip install fractional_brownian_motion が必要
# fractional_brownian_motionじゃなくてfbmちゃう？
from fbm import FBM

# Fractional Brownian Motion のサンプルパスを生成
fbm = FBM(n=N, hurst=H)
x = fbm.fbm()  # サンプルデータを取得
x = x - np.mean(x)  # 平均0に

# ランダムウォーク解析
# 累積和の計算
y = np.cumsum(x)

# F(s) の計算
F_s = np.zeros(n_s)
for i in range(n_s):
    D_y = y[:N - s[i]] - y[s[i]:N]
    F_s[i] = np.sqrt(np.mean(D_y**2))

# 周波数とスペクトル密度の計算
freq = np.arange(N+1) / N
S_f = np.abs(fft(x))**2 / N

# プロットの準備
plt.figure(figsize=(18, 5))

# ログスケールのスケール s と F(s)
logs = np.log10(s)
logFs = np.log10(F_s)

# ログ・ログプロット: ランダムウォーク解析
plt.subplot(1, 3, 1)
model = LinearRegression()
model.fit(logs.reshape(-1, 1), logFs)
plt.plot(logs, logFs, 'o', color="blue", label="Data")
plt.plot(logs, model.predict(logs.reshape(-1, 1)), 'r--', label="Fit")
plt.xlabel("log10 s")
plt.ylabel("log10 F(s)")
plt.title("Random Walk Analysis")
plt.legend()

# ログ・ログプロット: スペクトル解析
plt.subplot(1, 3, 2)
log_freq = np.log10(freq[(freq > 0) & (freq < 0.5)])
log_Sf = np.log10(S_f[(freq > 0) & (freq < 0.5)])
model.fit(log_freq.reshape(-1, 1), log_Sf)
plt.plot(log_freq, log_Sf, color="green")
plt.plot(log_freq, model.predict(log_freq.reshape(-1, 1)), 'r--')
plt.xlabel("log10 f")
plt.ylabel("log10 S(f)")
plt.title("Spectral Analysis")

# 比較プロット
F2_Sf = np.zeros(n_s)
for i in range(n_s):
    F2_Sf[i] = sum(np.sin(np.pi * k / N * s[i])**2 / np.sin(np.pi * k / N)**2 * S_f[k]
                   for k in range(1, N))

plt.subplot(1, 3, 3)
plt.plot(logs, logFs, 'o', color="blue", label="F(s)")
plt.plot(logs, np.log10(F2_Sf / N) / 2, 's', color="red", label="F2(s)")
plt.xlabel("log10 s")
plt.ylabel("log10 F(s)")
plt.title("Comparison")
plt.legend()

plt.tight_layout()
plt.show()

# %%
