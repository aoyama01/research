# %%
import matplotlib.pyplot as plt
import numpy as np

# 時系列の長さ（時系列は奇数長になります）
N = 2**12
# 奇数に変換
N = round(N / 2) * 2 + 1
# 半分
M = (N - 1) // 2


# モデルの自己共分散関数を与える
# 例：AR(2)過程の自己共分散関数
def AR2_model(n, a1, a2, sig2):
    Cov = [0] * (n + 1)
    Cov[0] = sig2 * (1 - a2) / (1 - a1**2 - a2 - a1**2 * a2 - a2**2 + a2**3)
    Cov[1] = Cov[0] * a1 / (1 - a2)
    for k in range(2, n + 1):
        Cov[k] = a1 * Cov[k - 1] + a2 * Cov[k - 2]
    return Cov


# パラメタの設定
a1 = 1.6
a2 = -0.9
sig2 = 1

# [-M, M]区間ではなく、[0, 2M-1]にしている
acov_model = AR2_model(M, a1, a2, sig2) + AR2_model(M, a1, a2, sig2)[1:][::-1]
lag = np.concatenate((np.arange(0, M + 1), -np.arange(1, M + 1)[::-1]))

# 自己共分散関数のフーリエ変換
fft_model = np.fft.fft(acov_model)
PSD_model = np.real(fft_model)
f = np.concatenate((np.arange(0, M + 1) / N, -np.arange(1, M + 1) / N))

# 白色ノイズの生成
WN = np.random.normal(size=N)

# ホワイトノイズのフーリエ変換
fft_WN = np.fft.fft(WN)

# サンプル時系列の生成
fft_sim = np.sqrt(PSD_model) * fft_WN
x_sim = np.real(np.fft.ifft(fft_sim)) / N

# 結果の描画
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Results")

# プロットの例
axes[0, 0].plot(x_sim)
axes[0, 0].set_title("Simulated Time Series")
axes[0, 1].plot(f[: len(PSD_model) // 2], PSD_model[: len(PSD_model) // 2])
axes[0, 1].set_title("Power Spectral Density")
axes[1, 0].plot(acov_model[: len(acov_model) // 2])
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("ACov Model (Log Scale X-axis)")
# 必要に応じて他のプロットも追加
# 実際の出力に合わせたプロットを続けて追加可能

plt.tight_layout()
plt.show()

# %%
