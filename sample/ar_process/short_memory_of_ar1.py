# %%
import matplotlib.pyplot as plt
import numpy as np

# 定数の設定
sig = 1.0  # σ^2
a = 0.9  # a

# 周波数範囲の設定 (両対数プロットのため、対数的に広範囲に設定)
f = np.logspace(-5, -0.25, 500)  # 10e-5 から 10e-0.25 まで500分割したやつを生成

# S(f) の計算
S_f = sig**2 / ((1 - a) ** 2 - 2 * a * (np.cos(2 * np.pi * f) - 1))

# プロット
plt.figure(figsize=(8, 6))
plt.loglog(f, S_f, label=r"$S(f) = \frac{\sigma^2}{(1-a)^2 + 4\pi^2 f^2}$", color="b")
plt.xlabel("f (Hz)")
plt.ylabel("S(f)")
plt.title("Log-Log Plot of S(f)")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# %%
