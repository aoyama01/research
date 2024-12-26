# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ファイルのパスを設定
file_path = "../../../data/睡眠段階まとめ_copy/2019A自宅_EEG_RRI.csv"

# 1つ目のファイルを読み込む
data = pd.read_csv(file_path, encoding="shift-jis")

# クロス相関に必要な列を抽出：1つ目のファイルからDelta_Ratio、2つ目のファイルからRRI
x1 = data["Delta_Ratio"].values
x2 = data["meanRR"].values

# 平均を0、標準偏差を1に標準化
# x1 = (x1 - np.nanmean(x1)) / np.nanstd(x1)
# x2 = (x2 - np.nanmean(x2)) / np.nanstd(x2)

# 時系列の長さ
n = len(x1)

# プロット設定
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(range(n), x1, color="green")
axs[0, 0].set_title("Delta ratio")
axs[0, 0].set_xlabel("i")
axs[0, 0].set_ylabel("Delta ratio")

# # 現在の横軸の目盛り位置とラベルを取得
# current_ticks = plt.gca().get_xticks()
# # 目盛りを30倍した新しいラベルを設定
# new_labels = [int(tick * 30) for tick in current_ticks]
# plt.xticks(ticks=current_ticks, labels=new_labels)
# plt.xlim(-75, 1425)

axs[0, 1].plot(range(n), x2, color="blue")
axs[0, 1].set_title("RR-interval")
axs[0, 1].set_xlabel("i")
axs[0, 1].set_ylabel("RRI")

# DMAで解析するスケールは奇数のみ
n_s = 20
s = np.unique(
    np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1
).astype(int)

# 初期化
F1 = []
F2 = []
F12_sq = []

# 平均を0、標準偏差を1に標準化
x1 = (x1 - np.nanmean(x1)) / np.nanstd(x1)
x2 = (x2 - np.nanmean(x2)) / np.nanstd(x2)

# データ内のNaNをデータ全体の中央値で埋める
# x1 = np.nan_to_num(x1, nan=np.nanmedian(x1))
# x2 = np.nan_to_num(x2, nan=np.nanmedian(x2))

# データ内のNaNを線形補間で埋める
x1 = pd.Series(x1).interpolate(limit_direction="both").values
x2 = pd.Series(x2).interpolate(limit_direction="both").values

# 時系列の積分
y1 = np.cumsum(x1)
y2 = np.cumsum(x2)


# 0次DMAとDMCA
for si in s:
    # Detrending
    # REVIEW: 第4引数のmodeによってCross-correlationとScalingのグラフが異なる
    # (interp(線形補間)のグラフがR言語でプロットしたやつとめちゃ似てる)
    y1_detrend = y1 - savgol_filter(y1, window_length=si, polyorder=0, mode="interp")
    y2_detrend = y2 - savgol_filter(y2, window_length=si, polyorder=0, mode="interp")
    F1.append(np.sqrt(np.mean(y1_detrend**2)))
    F2.append(np.sqrt(np.mean(y2_detrend**2)))
    F12_sq.append(np.mean(y1_detrend * y2_detrend))

F1 = np.array(F1)
F2 = np.array(F2)
F12_sq = np.array(F12_sq)

rho = F12_sq / (F1 * F2)

# クロス相関プロット
axs[1, 0].plot(np.log10(s), rho, color="red")
axs[1, 0].set_ylim(-1, 1)
axs[1, 0].axhline(0, linestyle="--", color="gray")
axs[1, 0].set_title("Cross-correlation")
axs[1, 0].set_xlabel("log10(s)")
axs[1, 0].set_ylabel("rho")

# スケーリングプロット
log10F1 = np.log10(F1)
log10F2 = np.log10(F2)
log10F12 = np.log10(np.abs(F12_sq)) / 2

y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
y_max = max(log10F1.max(), log10F2.max(), log10F12.max())

axs[1, 1].scatter(np.log10(s), log10F1, color="green", label="log10(F1)", marker="^")
axs[1, 1].scatter(np.log10(s), log10F2, color="blue", label="log10(F2)", marker="s")
axs[1, 1].scatter(
    np.log10(s), log10F12, color="red", label="log10(|F12|)/2", marker="o"
)
axs[1, 1].set_ylim(y_min, y_max)
axs[1, 1].set_title("Scaling")
axs[1, 1].set_xlabel("log10(s)")
axs[1, 1].set_ylabel("log10(F(s))")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# 1/f ゆらぎは数十秒から数時間あるから
# 30秒間隔のサンプリングでも問題ない？（2024/11/06）
