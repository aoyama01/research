# %%
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chardet import detect
from icecream import ic
from scipy.signal import savgol_filter
from scipy.stats import zscore

###
# ic.disable()  # icによるデバッグを無効化
###

# 実行中のスクリプトが存在するディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# カレントディレクトリをスクリプトのディレクトリに設定
os.chdir(script_dir)

DIR_EEG = "../../../data/睡眠段階まとめ_copy"  # ディレクトリの指定

# ディレクトリ内の全てのcsvファイルのファイル名を取得(2023年度のデータはおかしいので除く)
all_combined_files = [f for f in os.listdir(DIR_EEG) if f.endswith("_EEG_RRI.csv") and "2023" not in f]
ic(all_combined_files)

# # %%
# os.getcwd()
# os.chdir(DIR_EEG)  # ディレクトリの移動
# ic(all_combined_files[-2:])

# %%
for file_name in all_combined_files[11:12]:
    # for file_name in all_combined_files[3:4] + all_combined_files[5:6] + all_combined_files[7:12]:
    # ファイルの読み込み
    os.chdir(script_dir)
    os.chdir(DIR_EEG)  # ディレクトリの移動
    ic(file_name)
    # data = pd.read_csv(file_name, encoding=detect(file_name)["encoding"])
    # data = pd.read_csv(file_name, encoding="shift-jis")
    with open(file_name, "rb") as file:
        # ファイルのエンコーディングを検出
        detected_encoding = detect(file.read())["encoding"]
    ic(detected_encoding)
    # 正しいエンコーディングでファイルを読み込む
    data = pd.read_csv(file_name, encoding=detected_encoding)

    # 睡眠段階でフィルタリング
    # sleep_stage = "N3"  # N1, N2, N3, R, W
    # data = data[data.iloc[:, 2] == sleep_stage]  # 3列目が「sleep_stage」の行を抽出

    # 横軸(data columns)の文字列
    labels = [
        "Delta_Ratio",
        "Theta_Ratio",
        "Alpha_Ratio",
        "Beta_Ratio",
        "Gamma_Ratio",
        "Sigma_Ratio",
    ]

    for label in labels:
        # 列の指定
        column1 = label
        column2 = "SDRR"
        # 解析対象となる列を抽出
        x1 = data[column1].values
        x2 = data[column2].values

        # DMCAの次数
        orders = [0, 2, 4]

        # スケールの計算
        n = len(x1)  # 時系列の長さ
        n_s = 40
        s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

        # プロット設定
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        # 行0列0にx1をプロット
        axs[0, 0].plot(range(n), x1, color="green")
        axs[0, 0].set_title(column1, fontsize=14)
        axs[0, 0].set_xlabel("i", fontsize=12)
        axs[0, 0].set_ylabel(column1, fontsize=12)

        # 行1列0にx2をプロット
        axs[1, 0].plot(range(n), x2, color="blue")
        axs[1, 0].set_title(column2, fontsize=14)
        axs[1, 0].set_xlabel("i", fontsize=12)
        axs[1, 0].set_ylabel(column2, fontsize=12)
        # 途中が直線になっているのは，データに欠損があり，その部分を補完しているため

        # 各 order に対応する Cross-correlation と Slope のプロット
        for col_idx, order in enumerate(orders, start=1):
            # 初期化
            F1 = []
            F2 = []
            F12_sq = []

            # 標準化と補間
            x1_norm = (x1 - np.nanmean(x1)) / np.nanstd(x1)
            x2_norm = (x2 - np.nanmean(x2)) / np.nanstd(x2)

            # データ内のNaNをデータ全体の中央値で埋める
            # x1_norm = np.nan_to_num(x1, nan=np.nanmedian(x1))
            # x2_norm = np.nan_to_num(x2, nan=np.nanmedian(x2))

            # データ内のNaNを線形補間で埋める
            x1_norm = pd.Series(x1_norm).interpolate(limit_direction="both").values
            x2_norm = pd.Series(x2_norm).interpolate(limit_direction="both").values

            # 積分時系列の計算
            y1 = np.cumsum(x1_norm)
            y2 = np.cumsum(x2_norm)

            # DMA計算
            for si in s:
                # Detrending
                # REVIEW: 第4引数のmodeによってCross-correlationとScalingのグラフが異なる
                # (interp(線形補間)のグラフがR言語でプロットしたやつとめちゃ似てる)
                y1_detrend = y1 - savgol_filter(y1, window_length=si, polyorder=order, mode="interp")
                y2_detrend = y2 - savgol_filter(y2, window_length=si, polyorder=order, mode="interp")
                F1.append(np.sqrt(np.mean(y1_detrend**2)))
                F2.append(np.sqrt(np.mean(y2_detrend**2)))
                F12_sq.append(np.mean(y1_detrend * y2_detrend))

            F1 = np.array(F1)
            F2 = np.array(F2)
            F12_sq = np.array(F12_sq)

            rho = F12_sq / (F1 * F2)

            # スケーリングプロット
            log10F1 = np.log10(F1)
            log10F2 = np.log10(F2)
            log10F12 = np.log10(np.abs(F12_sq)) / 2

            # Zスコアをもとに外れ値を除外
            valid_ind = (
                (np.abs(zscore(log10F1)) < 3) & (np.abs(zscore(log10F2)) < 3) & (np.abs(zscore(log10F12)) < 3)
            )  # 外れ値除外後のインデックスを取得
            log10F1 = log10F1[valid_ind]
            log10F2 = log10F2[valid_ind]
            log10F12 = log10F12[valid_ind]
            s_clean = np.array(s)[valid_ind]  # sも対応するインデックスでフィルタリング
            rho = rho[valid_ind]  # rhoも対応するインデックスでフィルタリング

            coeff = np.polyfit(np.log10(s_clean), log10F12, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            fitted = np.poly1d(coeff)  # 回帰直線の式
            ic(coeff[0])  # 回帰係数

            # 行0, 列col_idxにCross-correlationプロット
            axs[0, col_idx].plot(np.log10(s_clean), rho, color="red")
            axs[0, col_idx].set_ylim(-1, 1)
            axs[0, col_idx].axhline(0, linestyle="--", color="gray")
            axs[0, col_idx].set_title(f"DMCA{order}\nCross-correlation", fontsize=12)
            axs[0, col_idx].set_xlabel("log10(s)", fontsize=12)
            axs[0, col_idx].set_ylabel("rho", fontsize=12)

            # 行1, 列col_idxにSlopeプロット
            axs[1, col_idx].scatter(np.log10(s_clean), log10F1, color="green", label="log10(F1)", marker="^")
            axs[1, col_idx].scatter(np.log10(s_clean), log10F2, color="blue", label="log10(F2)", marker="s")
            axs[1, col_idx].scatter(np.log10(s_clean), log10F12, color="red", label="log10(|F12|)/2", marker="o")
            axs[1, col_idx].plot(np.log10(s), fitted(np.log10(s)), color="black", linestyle="--", label="Fitted")
            # y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
            # y_max = max(log10F1.max(), log10F2.max(), log10F12.max())
            # axs[1, col_idx].set_ylim(y_min, y_max)
            axs[1, col_idx].set_title(f"DMCA{order}\nSlope = {coeff[0]:.3f}", fontsize=12)
            axs[1, col_idx].set_xlabel("log10(s)", fontsize=12)
            axs[1, col_idx].set_ylabel("log10(F(s))", fontsize=12)
            axs[1, col_idx].legend()

        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# %%
