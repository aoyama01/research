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
all_combined_files = [f for f in os.listdir(DIR_EEG) if f.endswith("_EEG_RRI.csv")]
ic(all_combined_files)

# # %%
# os.getcwd()
# os.chdir(DIR_EEG)  # ディレクトリの移動
# ic(all_combined_files[-2:])

# %%
### OPTIONS ###
# 空文字, N1, N2, N3, R, W のいずれかを入力(睡眠段階で切り出さないときは空文字列)
sleep_stage = ""
# 16: meanRR, 17: SDNN（, 18: RMSSD, 19: pNN50, 20: LF, 21: HF, 22: LF/HF）
column_index_of_HRV_measure = 16
### OPTIONS ###

# [3:4]は19E自宅,[11:12]は19O自宅，[12:13]は20A自宅1，[19:20]は20I自宅2
for file_name in all_combined_files:
    # for file_name in all_combined_files[3:4] + all_combined_files[5:6] + all_combined_files[7:12]:
    # ファイルの読み込み
    os.chdir(script_dir)
    os.chdir(DIR_EEG)  # ディレクトリの移動
    print(file_name)
    # data = pd.read_csv(file_name, encoding=detect(file_name)["encoding"])
    # data = pd.read_csv(file_name, encoding="shift-jis")
    with open(file_name, "rb") as file:
        # ファイルのエンコーディングを検出
        detected_encoding = detect(file.read())["encoding"]
    print(detected_encoding)
    # 正しいエンコーディングでファイルを読み込む
    data = pd.read_csv(file_name, encoding=detected_encoding)

    # 睡眠段階でフィルタリング(睡眠段階で切り出さないときは空文字列)
    if sleep_stage != "":
        data = data[data.iloc[:, 2] == sleep_stage]  # 3列目が「sleep_stage」の行を抽出

    # 列名に対応した文字列(csvファイルによって列名の形式が異なるため，こっちで指定)
    labels = [
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Gamma",
        "Sigma",
    ]

    for label_ind, label in enumerate(labels):
        # 解析対象となる列を抽出
        # x1 = data.iloc[:, 9 + label_ind].values
        # x2 = data.iloc[:, column_index_of_HRV_measure].values

        ### エラーチェック(列の取得も) ###
        # 指定した列番号にデータが存在しない場合のチェック
        if column_index_of_HRV_measure >= data.shape[1]:
            print(f"列番号 {column_index_of_HRV_measure} が存在しません．スキップします．")
            break  # 次のファイルへ

        # 解析対象となる列を抽出
        x1 = data.iloc[:, 9 + label_ind].values
        x2 = data.iloc[:, column_index_of_HRV_measure].values

        # x2の欠損値の割合が25%以上の場合のチェック
        nan_ratio = np.isnan(x2).sum() / len(x2)  # 欠損値(nan)の割合
        if nan_ratio > 0.25:
            print(f"x2の欠損値の割合が25%以上 ({nan_ratio * 100:.1f}%) のため，スキップします．")
            break  # 次のファイルへ

        # データ欠損部分の補完により，途中が直線になっているものを除外したい
        # 補完したらSDRRが小さくなるっぽいから(ほとんど1未満になってる感じ)，その箇所が多いデータを除外する
        # sdrrの25%以上が1より小さい場合のチェック
        threshold = 1  # SDRRのしきい値(0.5ぐらいに変更するのもあり)
        sdrr = data["SDRR"].values
        lesser_sdrr_ratio = np.sum(sdrr < threshold) / len(sdrr)  # ゼロの割合
        if lesser_sdrr_ratio > 0.25:
            print(f"{threshold}より小さいSDRRの割合が25%以上 ({lesser_sdrr_ratio * 100:.1f}%) のため，スキップします．")
            break  # 次のファイルへ
        ### エラーチェック(列の取得も) ###

        # DMCAの次数
        orders = [0, 2, 4]

        # スケールの計算
        n = len(x1)  # 時系列の長さ
        n_s = 40
        s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

        # プロット設定
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        # 解析対象の列名を取得
        column_name_of_brain_wave = f"{label} Ratio"
        column_name_of_HRV_measure = data.columns[column_index_of_HRV_measure]

        fig.suptitle(f"DMCA to brain waves and {column_name_of_HRV_measure} ({file_name.replace('_EEG_RRI.csv', '')})", fontsize=18, y=0.935)

        # 行0列0にx1をプロット
        axs[0, 0].plot(range(n), x1, color="green")
        axs[0, 0].set_title(column_name_of_brain_wave, fontsize=14)
        axs[0, 0].set_xlabel("i", fontsize=12)
        axs[0, 0].set_ylabel(column_name_of_brain_wave, fontsize=12)

        # 行1列0にx2をプロット
        axs[1, 0].plot(range(n), x2, color="blue")
        axs[1, 0].set_title(column_name_of_HRV_measure, fontsize=14)
        axs[1, 0].set_xlabel("i", fontsize=12)
        axs[1, 0].set_ylabel(column_name_of_HRV_measure, fontsize=12)
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

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

        os.chdir(script_dir)
        DIR_OUT = "../../../results/" + file_name.replace("_EEG_RRI.csv", "")
        if not os.path.exists(DIR_OUT):
            os.makedirs(DIR_OUT)
        os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
        plt.savefig(
            f"DMCA{f"_{sleep_stage}" if sleep_stage != '' else ''}_{column_name_of_HRV_measure}_{label_ind}_{label}" + ".png",
            dpi=300,
            bbox_inches="tight",
        )  # labelの区切り文字の前までを小文字で取得
        plt.show()

# %%
