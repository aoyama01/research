# %% ライブラリのインポート，ディレクトリの設定，ファイルの読み込み
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chardet import detect
from icecream import ic
from IPython.display import display
from scipy.signal import savgol_filter
from scipy.stats import zscore

###
ic.disable()  # icによるデバッグを無効化
###

# 実行中のスクリプトが存在するディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# カレントディレクトリをスクリプトのディレクトリに設定
os.chdir(script_dir)

DIR_EEG = "../../../data/睡眠段階まとめ_copy"  # ディレクトリの指定

# ディレクトリ内の全てのcsvファイルのファイル名を取得(2023年度のデータはおかしいので除く)
all_combined_files = [f for f in os.listdir(DIR_EEG) if f.endswith("_EEG_RRI.csv")]
ic(all_combined_files)


# %% OPTIONS
### OPTIONS ###
# エラーチェックのみを行うかどうか
is_error_check_only = False
# グラフを出力するかどうか
is_plot = False
# グラフを保存するかどうか
is_savefig = False
# 空文字, N1, N2, N3, R, W のいずれかを入力(睡眠段階で切り出さないときは空文字列．切り出した行数が少ないとエラーが生じて解析できない)
select_sleep_stage = ""
# 除外したい睡眠段階
remove_sleep_stage = ""
# 16: MeanRR, 17: SDNN（, 18: RMSSD, 19: pNN50, 20: LF, 21: HF, 22: LF/HF）
column_index_of_HRV_measure = 16
### OPTIONS ###

# %% 脳波とHRVに対するDMCAを，それぞれのファイルで行う
# 全ファイルの平均を求めるための変数を用意
# ゆらぎ関数を格納する4次元配列[ファイル数, ラベル数, 次数, スケール(総数がファイルごとに異なるので余分にとってる．プロット時にはsliceオブジェクトで範囲を指定する)]
log10F1_4d_array = np.zeros((len(all_combined_files), 6, 3, 40))
log10F2_4d_array = np.zeros((len(all_combined_files), 6, 3, 40))
log10F12_4d_array = np.zeros((len(all_combined_files), 6, 3, 40))
# 相関係数を格納する4次元配列[ファイル数, ラベル数, 次数, スケール]
rho_4d_array = np.zeros((len(all_combined_files), 6, 3, 40))  # 40はスケールの数(len(s))
# 相関係数の積分値を格納する3次元配列[ファイル数, ラベル数, 次数]
rho_integrated = np.zeros((len(all_combined_files), 6, 3))

# Slopeを格納する3次元配列[ファイル数, ラベル数, 次数]
slopes1 = np.zeros((len(all_combined_files), 6, 3))
slopes2 = np.zeros((len(all_combined_files), 6, 3))
slopes12 = np.zeros((len(all_combined_files), 6, 3))
mask = np.ones(len(all_combined_files), dtype=bool)

# ファイルごとに一連の処理を行う
for file_ind, file_name in enumerate(all_combined_files):
    # [3:4]は19E自宅,[11:12]は19O自宅，[12:13]は20A自宅1，[19:20]は20I自宅2
    # for file_name in all_combined_files[3:4] + all_combined_files[5:6] + all_combined_files[7:12]:
    # ファイルの読み込み
    os.chdir(script_dir)
    os.chdir(DIR_EEG)  # ディレクトリの移動
    print(f"{file_name} の読み込み中...")
    # data = pd.read_csv(file_name, encoding=detect(file_name)["encoding"])
    # data = pd.read_csv(file_name, encoding="shift-jis")
    with open(file_name, "rb") as file:
        # ファイルのエンコーディングを検出
        detected_encoding = detect(file.read())["encoding"]
    print(f"検出されたファイルエンコーディング: {detected_encoding}")
    # 正しいエンコーディングでファイルを読み込む
    data = pd.read_csv(file_name, encoding=detected_encoding)

    # 睡眠段階でフィルタリング(睡眠段階で切り出さないときは空文字列)
    if select_sleep_stage != "":
        data = data[data.iloc[:, 2] == select_sleep_stage]  # 3列目が「sleep_stage」の行を抽出
    if remove_sleep_stage != "":
        data = data[data.iloc[:, 2] != remove_sleep_stage]  # 3列目が「sleep_stage」でない行を抽出

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
            print(f"列番号 {column_index_of_HRV_measure} が存在しません．このファイルの処理をスキップします．\n")
            mask[file_ind] = False
            break  # 次のファイルへ

        # 解析対象となる列を抽出
        x1 = data.iloc[:, 9 + label_ind].values
        # # 睡眠段階を解析する場合
        # x1 = data.iloc[:, 2].values
        # # 置き換え用の辞書を定義
        # # replace_dict = {"W": 0, "R": -1, "N1": -2, "N2": -3, "N3": -4}
        # # replace_dict = {"W": 0, "R": -1, "N1": -1, "N2": -1, "N3": -1}  # 覚醒から睡眠への移行を見たい場合はこの二値化でOK？
        # # replace_dict = {"W": 0, "N1": -1, "N2": -2, "N3": -3}  # Rを除外して考える場合
        # replace_dict = {"W": 0, "N1": -1, "N2": -1, "N3": -1}  # Rを除外して覚醒から睡眠への移行を見たい場合場合
        # # 辞書を使って置き換え
        # x1 = [replace_dict[element] for element in x1]

        x2 = data.iloc[:, column_index_of_HRV_measure].values

        # 解析対象の列名を取得
        column_name_of_brain_wave = f"{label} Ratio"
        column_name_of_HRV_measure = data.columns[column_index_of_HRV_measure]

        # x2の欠損値の割合が25%以上の場合のチェック
        nan_ratio = np.isnan(x2).sum() / len(x2)  # 欠損値(nan)の割合
        if nan_ratio > 0.25:
            print(f"{column_name_of_HRV_measure}の欠損値の割合が25%以上 ({nan_ratio * 100:.1f}%) のため，このファイルの処理をスキップします．\n")
            mask[file_ind] = False
            break  # 次のファイルへ

        # データ欠損部分の補完により，途中が直線になっているものを除外したい
        # 補完したらSDRRが小さくなるっぽいから(ほとんど1未満になってる感じ)，その箇所が多いデータを除外する
        # sdrrの25%以上が1より小さい場合のチェック
        threshold = 1  # SDRRのしきい値(0.5ぐらいに変更するのもあり)
        sdrr = data["SDRR"].values
        lesser_sdrr_ratio = np.sum(sdrr < threshold) / len(sdrr)  # ゼロの割合
        if lesser_sdrr_ratio > 0.25:
            print(
                f"しきい値({threshold:.1f})より小さいSDRRの割合が25%以上 ({lesser_sdrr_ratio * 100:.1f}%) のため，このファイルの処理をスキップします．\n"
            )
            mask[file_ind] = False
            break  # 次のファイルへ

        print("このファイルの処理でエラーは生じませんでした．\n")

        # エラーチェックのみを行う場合はグラフを描画せずに終了
        if is_error_check_only:
            break  # デバッグ用
        ### エラーチェック(列の取得も) ###

        # DMCAの次数
        orders = [0, 2, 4]

        # スケールの計算
        n = len(x1)  # 時系列の長さ
        n_s = 40
        s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)
        print(f"スケールの計算: {len(s)}")

        # 解析対象の列名を取得
        # column_name_of_brain_wave = f"{label} Ratio"
        # column_name_of_HRV_measure = data.columns[column_index_of_HRV_measure]

        if is_plot:
            # プロット設定
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
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
            axs[1, 0].set_ylabel(f"{column_name_of_HRV_measure} [ms]", fontsize=12)
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

            # ゆらぎ関数を格納(全ファイルの平均を求めるため)
            log10F1_padded = np.zeros(40)  # 長さ34にゼロ埋め
            log10F1_padded[: len(log10F1)] = log10F1  # rho の値を先頭に埋め込む
            log10F1_4d_array[file_ind, label_ind, col_idx - 1] = log10F1_padded
            log10F2_padded = np.zeros(40)  # 長さ34にゼロ埋め
            log10F2_padded[: len(log10F2)] = log10F2  # rho の値を先頭に埋め込む
            log10F2_4d_array[file_ind, label_ind, col_idx - 1] = log10F2_padded
            log10F12_padded = np.zeros(40)  # 長さ34にゼロ埋め
            log10F12_padded[: len(log10F12)] = log10F12  # rho の値を先頭に埋め込む
            log10F12_4d_array[file_ind, label_ind, col_idx - 1] = log10F12_padded

            # 相関係数を格納(全ファイルの平均を求めるため)
            rho_padded = np.zeros(40)  # 長さ34にゼロ埋め
            rho_padded[: len(rho)] = rho  # rho の値を先頭に埋め込む
            rho_4d_array[file_ind, label_ind, col_idx - 1] = rho_padded

            # Zスコアをもとに外れ値を除外
            valid_ind = (
                (np.abs(zscore(log10F1)) < 3) & (np.abs(zscore(log10F2)) < 3) & (np.abs(zscore(log10F12)) < 3)
            )  # 外れ値除外後のインデックスを取得
            log10F1 = log10F1[valid_ind]
            log10F2 = log10F2[valid_ind]
            log10F12 = log10F12[valid_ind]
            s_clean = np.array(s)[valid_ind]  # sも対応するインデックスでフィルタリング
            rho = rho[valid_ind]  # rhoも対応するインデックスでフィルタリング

            # 相関係数の積分値を格納
            rho_integrated[file_ind, label_ind, col_idx - 1] = np.trapz(rho, np.log10(s_clean))

            coeff1 = np.polyfit(np.log10(s_clean), log10F1, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            coeff2 = np.polyfit(np.log10(s_clean), log10F2, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            coeff12 = np.polyfit(np.log10(s_clean), log10F12, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            fitted1 = np.poly1d(coeff1)  # 回帰直線の式
            fitted2 = np.poly1d(coeff2)  # 回帰直線の式
            fitted12 = np.poly1d(coeff12)  # 回帰直線の式

            # Slopeを格納
            slopes1[file_ind, label_ind, col_idx - 1] = coeff1[0]
            slopes2[file_ind, label_ind, col_idx - 1] = coeff2[0]
            slopes12[file_ind, label_ind, col_idx - 1] = coeff12[0]

            if is_plot:
                # 行0, 列col_idxにCross-correlationプロット
                axs[0, col_idx].plot(np.log10(s_clean), rho, color="red")
                axs[0, col_idx].set_ylim(-1, 1)
                axs[0, col_idx].axhline(0, linestyle="--", color="gray")
                axs[0, col_idx].set_title(f"DMCA{order}\nCross-correlation", fontsize=12)
                axs[0, col_idx].set_xlabel("log10(s)", fontsize=12)
                axs[0, col_idx].set_ylabel("rho", fontsize=12)
                axs[0, col_idx].legend(title=f"Max:  {max(rho):.3f}\nMin:  {min(rho):.3f}", title_fontsize=10.5)

                # 行1, 列col_idxにSlopeプロット
                axs[1, col_idx].scatter(np.log10(s_clean), log10F1, color="green", label="log10(F1)", marker="^")
                axs[1, col_idx].scatter(np.log10(s_clean), log10F2, color="blue", label="log10(F2)", marker="s")
                axs[1, col_idx].scatter(np.log10(s_clean), log10F12, color="red", label="log10(|F12|)/2", marker="o")
                axs[1, col_idx].plot(np.log10(s), fitted1(np.log10(s)), color="green", linestyle="--")
                axs[1, col_idx].plot(np.log10(s), fitted2(np.log10(s)), color="blue", linestyle="--")
                axs[1, col_idx].plot(np.log10(s), fitted12(np.log10(s)), color="red", linestyle="--")
                # y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
                # y_max = max(log10F1.max(), log10F2.max(), log10F12.max())
                # axs[1, col_idx].set_ylim(y_min, y_max)
                axs[1, col_idx].set_title(
                    f"DMCA{order}\nSlope1 = {coeff1[0]:.3f},  Slope2 = {coeff2[0]:.3f},  Slope12 = {coeff12[0]:.3f}", fontsize=12
                )
                axs[1, col_idx].set_xlabel("log10(s)", fontsize=12)
                axs[1, col_idx].set_ylabel("log10(F(s))", fontsize=12)
                axs[1, col_idx].legend()

        if is_plot:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

            if is_savefig:
                # グラフの保存
                os.chdir(script_dir)
                DIR_OUT = "../../../results/" + file_name.replace("_EEG_RRI.csv", "")
                if not os.path.exists(DIR_OUT):
                    os.makedirs(DIR_OUT)
                os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
                plt.savefig(
                    f"DMCA_{column_name_of_HRV_measure}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}_{label_ind}_{label}" + ".png",
                    dpi=300,
                    bbox_inches="tight",
                )
            # グラフの表示
            plt.show()

        # break  # 睡眠段階で解析するから，ラベルのループは1回でOK


# %% Fとrhoのマスクと平均値の計算
log10F1_4d_array_masked = log10F1_4d_array[mask]
log10F2_4d_array_masked = log10F2_4d_array[mask]
log10F12_4d_array_masked = log10F12_4d_array[mask]
rho_4d_array_masked = rho_4d_array[mask]

log10F1_mean = np.mean(log10F1_4d_array_masked, axis=0)
log10F2_mean = np.mean(log10F2_4d_array_masked, axis=0)
log10F12_mean = np.mean(log10F12_4d_array_masked, axis=0)
rho_mean = np.mean(rho_4d_array_masked, axis=0)

print(f"log10F12_4d_array_masked: {log10F12_4d_array_masked.shape}")
print(f"log10F12_mean: {log10F12_mean.shape}")
print(f"rho_4d_array_masked.shape: {rho_4d_array_masked.shape}")
print(f"rho_maen.shape: {rho_mean.shape}")


# %% すべてのファイルにおける相関係数とゆらぎ関数の平均を全ての脳波でプロット(次数は指定する)

# プロットする範囲をsliceオブジェクトにする
range_slice = slice(1, len(s))
print(f"len(s): {len(s)}")

# DMCAの次数(0, 2, 4)
# order = 2
for order in orders:
    fig, axs = plt.subplots(2, 5, figsize=(30, 14))
    fig.suptitle(
        f"Mean XCorr of DMCA{order} to Brain Waves and {column_name_of_HRV_measure}  {f'(Stage: {select_sleep_stage})' if select_sleep_stage != '' else ''}",
        fontsize=18,
        y=0.935,
    )

    for label_ind, label in enumerate(labels[:5]):
        # 4次の結果(rho_mean[label_ind][2])のみを表示
        # order // 2 の処理 → 0 // 2 = 0,  2 // 2 = 1,  4 // 2 = 2
        log10F1_mean_dmca4 = log10F1_mean[label_ind][order // 2][range_slice]
        log10F2_mean_dmca4 = log10F2_mean[label_ind][order // 2][range_slice]
        log10F12_mean_dmca4 = log10F12_mean[label_ind][order // 2][range_slice]
        rho_mean_dmca4 = rho_mean[label_ind][order // 2][range_slice]  # 最初のやつはハズレ値っぽいから除外

        # [label_ind(li)] を [label_ind/3, label_ind%3]

        # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

        axs[0, label_ind].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
        # axs[label_ind//3, label_ind%3].set_xlim(0.612110372200782, 2.523022279175993)
        axs[0, label_ind].set_ylim(-1, 1)
        axs[0, label_ind].axhline(0, linestyle="--", color="gray")
        axs[0, label_ind].set_title(f"{label} Ratio", fontsize=16)
        axs[0, label_ind].set_xlabel("log10(s)", fontsize=14)
        axs[0, label_ind].set_ylabel("rho", fontsize=14)
        axs[0, label_ind].legend(
            title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
            title_fontsize=12,
        )

        coeff1_mean = np.polyfit(np.log10(s[range_slice]), log10F1_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        coeff2_mean = np.polyfit(np.log10(s[range_slice]), log10F2_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        coeff12_mean = np.polyfit(np.log10(s[range_slice]), log10F12_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        fitted1_mean = np.poly1d(coeff1_mean)  # 回帰直線の式
        fitted2_mean = np.poly1d(coeff2_mean)  # 回帰直線の式
        fitted12_mean = np.poly1d(coeff12_mean)  # 回帰直線の式

        axs[1, label_ind].scatter(np.log10(s[range_slice]), log10F1_mean_dmca4, color="green", label="log10(F1)", marker="^", facecolors="none", s=75)
        axs[1, label_ind].scatter(np.log10(s[range_slice]), log10F2_mean_dmca4, color="blue", label="log10(F2)", marker="s", facecolors="none", s=75)
        axs[1, label_ind].scatter(np.log10(s[range_slice]), log10F12_mean_dmca4, color="red", label="log10(|F12|)/2", marker="x", s=75)
        # axs[1, label_ind].plot(np.log10(s), fitted1_mean(np.log10(s)), color="green", linestyle="--")
        # axs[1, label_ind].plot(np.log10(s), fitted2_mean(np.log10(s)), color="blue", linestyle="--")
        axs[1, label_ind].plot(np.log10(s[range_slice]), fitted12_mean(np.log10(s[range_slice])), color="red", linestyle="--")
        # y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
        # y_max = max(log10F1.max(), log10F2.max(), log10F12.max())
        # axs[1, label_ind].set_ylim(y_min, y_max)
        axs[1, label_ind].set_title(
            # f"DMCA{order}\nSlope1 = {coeff1_mean[0]:.3f},  Slope2 = {coeff2_mean[0]:.3f},  Slope12 = {coeff12_mean[0]:.3f}", fontsize=16
            f"DMCA{order}\nSlope1 = {coeff1_mean[0]:.3f},  Slope2 = {coeff2_mean[0]:.3f},  Slope12 = {coeff12_mean[0]:.3f}",
            fontsize=16,
        )
        axs[1, label_ind].set_xlabel("log10(s)", fontsize=14)
        axs[1, label_ind].set_ylabel("log10(F(s))", fontsize=14)
        axs[1, label_ind].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

    # グラフの保存と表示
    if is_savefig:
        # グラフの保存
        os.chdir(script_dir)
        DIR_OUT = "../../../results/Mean/"
        if not os.path.exists(DIR_OUT):
            os.makedirs(DIR_OUT)
        os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
        plt.savefig(
            f"Mean_XCorrMean_{column_name_of_HRV_measure}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}" + ".png",
            dpi=300,
            bbox_inches="tight",
        )
    # グラフの表示
    plt.show()


# %% 脳波ごとに，DMCA(0次，2次，4次)の相関係数の平均値をプロット
# for label_ind, label in enumerate(labels):
#     fig, axs = plt.subplots(1, 3, figsize=(20, 8))
#     fig.suptitle(f"DMCA to {label} Ratio and {column_name_of_HRV_measure}", fontsize=18, y=0.935)

#     for col_idx, order in enumerate(orders):
#         axs[col_idx].plot(np.log10(s[1:]), rho_maen[label_ind][col_idx][1 : len(s)], color="red")
#         # axs[col_idx].set_xlim(0.612110372200782, 2.523022279175993)
#         axs[col_idx].set_ylim(-1, 1)
#         axs[col_idx].axhline(0, linestyle="--", color="gray")
#         axs[col_idx].set_title(f"DMCA{order}\nCross-correlation", fontsize=12)
#         axs[col_idx].set_xlabel("log10(s)", fontsize=12)
#         axs[col_idx].set_ylabel("rho", fontsize=12)
#         axs[col_idx].legend(
#             title=f"Max:  {max(rho_maen[label_ind][col_idx][1 : len(s)]):.3f}\nMin:  {min(rho_maen[label_ind][col_idx][1 : len(s)]):.3f}",
#             title_fontsize=10.5,
#         )

#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

#     # グラフの保存
#     os.chdir(script_dir)
#     DIR_OUT = "../../../results/Average/"
#     if not os.path.exists(DIR_OUT):
#         os.makedirs(DIR_OUT)
#     os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
#     plt.savefig(
#         f"DMCA_XCorrMean{f'_{sleep_stage}' if sleep_stage != '' else ''}_{column_name_of_HRV_measure}_{label_ind}_{label}" + ".png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()


# %% DMCA(4次)の相関係数の平均値をすべての脳波でプロット
fig, axs = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle(
    f"Mean XCorr of DMCA4 to Brain Waves and {column_name_of_HRV_measure}  {f'(Stage: {select_sleep_stage})' if select_sleep_stage != '' else ''}",
    fontsize=18,
    y=0.935,
)

for label_ind, label in enumerate(labels[:5]):
    # 4次の結果(rho_maen[label_ind][2])のみを表示
    rho_mean_dmca4 = rho_mean[label_ind][0][1 : len(s)]  # 最初のやつはハズレ値っぽいから除外

    # [label_ind(li)] を [label_ind/3, label_ind%3]

    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

    axs[label_ind // 3, label_ind % 3].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
    # axs[label_ind//3, label_ind%3].set_xlim(0.612110372200782, 2.523022279175993)
    axs[label_ind // 3, label_ind % 3].set_ylim(-1, 1)
    axs[label_ind // 3, label_ind % 3].axhline(0, linestyle="--", color="gray")
    axs[label_ind // 3, label_ind % 3].set_title(f"{label} Ratio", fontsize=16)
    axs[label_ind // 3, label_ind % 3].set_xlabel("log10(s)", fontsize=14)
    axs[label_ind // 3, label_ind % 3].set_ylabel("rho", fontsize=14)
    axs[label_ind // 3, label_ind % 3].legend(
        title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
        title_fontsize=12,
    )
# 6つ目のサブプロットを空白に設定
axs[1, 2].axis("off")  # 軸を非表示
plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

# グラフの保存と表示
if is_savefig:
    # グラフの保存
    os.chdir(script_dir)
    DIR_OUT = "../../../results/Mean/"
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
    os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
    plt.savefig(
        f"Mean_XCorrMean_{column_name_of_HRV_measure}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}" + ".png",
        dpi=300,
        bbox_inches="tight",
    )
# グラフの表示
plt.show()


# %%
# x軸の範囲を取得
# xlim = axs[2].get_xlim()
# print("x軸の範囲:", xlim)


# %% 相関係数の積分値およびSlopeの平均値を表として出力
# 正常なデータのみを抽出
rho_integrated_masked = rho_integrated[mask]
slopes1_masked = slopes1[mask]
slopes2_masked = slopes2[mask]
slopes12_masked = slopes12[mask]

# DataFrameに変換する際の行名と列名
row_names = labels
col_names = ["DMCA0", "DMCA2", "DMCA4"]

rho_integrated_mean = np.mean(rho_integrated_masked, axis=0)
rho_integrated_mean_df = pd.DataFrame(rho_integrated_mean, index=row_names, columns=col_names)

slopes1_mean = np.mean(slopes1_masked, axis=0)
slopes1_mean_df = pd.DataFrame(slopes1_mean, index=row_names, columns=col_names)

slopes2_mean = np.mean(slopes2_masked, axis=0)
slopes2_mean_df = pd.DataFrame(slopes2_mean, index=row_names, columns=col_names)

slopes12_mean = np.mean(slopes12_masked, axis=0)
slopes12_mean_df = pd.DataFrame(slopes12_mean, index=row_names, columns=col_names)

print(f"相関係数の積分 (EEG & {column_name_of_HRV_measure}) の平均値")
display(rho_integrated_mean_df.round(3))
print("Slope1 (EEG) の平均値")
display(slopes1_mean_df.round(3))
print(f"\nSlope2 ({column_name_of_HRV_measure}) の平均値")
display(slopes2_mean_df.round(3))
print(f"\nSlope12 (EEG & {column_name_of_HRV_measure}) の平均値")
display(slopes12_mean_df.round(3))

# %%
