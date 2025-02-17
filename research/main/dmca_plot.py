# %% ライブラリのインポート，ディレクトリの設定，ファイルの読み込み
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from chardet import detect
from icecream import ic
from IPython.display import display
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, zscore

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

# %% MEMO
#
# "W"を除外すると，SDRR，HRVI，TINNでの相関が強くなる．これらはいずれも心拍間隔のばらつき度合いを表す指標
# LF，LF/HFでの相関も強くなる．
# だけど，MeanRRでの相関はめちゃ小さくなる．
#
# "R"を除外すると，MeanRRでの相関がちょっと強くなる
#
#
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
# 除外したい睡眠段階その2
remove_sleep_stage_2 = ""
# 16:MeanRR, 17:SDRR, 18:RMSSD, 19:pNN50, 20:HRVI. 21:TINN, 22:LF, 23:HF, 24:LF/HF
column_index_of_HRV_measure = 18
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
    if remove_sleep_stage_2 != "":
        data = data[data.iloc[:, 2] != remove_sleep_stage_2]  # 3列目が「sleep_stage」でない行を抽出

    # 列名に対応した文字列(csvファイルによって列名の形式が異なるため，こっちで指定)
    eeg_bands = [
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Gamma",
        "Sigma",
    ]

    for band_ind, eeg_band in enumerate(eeg_bands):
        # 解析対象となる列を抽出
        # x1 = data.iloc[:, 9 + label_ind].values
        # x2 = data.iloc[:, column_index_of_HRV_measure].values

        ### エラーチェック(列の取得も) ###
        # 指定した列番号にデータが存在しない場合のチェック
        if data.columns[16] != "MeanRR":
            print(f"列番号 {16} がMeanRRではありません．このファイルの処理をスキップします．\n")
            mask[file_ind] = False
            break  # 次のファイルへ

        # if column_index_of_HRV_measure >= data.shape[1]:
        #     print(f"列番号 {column_index_of_HRV_measure} が存在しません．このファイルの処理をスキップします．\n")
        #     mask[file_ind] = False
        #     break  # 次のファイルへ

        # 解析対象となる列を抽出
        x1 = data.iloc[:, 9 + band_ind].values
        # # x1に強引にDeltaを入れたい場合
        # break
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
        column_name_of_brain_wave = f"{eeg_band} Ratio"
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
            log10F1_4d_array[file_ind, band_ind, col_idx - 1] = log10F1_padded
            log10F2_padded = np.zeros(40)  # 長さ34にゼロ埋め
            log10F2_padded[: len(log10F2)] = log10F2  # rho の値を先頭に埋め込む
            log10F2_4d_array[file_ind, band_ind, col_idx - 1] = log10F2_padded
            log10F12_padded = np.zeros(40)  # 長さ34にゼロ埋め
            log10F12_padded[: len(log10F12)] = log10F12  # rho の値を先頭に埋め込む
            log10F12_4d_array[file_ind, band_ind, col_idx - 1] = log10F12_padded

            # 相関係数を格納(全ファイルの平均を求めるため)
            rho_padded = np.zeros(40)  # 長さ34にゼロ埋め
            rho_padded[: len(rho)] = rho  # rho の値を先頭に埋め込む
            rho_4d_array[file_ind, band_ind, col_idx - 1] = rho_padded

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
            rho_integrated[file_ind, band_ind, col_idx - 1] = np.trapz(rho, np.log10(s_clean))

            coeff1 = np.polyfit(np.log10(s_clean), log10F1, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            coeff2 = np.polyfit(np.log10(s_clean), log10F2, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            coeff12 = np.polyfit(np.log10(s_clean), log10F12, 1)  # 回帰係数(polyfitは傾きと切片を返す)
            fitted1 = np.poly1d(coeff1)  # 回帰直線の式
            fitted2 = np.poly1d(coeff2)  # 回帰直線の式
            fitted12 = np.poly1d(coeff12)  # 回帰直線の式

            # Slopeを格納
            slopes1[file_ind, band_ind, col_idx - 1] = coeff1[0]
            slopes2[file_ind, band_ind, col_idx - 1] = coeff2[0]
            slopes12[file_ind, band_ind, col_idx - 1] = coeff12[0]

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
                DIR_OUT = "../../../results/persons/" + file_name.replace("_EEG_RRI.csv", "")
                if not os.path.exists(DIR_OUT):
                    os.makedirs(DIR_OUT)
                os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
                plt.savefig(
                    f"DMCA_{column_name_of_HRV_measure}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}_{band_ind}_{eeg_band}" + ".png",
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

# %% Slope1 と Slope2 の平均と標準偏差を求める
# バンドでの平均求めてみる
log10F1_band_mean = np.mean(log10F1_4d_array_masked, axis=1)
log10F2_band_mean = np.mean(log10F2_4d_array_masked, axis=1)

# 4次の部分だけ取得
log10F1_band_mean_dmca4 = log10F1_band_mean[:, 2:3, :]
log10F2_band_mean_dmca4 = log10F2_band_mean[:, 2:3, :]
# (15, 1, 40) から (15, 40) に reshape
log10F1_band_mean_dmca4 = log10F1_band_mean_dmca4.reshape(15, 40)
log10F2_band_mean_dmca4 = log10F2_band_mean_dmca4.reshape(15, 40)

# personごとのスロープを求める
range_slice = slice(1, len(s))  # 範囲をsliceオブジェクトにする
slope1_each_person = []
slope2_each_person = []
for person_i in range(log10F1_band_mean_dmca4.shape[0]):
    slope1_each_person.append(
        np.polyfit(np.log10(s[range_slice]), log10F1_band_mean_dmca4[person_i][range_slice], 1)[0]
    )  # 回帰係数(polyfitは傾き[0]と切片[1]を返す)
    slope2_each_person.append(
        np.polyfit(np.log10(s[range_slice]), log10F2_band_mean_dmca4[person_i][range_slice], 1)[0]
    )  # 回帰係数(polyfitは傾き[0]と切片[1]を返す)

# 平均と標準偏差を求める
slope1_mean_person = np.mean(slope1_each_person)
slope1_sd_person = np.std(slope1_each_person)
slope2_mean_person = np.mean(slope2_each_person)
slope2_sd_person = np.std(slope2_each_person)
print(f"Slope1の平均値:     {slope1_mean_person}")
print(f"Slope1の標準偏差:   {slope1_sd_person}")
print(f"Slope2の平均値:     {slope2_mean_person}")
print(f"Slope2の標準偏差:   {slope2_sd_person}")


# %% Slope1とSlope2(15人分)の相関係数の計算
print(f"脳波と{column_name_of_HRV_measure}\n")

# 相関係数を計算
correlation_matrix = np.corrcoef(slope1_each_person, slope2_each_person)
# 結果の相関係数を取得（上三角・下三角が同じ）
correlation_coefficient = correlation_matrix[0, 1]
print("相関係数:", correlation_coefficient)

print()  # 改行

# ピアソンの相関係数と p 値を計算
corr, p_value = pearsonr(slope1_each_person, slope2_each_person)
print("相関係数:", corr)
print("p値:", p_value)


# %% 独立(対応なしの)t検定 (Slopeの平均値の差を検定)
print(f"脳波と{column_name_of_HRV_measure}\n")

# 与えられたデータ
mean_A = slope1_mean_person  # パラメータAの平均
mean_B = slope2_mean_person  # パラメータBの平均
var_A = slope1_sd_person  # パラメータAの分散
var_B = slope2_sd_person  # パラメータBの分散
n_A = len(slope1_each_person)  # サンプル数
n_B = len(slope2_each_person)  # サンプル数
# 標準偏差を計算
std_A = np.sqrt(var_A)
std_B = np.sqrt(var_B)

# F検定（分散の等しさを検定）
print("F検定を行います．")
F_stat = var_A / var_B  # F値の計算
df1 = n_A - 1  # 自由度1
df2 = n_B - 1  # 自由度2

# p値の計算（片側検定）
p_value_F = stats.f.cdf(F_stat, df1, df2) * 2  # 両側検定のため2倍する
# 結果を出力
print(f"F値: {F_stat}")
print(f"p値: {p_value_F}")

if p_value_F > 0.05:
    print(
        "p値が0.05よりも大きいため，データAとデータBの分散の差は統計的に優位ではないと考えられます．\nつまり，分散が等しいとみなすことができます．\n"
    )


# ウェルチのt検定（独立t検定・等分散を仮定しない）
t_stat, p_value = stats.ttest_ind_from_stats(
    mean1=mean_A,
    std1=std_A,
    nobs1=n_A,
    mean2=mean_B,
    std2=std_B,
    nobs2=n_B,
    equal_var=False,  # 等分散を仮定しない
)
# 結果を出力
print("ウェルチのt検定（独立t検定・等分散を仮定しない・正規性を仮定）の結果: ")
print(f"t値: {t_stat}")
print(f"p値: {p_value}")
if p_value_F > 0.05:
    print("p値が0.05よりも大きいため，統計的に有意な差はないと考えられます．\nつまり，2つの平均値には有意な差があるとは言えません．\n")

print()  # 改行

# スチューデントのt検定（等分散を仮定）
t_stat_student, p_value_student = stats.ttest_ind_from_stats(
    mean1=mean_A,
    std1=std_A,
    nobs1=n_A,
    mean2=mean_B,
    std2=std_B,
    nobs2=n_B,
    equal_var=True,  # 等分散を仮定
)
# 結果を出力
print("スチューデントのt検定（独立t検定・等分散を仮定・正規性を仮定）の結果: ")
print(f"t値: {t_stat_student}")
print(f"p値: {p_value_student}")
if p_value_F > 0.05:
    print("p値が0.05よりも大きいため，統計的に有意な差はないと考えられます．\nつまり，2つの平均値には有意な差があるとは言えません．\n")


# %% 対応ありのt検定
print(f"脳波と{column_name_of_HRV_measure}\n")

# 有意水準
alpha = 0.05

# t検定を実行
t_statistic, p_value = stats.ttest_rel(slope1_each_person, slope2_each_person)

print("t統計量:", t_statistic)
print("p値:", p_value)

# 判定
if p_value < alpha:
    print("帰無仮説を棄却（統計的に有意）")
else:
    print("帰無仮説を採択（統計的に有意ではない）")


# %% データの正規性を検定
print(f"脳波と{column_name_of_HRV_measure}\n")


# データが正規分布に従うかどうかを コルモゴロフ・スミルノフ検定 (Kolmogorov-Smirnov test) と シャピロ・ウィルク検定 (Shapiro-Wilk test) で調べる関数
def check_normality(data, alpha=0.05):
    """
    与えられたデータが正規分布に従うかどうかをコルモゴロフ・スミルノフ検定とシャピロ・ウィルク検定で調べる関数
    :param data: 検定するデータ（1次元配列）
    :param alpha: 有意水準（デフォルトは0.05）
    :return: 検定結果
    """
    print("=== データの正規性検定 ===")

    # コルモゴロフ・スミルノフ検定
    ks_stat, ks_p = stats.kstest(data, "norm")
    print(f"Kolmogorov-Smirnov Test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    if ks_p > alpha:
        print("Kolmogorov-Smirnov Test: データは正規分布に従う (p > 0.05)")
    else:
        print("Kolmogorov-Smirnov Test: データは正規分布に従わない (p <= 0.05)")

    # シャピロ・ウィルク検定
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    if shapiro_p > alpha:
        print("Shapiro-Wilk Test: データは正規分布に従う (p > 0.05)")
    else:
        print("Shapiro-Wilk Test: データは正規分布に従わない (p <= 0.05)")


# Slope1の検定
print("\nSlope1の検定:")
check_normality(slope1_each_person)

# 正規分布に従わないデータの検定
print("\nSlope2の検定:")
check_normality(slope2_each_person)

# QQプロットもしてみる(プロットが45度の直線に沿えば正規性あり)
print("\nQQプロットもしてみる")
# QQプロットの作成
stats.probplot(slope1_each_person, dist="norm", plot=plt)  # 正規分布を指定
# stats.probplot(slope1_each_person, dist=stats.expon, plot=plt)  # 指数分布を指定
# stats.probplot(slope1_each_person, dist=stats.lognorm, sparams=(1,), plot=plt)  # 対数正規分布を指定
# stats.probplot(slope1_each_person, dist=stats.gamma, sparams=(2,), plot=plt)  # ガンマ分布を指定
plt.title("QQ Plot (Slope1)")
plt.show()
# QQプロットの作成
stats.probplot(slope2_each_person, dist="norm", plot=plt)  # 正規分布を指定
# stats.probplot(slope2_each_person, dist=stats.expon, plot=plt)  # 指数分布を指定
# stats.probplot(slope2_each_person, dist=stats.lognorm, sparams=(1,), plot=plt)  # 対数正規分布を指定
plt.title("QQ Plot (Slope2)")
plt.show()


# %% ノンパラメトリック検定 (Slopeの中央値の差を検定)
print(f"脳波と{column_name_of_HRV_measure}\n")


def compare_means_nonparametric(data_A, data_B, paired=False):
    """
    正規性が仮定できない場合に、データAとデータBの平均値を比較する関数。

    パラメータ:
    data_A: list or array, データAの値
    data_B: list or array, データBの値
    paired: bool, Trueならウィルコクソンの符号付き順位検定（対応あり）、
                  Falseならマン・ホイットニーU検定（独立群）

    出力:
    統計量とp値
    """
    if paired:
        # ウィルコクソンの符号付き順位検定（対応あり）
        stat, p_value = stats.wilcoxon(data_A, data_B)
    else:
        # マン・ホイットニーU検定（独立群）
        stat, p_value = stats.mannwhitneyu(data_A, data_B, alternative="two-sided")

    return stat, p_value


# 使用例（データは適当に設定）
data_A = slope1_each_person
data_B = slope2_each_person

# 独立した2群の比較（マン・ホイットニーU検定）
stat_u, p_value_u = compare_means_nonparametric(data_A, data_B, paired=False)
print(f"Mann-Whitney U 検定: U値 = {stat_u:.3f}, p値 = {p_value_u}")

# 対応のあるデータの比較（ウィルコクソン符号付き順位検定）
stat_w, p_value_w = compare_means_nonparametric(data_A, data_B, paired=True)
print(f"Wilcoxon 検定: W値 = {stat_w}, p値 = {p_value_w}")

# %% 【実行する】すべてのファイルにおける相関係数とゆらぎ関数の平均を全ての脳波でプロット(次数は指定する)
# プロットする範囲をsliceオブジェクトにする
range_slice = slice(1, len(s))
print(f"len(s): {len(s)}")

# DMCAの次数(0, 2, 4)
# order = 2
fs_title = 40
fs_label = 40
fs_ticks = 30
fs_legend = 30

# labels = [
#     "Delta",
#     "Theta",
#     "Alpha",
#     "Beta",
#     "Gamma",
#     "Sigma",
# ]

# ラベルを付ける
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]

for order in orders:
    fig, axs = plt.subplots(2, 5, figsize=(30, 14))
    fig.suptitle(
        f"Mean XCorr and FFunc of DMCA{order} to Brain Waves and {column_name_of_HRV_measure}  {f'(Stage: {select_sleep_stage})' if select_sleep_stage != '' else ''} {f'(Stage: {remove_sleep_stage}{remove_sleep_stage_2}_removed)' if remove_sleep_stage != '' else ''}",
        fontsize=fs_title,
        y=0.935,
    )

    for band_ind, eeg_band in enumerate(eeg_bands[:5]):  # Sigmaは除く
        # 4次の結果(rho_mean[label_ind][2])のみを表示
        # order // 2 の処理 → 0 // 2 = 0,  2 // 2 = 1,  4 // 2 = 2
        log10F1_mean_dmca4 = log10F1_mean[band_ind][order // 2][range_slice]
        log10F2_mean_dmca4 = log10F2_mean[band_ind][order // 2][range_slice]
        log10F12_mean_dmca4 = log10F12_mean[band_ind][order // 2][range_slice]
        rho_mean_dmca4 = rho_mean[band_ind][order // 2][range_slice]  # 最初のやつはハズレ値っぽいから除外

        # [label_ind(li)] を [label_ind/3, label_ind%3]

        # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

        # rhoの平均をプロット
        axs[0, band_ind].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
        # rhoの平均の積分時系列をプロットしたい場合
        # axs[0, label_ind].plot(np.log10(s[1:]), np.cumsum(rho_mean_dmca4), color="red", linestyle="None", marker="x", ms=10)
        # axs[label_ind//3, label_ind%3].set_xlim(0.612110372200782, 2.523022279175993)
        axs[0, band_ind].set_ylim(-1, 1)
        axs[0, band_ind].axhline(0, linestyle="--", color="gray")
        axs[0, band_ind].set_title(f"{eeg_band} Ratio", fontsize=fs_title)
        # axs[0, label_ind].set_xlabel("log10(s)", fontsize=14)
        if band_ind == 0:
            axs[0, band_ind].set_ylabel(r"$\rho$", fontsize=fs_label + 10)
        # y軸ラベルはlabel_indが0の場合のみ表示
        axs[0, band_ind].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2, labelbottom=False, labelleft=(band_ind == 0))

        axs[0, band_ind].legend(
            title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
            title_fontsize=fs_legend,
        )
        # サブキャプション
        axs[0, band_ind].text(
            0.02,  # x座標
            0.95,  # y座標
            # -0.1,
            # 1.1,
            labels[band_ind],
            transform=axs[0, band_ind].transAxes,  # 相対座標に変換
            fontsize=fs_label,
            fontweight="bold",
            va="top",
            ha="left",
        )

        coeff1_mean = np.polyfit(np.log10(s[range_slice]), log10F1_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        coeff2_mean = np.polyfit(np.log10(s[range_slice]), log10F2_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        coeff12_mean = np.polyfit(np.log10(s[range_slice]), log10F12_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
        fitted1_mean = np.poly1d(coeff1_mean)  # 回帰直線の式
        fitted2_mean = np.poly1d(coeff2_mean)  # 回帰直線の式
        fitted12_mean = np.poly1d(coeff12_mean)  # 回帰直線の式

        axs[1, band_ind].scatter(np.log10(s[range_slice]), log10F1_mean_dmca4, color="green", label="$F_1$", marker="^", facecolors="none", s=75)
        axs[1, band_ind].scatter(np.log10(s[range_slice]), log10F2_mean_dmca4, color="blue", label="$F_2$", marker="s", facecolors="none", s=75)
        axs[1, band_ind].scatter(np.log10(s[range_slice]), log10F12_mean_dmca4, color="red", label="$F_{12}$", marker="x", s=75)
        # axs[1, label_ind].plot(np.log10(s), fitted1_mean(np.log10(s)), color="green", linestyle="--")
        # axs[1, label_ind].plot(np.log10(s), fitted2_mean(np.log10(s)), color="blue", linestyle="--")
        axs[1, band_ind].plot(np.log10(s[range_slice]), fitted12_mean(np.log10(s[range_slice])), color="red", linestyle="--")
        # y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
        # y_max = max(log10F1.max(), log10F2.max(), log10F12.max())
        # axs[1, label_ind].set_ylim(y_min, y_max)
        # axs[1, label_ind].set_title(
        #     # f"DMCA{order}\nSlope1 = {coeff1_mean[0]:.3f},  Slope2 = {coeff2_mean[0]:.3f},  Slope12 = {coeff12_mean[0]:.3f}",
        #     f"Slope12 = {coeff12_mean[0]:.3f}",
        #     fontsize=fs_title,
        # )
        axs[1, band_ind].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_title)
        if band_ind == 0:
            axs[1, band_ind].set_ylabel(r"$\log_{10}F_{12}(s)$", fontsize=fs_title)
        # y軸ラベルはlabel_indが0の場合のみ表示
        axs[1, band_ind].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2, labelleft=(band_ind == 0))

        axs[1, band_ind].legend(
            fontsize=fs_legend,
            labelspacing=0.3,  # ラベル間の縦のスペースを調整
            handlelength=1,  # 凡例内の線（ハンドル）の長さを調整
            handletextpad=0.1,  # 線とテキスト間のスペースを調整
            borderpad=0.2,  # 凡例全体の内側の余白
        )

        # サブキャプション
        axs[1, band_ind].text(
            0.02,  # x座標
            0.95,  # y座標
            # -0.1,
            # 1.1,
            labels[band_ind + 5],
            transform=axs[1, band_ind].transAxes,  # 相対座標に変換
            fontsize=fs_label,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # 傾きを回帰直線の近くに表示
        # x_comment = 1.05
        # y_comment = coeff12_mean[0] * x_comment + coeff12_mean[1] - 0.35
        # axs[1, band_ind].text(x_comment, y_comment, f"Slope = {coeff12_mean[0]:.3f}", fontsize=30, color="black", va="bottom")
        axs[1, band_ind].text(
            0.2,
            0.025,
            f"Slope: {coeff12_mean[0]:.3f}",
            transform=axs[1, band_ind].transAxes,  # 相対座標に変換
            fontsize=30,
            color="black",
            va="bottom",
        )  # 回帰直線との相対的な位置に設定したらグラフごとに微妙なズレが生じる

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

    # グラフの保存と表示
    if is_savefig:
        # グラフの保存
        os.chdir(script_dir)
        DIR_OUT = "../../../results/mean/"
        if not os.path.exists(DIR_OUT):
            os.makedirs(DIR_OUT)
        os.chdir(DIR_OUT)
        # 'LF/HF' はエラーになるから変換
        measure_name = column_name_of_HRV_measure if column_name_of_HRV_measure != "LF/HF" else "LFHF"
        # ステージ名の追加（空でない場合）
        selected_stage_suffix = f"_{select_sleep_stage}" if select_sleep_stage != "" else ""
        removed_stage_suffix = f"_{remove_sleep_stage}" if remove_sleep_stage != "" else ""
        plt.savefig(
            # f"EEG_{column_index_of_HRV_measure}_{f'{column_name_of_HRV_measure}' if column_name_of_HRV_measure != 'LF/HF' else 'LFHF'}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}_DMCA{order}"
            f"EEG_{column_index_of_HRV_measure}_{measure_name}" + selected_stage_suffix + f"{removed_stage_suffix}_removed" + f"_DMCA{order}.png",
            dpi=300,
            bbox_inches="tight",
        )
    # グラフの表示
    plt.show()


# %% 【実行する】アブスト用のグラフその1(DeltaとGammaの0次DMCA)
order = 4  # 次数を指定
# プロットしたいバンドを指定(Delta:0,Theta:1, Alpha:2, Beta:3, Gamma:4)
band_inds = [0, 4]
eeg_bands_selected = ["Delta", "Gamma"]

fs_title = 40
fs_label = 40
fs_ticks = 25
fs_legend = 30

labels = ["(a)", "(b)", "(c)", "(d)"]

bands = [r"$\delta$", r"$\gamma$"]

fig, axs = plt.subplots(1, 4, figsize=(30, 7.5))
# fig.suptitle(
#     f"Mean XCorr and FFunc of DMCA{order} to Brain Waves and {column_name_of_HRV_measure}  {f'(Stage: {select_sleep_stage})' if select_sleep_stage != '' else ''}",
#     fontsize=fs_title,
#     y=0.935,
# )
for i, (band_ind, eeg_band) in enumerate(zip(band_inds, eeg_bands_selected)):
    # 4次の結果(rho_mean[label_ind][2])のみを表示
    # order // 2 の処理 → 0 // 2 = 0,  2 // 2 = 1,  4 // 2 = 2
    log10F1_mean_dmca4 = log10F1_mean[band_ind][order // 2][range_slice]
    log10F2_mean_dmca4 = log10F2_mean[band_ind][order // 2][range_slice]
    log10F12_mean_dmca4 = log10F12_mean[band_ind][order // 2][range_slice]
    rho_mean_dmca4 = rho_mean[band_ind][order // 2][range_slice]  # 最初のやつはハズレ値っぽいから除外

    # [label_ind(li)] を [label_ind/3, label_ind%3]

    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

    # rhoの平均をプロット
    axs[i * 2].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
    # rhoの平均の積分時系列をプロットしたい場合
    # axs[i * 2].plot(np.log10(s[1:]), np.cumsum(rho_mean_dmca4), color="red", linestyle="None", marker="x", ms=10)
    # axs[label_ind//3, label_ind%3].set_xlim(0.612110372200782, 2.523022279175993)
    axs[i * 2].set_ylim(-1, 1)
    axs[i * 2].axhline(0, linestyle="--", color="gray")
    axs[i * 2].set_title(f"{bands[i]} ratio vs. {column_name_of_HRV_measure}", fontsize=fs_title)
    axs[i * 2].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
    axs[i * 2].set_ylabel(r"$\rho$", fontsize=fs_label)
    # y軸ラベルはband_indが0の場合のみ表示
    axs[i * 2].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2)
    axs[i * 2].legend(
        title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
        title_fontsize=fs_legend,
    )
    # サブキャプション
    axs[i * 2].text(
        0.02,  # x座標
        0.95,  # y座標
        # -0.1,
        # 1.1,
        labels[i * 2],
        transform=axs[i * 2].transAxes,  # 相対座標に変換
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )

    coeff1_mean = np.polyfit(np.log10(s[range_slice]), log10F1_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
    coeff2_mean = np.polyfit(np.log10(s[range_slice]), log10F2_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
    coeff12_mean = np.polyfit(np.log10(s[range_slice]), log10F12_mean_dmca4, 1)  # 回帰係数(polyfitは傾きと切片を返す)
    fitted1_mean = np.poly1d(coeff1_mean)  # 回帰直線の式
    fitted2_mean = np.poly1d(coeff2_mean)  # 回帰直線の式
    fitted12_mean = np.poly1d(coeff12_mean)  # 回帰直線の式

    axs[i * 2 + 1].scatter(np.log10(s[range_slice]), log10F1_mean_dmca4, color="green", label="$F_1$", marker="^", facecolors="none", s=75)
    axs[i * 2 + 1].scatter(np.log10(s[range_slice]), log10F2_mean_dmca4, color="blue", label="$F_2$", marker="s", facecolors="none", s=75)
    axs[i * 2 + 1].scatter(np.log10(s[range_slice]), log10F12_mean_dmca4, color="red", label="$F_{12}$", marker="x", s=75)
    # axs[i * 2 + 1].plot(np.log10(s), fitted1_mean(np.log10(s)), color="green", linestyle="--")
    # axs[i * 2 + 1].plot(np.log10(s), fitted2_mean(np.log10(s)), color="blue", linestyle="--")
    axs[i * 2 + 1].plot(np.log10(s[range_slice]), fitted12_mean(np.log10(s[range_slice])), color="red", linestyle="--")
    # y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
    # y_max = max(log10F1.max(), log10F2.max(), log10F12.max())
    # axs[i * 2 + 1].set_ylim(y_min, y_max)
    # axs[i * 2 + 1].set_title(
    #     # f"DMCA{order}\nSlope1 = {coeff1_mean[0]:.3f},  Slope2 = {coeff2_mean[0]:.3f},  Slope12 = {coeff12_mean[0]:.3f}",
    #     f"Slope12 = {coeff12_mean[0]:.3f}",
    #     fontsize=fs_title,
    # )
    axs[i * 2 + 1].set_title(f"{bands[i]} ratio vs. {column_name_of_HRV_measure}", fontsize=fs_title)
    axs[i * 2 + 1].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
    axs[i * 2 + 1].set_ylabel(r"$\log_{10}F_{12}(s)$", fontsize=fs_label)
    # y軸ラベルはband_indが0の場合のみ表示
    axs[i * 2 + 1].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2)
    axs[i * 2 + 1].legend(
        fontsize=fs_legend,
        labelspacing=0.3,  # ラベル間の縦のスペースを調整
        handlelength=1,  # 凡例内の線（ハンドル）の長さを調整
        handletextpad=0.1,  # 線とテキスト間のスペースを調整
        borderpad=0.2,  # 凡例全体の内側の余白
    )
    # サブキャプション
    axs[i * 2 + 1].text(
        0.02,  # x座標
        0.95,  # y座標
        # -0.1,
        # 1.1,
        labels[i * 2 + 1],
        transform=axs[i * 2 + 1].transAxes,  # 相対座標に変換
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )
    # 傾きを回帰直線の近くに表示
    # x_comment = 1.4
    # y_comment = coeff12_mean[0] * x_comment + coeff12_mean[1] - 0.35
    # axs[i * 2 + 1].text(x_comment, y_comment, f"Slope = {coeff12_mean[0]:.3f}", fontsize=30, color="black", va="bottom")
    axs[i * 2 + 1].text(
        0.2,
        0.025,
        f"Slope: {coeff12_mean[0]:.3f}",
        transform=axs[i * 2 + 1].transAxes,  # 相対座標に変換
        fontsize=30,
        color="black",
        va="bottom",
    )  # 回帰直線との相対的な位置に設定したらグラフごとに微妙なズレが生じる

plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

# グラフの保存と表示
# if is_savefig:
#     # グラフの保存
#     os.chdir(script_dir)
#     DIR_OUT = "../../../results/mean/"
#     if not os.path.exists(DIR_OUT):
#         os.makedirs(DIR_OUT)
#     os.chdir(DIR_OUT)
#     # 'LF/HF' はエラーになるから変換
#     measure_name = column_name_of_HRV_measure if column_name_of_HRV_measure != "LF/HF" else "LFHF"
#     # ステージ名の追加（空でない場合）
#     stage_suffix = f"_{select_sleep_stage}" if select_sleep_stage != "" else ""
#     plt.savefig(
#         # f"EEG_{column_index_of_HRV_measure}_{f'{column_name_of_HRV_measure}' if column_name_of_HRV_measure != 'LF/HF' else 'LFHF'}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}_DMCA{order}"
#         f"EEG_{column_index_of_HRV_measure}_{measure_name}" + stage_suffix + f"_DMCA{order}.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
# グラフの表示
plt.show()


# %% 【一応実行する】アブスト用のグラフその2(生データとデルタ波の解析結果)
order = 4  # 次数を指定


# 任意のプロット用関数（例: ユーザーが提供する関数をここで受け取る）
def custom_plot_func(ax, plot_index):
    if plot_index == 0:
        # x = np.linspace(0, 10, 100)
        # y = np.sin(x)
        # ax.plot(x, y, label="Sine Wave", color="blue")
        # 脳波の生データをプロット
        ax.plot(range(n), x1, color="green")
        ax.set_title(r"$\delta$ ratio", fontsize=fs_title)
        ax.set_xlabel("i", fontsize=fs_label)
        ax.set_ylabel(r"$\delta$ ratio", fontsize=fs_label)

    elif plot_index == 1:
        # x = np.linspace(0, 10, 100)
        # y = np.cos(x)
        # ax.plot(x, y, label="Cosine Wave", color="green")
        # 心拍の生データをプロット
        ax.plot(range(n), x2, color="blue")
        ax.set_title(column_name_of_HRV_measure, fontsize=fs_title)
        ax.set_xlabel("i", fontsize=fs_label)
        ax.set_ylabel(f"{column_name_of_HRV_measure} [ms]", fontsize=fs_label)
    # ax.set_title(f"Custom Plot {plot_index+1}", fontsize=fs_title)
    # ax.legend(fontsize=fs_legend)
    # ax.set_xlabel("X-axis", fontsize=fs_label)
    # ax.set_ylabel("Y-axis", fontsize=fs_label)
    ax.tick_params(axis="both", which="both", labelsize=fs_ticks)


# グラフの作成
fig, axs = plt.subplots(1, 4, figsize=(30, 7.5))

# 左から1番目と2番目に任意のグラフをプロット
for i in range(2):
    custom_plot_func(axs[i], i)
    axs[i].text(
        0.02,
        0.95,
        labels[i],
        transform=axs[i].transAxes,
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )

bands = [r"$\delta$", r"$\gamma$"]

# 元のコードのグラフを左から3番目と4番目に移動([:1]でDelta波だけに限定)
for i, (band_ind, eeg_band) in enumerate(zip(band_inds[:1], eeg_bands_selected[:1])):
    log10F1_mean_dmca4 = log10F1_mean[band_ind][order // 2][range_slice]
    log10F2_mean_dmca4 = log10F2_mean[band_ind][order // 2][range_slice]
    log10F12_mean_dmca4 = log10F12_mean[band_ind][order // 2][range_slice]
    rho_mean_dmca4 = rho_mean[band_ind][order // 2][range_slice]

    # 移動したインデックスを調整
    axs[2 + i * 2].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
    axs[2 + i * 2].set_ylim(-1, 1)
    axs[2 + i * 2].axhline(0, linestyle="--", color="gray")
    axs[2 + i * 2].set_title(f"{bands[band_ind]} ratio vs. {column_name_of_HRV_measure}", fontsize=fs_title)
    axs[2 + i * 2].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
    axs[2 + i * 2].set_ylabel(r"$\rho$", fontsize=fs_label)
    axs[2 + i * 2].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2)
    axs[2 + i * 2].legend(
        title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
        title_fontsize=fs_legend,
    )
    axs[2 + i * 2].text(
        0.02,
        0.95,
        labels[2 + i * 2],
        transform=axs[2 + i * 2].transAxes,
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )

    coeff12_mean = np.polyfit(np.log10(s[range_slice]), log10F12_mean_dmca4, 1)
    fitted12_mean = np.poly1d(coeff12_mean)

    axs[2 + i * 2 + 1].scatter(np.log10(s[range_slice]), log10F1_mean_dmca4, color="green", label="$F_1$", marker="^", facecolors="none", s=75)
    axs[2 + i * 2 + 1].scatter(np.log10(s[range_slice]), log10F2_mean_dmca4, color="blue", label="$F_2$", marker="s", facecolors="none", s=75)
    axs[2 + i * 2 + 1].scatter(np.log10(s[range_slice]), log10F12_mean_dmca4, color="red", label="$F_{12}$", marker="x", s=75)
    axs[2 + i * 2 + 1].plot(np.log10(s[range_slice]), fitted12_mean(np.log10(s[range_slice])), color="red", linestyle="--")
    axs[2 + i * 2 + 1].set_title(f"{bands[band_ind]} ratio vs. {column_name_of_HRV_measure}", fontsize=fs_title)
    axs[2 + i * 2 + 1].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
    axs[2 + i * 2 + 1].set_ylabel(r"$\log_{10}F_{12}(s)$", fontsize=fs_label)
    axs[2 + i * 2 + 1].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2)
    axs[2 + i * 2 + 1].legend(fontsize=fs_legend)
    axs[2 + i * 2 + 1].text(
        0.02,
        0.95,
        labels[2 + i * 2 + 1],
        transform=axs[2 + i * 2 + 1].transAxes,
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )
    axs[2 + i * 2 + 1].legend(
        fontsize=fs_legend,
        labelspacing=0.3,  # ラベル間の縦のスペースを調整
        handlelength=1,  # 凡例内の線（ハンドル）の長さを調整
        handletextpad=0.1,  # 線とテキスト間のスペースを調整
        borderpad=0.2,  # 凡例全体の内側の余白
    )
    axs[2 + i * 2 + 1].text(
        0.2,
        0.025,
        f"Slope: {coeff12_mean[0]:.3f}",
        transform=axs[2 + i * 2 + 1].transAxes,  # 相対座標に変換
        fontsize=30,
        color="black",
        va="bottom",
    )  # 回帰直線との相対的な位置に設定したらグラフごとに微妙なズレが生じる

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %% アブスト用のグラフその3(生データ)
fs_title = 35
fs_label = 40
fs_ticks = 25
fs_legend = 30

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]

# プロットしたい生データを指定
# [3:4]は19E自宅,[11:12]は19O自宅，[12:13]は20A自宅1，[19:20]は20I自宅2，[29:30]は20P自宅2
file_name = all_combined_files[29]

# ファイルの読み込み
os.chdir(script_dir)
os.chdir(DIR_EEG)  # ディレクトリの移動
with open(file_name, "rb") as file:
    # ファイルのエンコーディングを検出
    detected_encoding = detect(file.read())["encoding"]
# 正しいエンコーディングでファイルを読み込む
data = pd.read_csv(file_name, encoding=detected_encoding)

fig, axs = plt.subplots(1, 6, figsize=(36, 7))

bands = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
# 脳波の生データをプロット
for band_i, band in enumerate(bands):
    x1 = data.iloc[:, 9 + band_i].values  # Deltaは +0 でおｋ
    n = len(x1)
    axs[band_i].plot(range(n), x1, color="green")
    axs[band_i].set_ylim(0, 1)
    axs[band_i].set_title(f"{band} waves", fontsize=fs_title)
    axs[band_i].set_xlabel("i", fontsize=fs_label)
    if band_i == 0:
        axs[band_i].set_ylabel("Relative Power", fontsize=fs_label)
    axs[band_i].tick_params(axis="both", which="both", labelsize=fs_ticks, length=15, width=2)
    axs[band_i].text(
        0.02,
        0.95,
        labels[band_i],
        transform=axs[band_i].transAxes,
        fontsize=fs_label,
        fontweight="bold",
        va="top",
        ha="left",
    )

# 心拍の生データをプロット
x2 = data.iloc[:, column_index_of_HRV_measure].values
column_name_of_HRV_measure = data.columns[column_index_of_HRV_measure]
axs[5].plot(range(n), x2, color="blue")
axs[5].set_title(column_name_of_HRV_measure, fontsize=fs_title)
axs[5].set_xlabel("i", fontsize=fs_label)
axs[5].set_ylabel(f"{column_name_of_HRV_measure} [ms]", fontsize=fs_label)
axs[5].tick_params(axis="both", which="both", labelsize=fs_ticks)
axs[5].text(
    0.02,
    0.95,
    labels[5],
    transform=axs[5].transAxes,
    fontsize=fs_label,
    fontweight="bold",
    va="top",
    ha="left",
)

# レイアウト調整と表示
plt.tight_layout()
plt.show()

# %% アブスト用のグラフその4(2つの心拍変動指標についてバンドの平均SlopeとXCorr)
order = 4  # 次数を指定

fs_title = 35
fs_label = 40
fs_ticks = 25
fs_legend = 30

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# プロットする範囲をsliceオブジェクトにする
range_slice = slice(1, len(s))
print(f"len(s): {len(s)}")

# プロット設定
# fig, axs = plt.subplots(1, 2, figsize=(18, 8))  # 横並びで2つのプロット
# fig.suptitle("Combined Plots for EEG Bands and F Functions", fontsize=20)

bands = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"]

# 1つ目のグラフ：0行0～4列を統合
# プロット用のcolorとlinestyleをリストで定義
colors = ["red", "blue", "green", "#CC5500", "purple"]
linestyles = ["-", "--", "-.", ":", (0, (5, 3))]  # 必要に応じて増やす
for band_ind, eeg_band in enumerate(eeg_bands[:5]):
    rho_mean_dmca4 = rho_mean[band_ind][order // 2][range_slice]  # データ取得
    # 各プロットに異なるlinestyleを指定
    axs[0].plot(
        np.log10(s[range_slice]),
        rho_mean_dmca4,
        label=f"{bands[band_ind]} ratio",
        color=colors[band_ind],
        linestyle=linestyles[band_ind % len(linestyles)],  # リストの範囲を超えないように
        lw=2.5,
    )

# グラフの装飾
# タイトルはなくて良い
# axs[0].set_title(f"XCorr of EEG vs. {column_name_of_HRV_measure}", fontsize=fs_title)
axs[0].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
axs[0].set_ylabel(r"$\rho$", fontsize=fs_label)
axs[0].set_ylim(-1, 1)
axs[0].axhline(0, linestyle="--", color="gray")
axs[0].legend(fontsize=20, loc="lower left")
axs[0].tick_params(axis="both", which="both", labelsize=fs_ticks)
axs[0].text(
    0.02,
    0.95,
    labels[0],
    transform=axs[0].transAxes,
    fontsize=fs_label,
    fontweight="bold",
    va="top",
    ha="left",
)

# 2つ目のグラフ：log10F1_mean_dmca4 と log10F2_mean_dmca4 を色分けしてプロット
axs[1].scatter(np.log10(s[range_slice]), log10F1_mean_dmca4, label=r"$F_1$", color="green", marker="^", facecolors="none", s=75)
axs[1].scatter(np.log10(s[range_slice]), log10F2_mean_dmca4, label=r"$F_2$", color="blue", marker="s", facecolors="none", s=75)
# 新しい直線の式を生成
# coeff1_mean_modified = [coeff1_mean[0], coeff1_mean[1]]
# fitted1_mean_modified = np.poly1d(coeff1_mean_modified)
axs[1].plot(np.log10(s[8:20]), fitted1_mean(np.log10(s[8:20])) - 0.2, color="green", linestyle=(0, (5, 3)), lw=3)
# fitted2_mean_modified = np.poly1d([fitted2_mean[1], fitted2_mean[0]])
axs[1].plot(np.log10(s[8:20]), fitted2_mean(np.log10(s[8:20])) + 0.1, color="blue", linestyle=(0, (5, 3)), lw=3)

axs[1].set_ylim(-0.95, 0.95)
# タイトルはなくて良い
# axs[1].set_title(f"EEG and {column_name_of_HRV_measure}", fontsize=fs_title)  # タイトルを空白に設定
# axs[1].set_title("    and           ", fontsize=fs_title)  # タイトルを空白に設定
# axs[1].text(
#     0.145,
#     1,
#     "EEG",
#     transform=axs[1].transAxes,  # 相対座標に変換
#     fontsize=fs_title,
#     color="green",
#     va="bottom",
# )
# axs[1].text(
#     0.515,
#     1,
#     f"{column_name_of_HRV_measure}",
#     transform=axs[1].transAxes,  # 相対座標に変換
#     fontsize=fs_title,
#     color="blue",
#     va="bottom",
# )

axs[1].set_xlabel(r"$\log_{10}(s)$", fontsize=fs_label)
axs[1].set_ylabel(r"$\log_{10}F(s)$", fontsize=fs_label)
axs[1].legend(fontsize=fs_label)
axs[1].tick_params(axis="both", which="both", labelsize=fs_ticks)
axs[1].legend(
    fontsize=fs_legend,
    labelspacing=0.3,  # ラベル間の縦のスペースを調整
    handlelength=1,  # 凡例内の線（ハンドル）の長さを調整
    handletextpad=0.1,  # 線とテキスト間のスペースを調整
    borderpad=0.2,  # 凡例全体の内側の余白
)
axs[1].text(
    0.02,
    0.95,
    labels[1],
    transform=axs[1].transAxes,
    fontsize=fs_label,
    fontweight="bold",
    va="top",
    ha="left",
)
# 脳波のバンドにおけるSlopeの平均値
band_slope = []
for band_ind, eeg_band in enumerate(eeg_bands):
    band_log10F1_mean = log10F1_mean[band_ind]
    band_coeff1_mean = np.polyfit(np.log10(s[range_slice]), band_log10F1_mean[order // 2][range_slice], 1)
    band_fitted1_mean = np.poly1d((band_coeff1_mean))
    band_slope.append(band_coeff1_mean[0])
# band_log10F1_mean を NumPy 配列に変換
band_slope = np.array(band_slope)
band_slope_mean = np.mean(band_slope)
band_slope_std = np.std(band_slope)
axs[1].text(
    0.37,
    0.15,
    rf"{band_slope_mean:.3f}$\pm${band_slope_std:.3f}",
    transform=axs[1].transAxes,  # 相対座標に変換
    fontsize=25,
    color="green",
    va="bottom",
    rotation=np.degrees(np.arctan(fitted1_mean[1])),
    rotation_mode="anchor",
)
axs[1].text(
    0.35,
    0.4,
    rf"{coeff2_mean[0]:.3f}",
    transform=axs[1].transAxes,  # 相対座標に変換
    fontsize=25,
    color="blue",
    va="bottom",
    rotation=np.degrees(np.arctan(fitted2_mean[1])),
    rotation_mode="anchor",
)

# レイアウト調整と表示
plt.tight_layout()
plt.show()


# %%
# 先にスロープを求めてその平均と標準偏差を求める
print(log10F1_mean.shape)
band_slope = []
for band_ind, eeg_band in enumerate(eeg_bands):
    band_log10F1_mean = log10F1_mean[band_ind]
    band_coeff1_mean = np.polyfit(np.log10(s[range_slice]), band_log10F1_mean[order // 2][range_slice], 1)
    band_fitted1_mean = np.poly1d((band_coeff1_mean))
    band_slope.append(band_coeff1_mean[0])

# band_log10F1_mean を NumPy 配列に変換
band_slope = np.array(band_slope)
# shape を表示
print(band_slope.shape)
band_slope_mean = np.mean(band_slope)
band_slope_std = np.std(band_slope)
print(band_slope_mean)
print(band_slope_std)

# %%
print(log10F1_mean_dmca4.shape)
print(np.mean(log10F1_mean_dmca4))

# %% DMCA(4次)の相関係数の平均値をすべての脳波でプロット
# fig, axs = plt.subplots(2, 3, figsize=(20, 14))
# fig.suptitle(
#     f"Mean XCorr of DMCA4 to Brain Waves and {column_name_of_HRV_measure}  {f'(Stage: {select_sleep_stage})' if select_sleep_stage != '' else ''}",
#     fontsize=18,
#     y=0.935,
# )

# for label_ind, label in enumerate(labels[:5]):
#     # 4次の結果(rho_maen[label_ind][2])のみを表示
#     rho_mean_dmca4 = rho_mean[label_ind][2][1 : len(s)]  # 最初のやつはハズレ値っぽいから除外

#     # [label_ind(li)] を [label_ind/3, label_ind%3]

#     # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

#     axs[label_ind // 3, label_ind % 3].plot(np.log10(s[1:]), rho_mean_dmca4, color="red", linestyle="None", marker="x", ms=10)
#     # axs[label_ind//3, label_ind%3].set_xlim(0.612110372200782, 2.523022279175993)
#     axs[label_ind // 3, label_ind % 3].set_ylim(-1, 1)
#     axs[label_ind // 3, label_ind % 3].axhline(0, linestyle="--", color="gray")
#     axs[label_ind // 3, label_ind % 3].set_title(f"{label} Ratio", fontsize=16)
#     axs[label_ind // 3, label_ind % 3].set_xlabel("log10(s)", fontsize=14)
#     axs[label_ind // 3, label_ind % 3].set_ylabel("rho", fontsize=14)
#     axs[label_ind // 3, label_ind % 3].legend(
#         title=f"Max:  {max(rho_mean_dmca4):.3f}\nMin:  {min(rho_mean_dmca4):.3f}",
#         title_fontsize=12,
#     )
# # 6つ目のサブプロットを空白に設定
# axs[1, 2].axis("off")  # 軸を非表示
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # グラフが重ならないようにレイアウト調整

# # グラフの保存と表示
# if is_savefig:
#     # グラフの保存
#     os.chdir(script_dir)
#     DIR_OUT = "../../../results/mean/"
#     if not os.path.exists(DIR_OUT):
#         os.makedirs(DIR_OUT)
#     os.chdir(DIR_OUT)  # 20YYXにディレクトリを移動
#     plt.savefig(
#         f"Mean_XCorrMean_{column_name_of_HRV_measure}{f'_{select_sleep_stage}' if select_sleep_stage != '' else ''}" + ".png",
#         dpi=300,
#         bbox_inches="tight",
#     )
# # グラフの表示
# plt.show()


# %% 相関係数の積分値およびSlopeの平均値を表として出力
# 正常なデータのみを抽出
rho_integrated_masked = rho_integrated[mask]
slopes1_masked = slopes1[mask]
slopes2_masked = slopes2[mask]
slopes12_masked = slopes12[mask]

# DataFrameに変換する際の行名と列名
row_names = eeg_bands
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


# %% 各ファイルの開始時刻、終了時刻、期間を計算する
# 異常値が含まれるファイルを除外する
# all_combined_files を NumPy 配列に変換してからマスクを適用
all_combined_files_array = np.array(all_combined_files)
all_combined_files_masked = all_combined_files_array[mask]

# 各ファイルの開始時刻、終了時刻、期間を格納するリストを初期化
start_times = []
end_times = []
periods = []

for file_ind, file_name in enumerate(all_combined_files_masked):
    # ファイルの読み込み
    os.chdir(script_dir)
    os.chdir(DIR_EEG)  # ディレクトリの移動
    # print(f"{file_name} の読み込み中...")

    with open(file_name, "rb") as file:
        # ファイルのエンコーディングを検出
        detected_encoding = detect(file.read())["encoding"]
    # print(f"検出されたファイルエンコーディング: {detected_encoding}")
    data = pd.read_csv(file_name, encoding=detected_encoding)

    # `Time` 列が存在するかをチェック
    if "Time" not in data.columns:
        print(f"ファイル {file_name} に 'Time' 列が存在しません。スキップします。\n")
        continue

    # `Time` 列を日時型に変換
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

    # 時刻データの有効な値を取得
    valid_times = data["Time"].dropna()

    # 開始時刻と終了時刻を取得
    if len(valid_times) > 0:
        start_time = valid_times.iloc[0]
        end_time = valid_times.iloc[-1]
        duration = end_time - start_time  # 期間を計算

        start_times.append(start_time)
        end_times.append(end_time)
        periods.append(duration)
    else:
        print(f"ファイル {file_name} の 'Time' 列に有効な時刻データが存在しません。スキップします。\n")

# 平均開始時刻、終了時刻、期間を計算
if len(periods) > 0:
    mean_start_time = pd.to_datetime(pd.Series(start_times)).mean().time()
    mean_end_time = pd.to_datetime(pd.Series(end_times)).mean().time()
    mean_duration = sum(periods, pd.Timedelta(0)) / len(periods)

    print("平均開始時刻:", mean_start_time)
    print("平均終了時刻:", mean_end_time)
    print("平均計測期間:", mean_duration)
else:
    print("有効な時刻データが存在しないため、平均値の計算はスキップされました。")

# %%
start_times

# %%
end_times

# %%
periods

# %% 計測機関の平均と標準偏差を求める
# Timedeltaリストを作成
periods = [
    pd.Timedelta("-1 days +07:17:30"),
    pd.Timedelta("-1 days +09:02:00"),
    pd.Timedelta("-1 days +09:00:00"),
    pd.Timedelta("0 days 05:07:30"),
    pd.Timedelta("0 days 08:45:00"),
    pd.Timedelta("0 days 08:26:00"),
    pd.Timedelta("0 days 08:56:30"),
    pd.Timedelta("-1 days +09:13:00"),
    pd.Timedelta("-1 days +09:07:30"),
    pd.Timedelta("-1 days +09:19:30"),
    pd.Timedelta("-1 days +09:24:00"),
    pd.Timedelta("-1 days +08:55:30"),
    pd.Timedelta("-1 days +08:50:30"),
    pd.Timedelta("0 days 09:18:00"),
    pd.Timedelta("0 days 09:02:00"),
]

# 時刻部分を秒単位に変換（1日の秒数を加算して負の値を補正）
seconds = [(td.total_seconds() + 86400) % 86400 for td in periods]

# 平均を計算
average_seconds = sum(seconds) / len(seconds)
# 標準偏差を計算
std_seconds = pd.Series(seconds).std()

# 平均秒数を時刻に変換
average_time = pd.to_timedelta(average_seconds, unit="s")
print(f"計測期間の平均：{average_time}")
# 08:38:58 → 8.6494 Hour

# 秒を時刻形式に変換
std_timedelta = pd.Timedelta(seconds=std_seconds)
print(f"計測期間の標準偏差：{std_timedelta}")
# 01:05:58.692424814 → 1.0996 Hour
