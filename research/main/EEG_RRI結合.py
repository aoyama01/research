# %%
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from icecream import ic

###
# ic.disable()  # icによるデバッグを無効化
###

# 実行中のスクリプトが存在するディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# カレントディレクトリをスクリプトのディレクトリに設定
os.chdir(script_dir)

# 確認
# print(f"現在の作業ディレクトリ: {os.getcwd()}")

# フォルダの指定
DIR_RRI = "../../../data/心拍変動まとめ_copy"
DIR_EEG = "../../../data/睡眠段階まとめ_copy"
DIR_OUT = "../../../data/睡眠段階まとめ_copy"

# ファイル名の指定
FN_RRI = "2019A自宅.csv"
FN_EEG = "2019A自宅睡眠段階.csv"
FN_OUT = "2019A自宅_EEG_RRI.csv"

# # %%

# # 任意のディレクトリ
# target_directory = (
#     "D:/OneDrive - OUMail (Osaka University)/B4_AW/GradRes/data/心拍変動まとめ_copy"
# )

# # 相対パスを計算
# relative_path = os.path.relpath(target_directory, os.getcwd())

# print(f"現在のディレクトリ: {os.getcwd()}")
# print(f"ターゲットディレクトリ: {target_directory}")
# print(f"相対パス: {relative_path}")


# ファイル読み込み
# if not os.path.exists(DIR_RRI):
#     raise FileNotFoundError(f"指定されたパスが見つかりません→→→: {DIR_RRI}")
# else:
#     os.chdir(DIR_RRI)
os.chdir(script_dir)
os.chdir(DIR_RRI)
# 計測開始日の取得
TMP_RRI = pd.read_csv(
    FN_RRI, nrows=4, encoding="shift-jis"
)  # ファイルの先頭4行を読み込み(1行目はヘッダーとして無視)
date_eeg = TMP_RRI.iloc[2][1]  # 3行目2列目に計測開始日が記載されている

# RRIデータの読み込み
TMP_RRI = pd.read_csv(FN_RRI, header=0, skiprows=5, encoding="shift-jis")
ic(TMP_RRI["time"])
# Pythonで日時データを扱いやすいように，%Y-%m-%d の形式に変換しとく
TMP_RRI["time"] = pd.to_datetime(TMP_RRI["time"])
ic(TMP_RRI["time"])

# EEGデータの読み込み
os.chdir(script_dir)
os.chdir(DIR_EEG)
TMP_EEG = pd.read_csv(FN_EEG, header=0, skiprows=0, encoding="shift-jis")
ic(TMP_EEG["Time"])
# EEGデータのTimeは時刻だけだったので，日付も追加
TMP_EEG["date.time"] = pd.to_datetime(
    date_eeg + " " + TMP_EEG["Time"], format="%Y/%m/%d %H:%M:%S"
)  # 日本時間のデータなので，utcオプションは必要ない
ic(TMP_EEG["date.time"])

# 時間が逆転している行を修正
N_EEG = len(TMP_EEG)
i_tmp = np.where(np.diff(TMP_EEG["date.time"].values.astype(np.int64)) < 0)[0] + 1
TMP_EEG.loc[i_tmp[0] : N_EEG, "date.time"] += timedelta(days=1)

# 正常値の設定
RRI_max = 1760
RRI_min = 350
RRI_diff = 200

RRI = TMP_RRI["RRI"]
time_RRI = TMP_RRI["time"]
D1_RRI = np.abs(np.diff(RRI, prepend=0))
D2_RRI = np.abs(np.diff(RRI, append=0))

mask = (RRI > RRI_min) & (RRI < RRI_max) & (D1_RRI < RRI_diff) & (D2_RRI < RRI_diff)
time_RRI_rev = time_RRI[mask]
RRI_rev = RRI[mask]

# リサンプリング
f_resamp = 2
time1 = min(time_RRI_rev)
time2 = max(time_RRI_rev)
time_r = pd.date_range(time1, time2, freq=f"{1/f_resamp}S")
RRI_r = np.interp(
    time_r.astype(np.int64) / 1e9, time_RRI_rev.astype(np.int64) / 1e9, RRI_rev
)

# RRIの平均値とSD
time_sub = TMP_EEG["date.time"]
n_sub = len(time_sub) - 1

time = []
meanRR = []
SDRR = []

for i in range(n_sub):
    time.append(time_sub.iloc[i + 1])
    sel = RRI_r[(time_r >= time_sub.iloc[i]) & (time_r < time_sub.iloc[i + 1])]
    meanRR.append(np.nanmean(sel))
    SDRR.append(np.nanstd(sel))

time = pd.to_datetime(time, utc=True).tz_convert("Asia/Tokyo")

# 統合データの作成
TMP_EEG = TMP_EEG.iloc[1:].copy()
TMP_EEG["meanRR"] = meanRR
TMP_EEG["SDRR"] = SDRR

# 統合データの書き出し
os.chdir(script_dir)
os.chdir(DIR_OUT)
TMP_EEG.to_csv(FN_OUT, index=False, sep=",")
TMP_EEG.to_csv(FN_OUT, index=False, sep=",")

# %%
