# %%
import os
from datetime import timedelta

import numpy as np
import pandas as pd

# フォルダの指定
DIR_RRI = "G:/Aoyama_RRI_EEG"
DIR_EEG = "G:/Aoyama_RRI_EEG"
DIR_OUT = "G:/Aoyama_RRI_EEG"

# ファイル名の指定
FN_RRI = "2019A自宅.csv"
FN_EEG = "2019A自宅睡眠段階.csv"
FN_OUT = "EEG_RRI.csv"

# 脳波の計測開始日
date_eeg = "2019-11-21"

# ファイル読み込み
os.chdir(DIR_RRI)
TMP_RRI = pd.read_csv(FN_RRI, header=0, skiprows=5)
TMP_RRI["time"] = pd.to_datetime(TMP_RRI["time"], utc=True).dt.tz_convert("Asia/Tokyo")

os.chdir(DIR_EEG)
TMP_EEG = pd.read_csv(FN_EEG, header=0, skiprows=0)
TMP_EEG["date.time"] = pd.to_datetime(
    date_eeg + " " + TMP_EEG["Time"], format="%Y-%m-%d %H:%M:%S", utc=True
).dt.tz_convert("Asia/Tokyo")

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
os.chdir(DIR_OUT)
TMP_EEG.to_csv(FN_OUT, index=False, sep=",")
TMP_EEG.to_csv(FN_OUT, index=False, sep=",")
