import os
import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime, timedelta

# ディレクトリとファイル名の指定
DIR_RRI = "D:/Documents/研究/緒形_睡眠/RRI"
FN = "A-homelab(20201203-08).csv"
DATE = "2020-12-03"
DATE = pd.to_datetime(DATE).date()

# ディレクトリの設定
os.chdir(DIR_RRI)

# パラメータの初期化
dt = 0.5
Q_level = 0.5
Q_RR = []
t_loc = []
VARR = []

logLF_m = []
logHF_m = []
logLF_nu_m = []
logHF_nu_m = []
logLF_TP_m = []
logHF_TP_m = []
LFHF_m = []
Q_level = 0.8

# CSVファイルを読み込む
TMP = pd.read_csv(FN, header=0, skiprows=5)
TMP['time'] = pd.to_datetime(TMP['time'])
TMP['date'] = TMP['time'].dt.date

# 指定した日付を選択
DAT = TMP[TMP['date'] == DATE]
time_R = pd.to_datetime(DAT['time'])
T1 = time_R.iloc[0].replace(second=0, microsecond=0)
T2 = time_R.iloc[-1].replace(second=0, microsecond=0)
RRI = DAT['RRI'].values

# 時間の計算
time_RRI = np.cumsum(RRI / 1000)

# 正常値の設定
RRI_max = 2000
RRI_min = 320
RRI_diff = 200

# 時系列の長さ
n_RRI = len(RRI)

# 異常値の除外
D1_RRI = np.zeros(n_RRI)
D1_RRI[1:] = np.abs(RRI[1:] - RRI[:-1])
valid_indices = (RRI > RRI_min) & (RRI < RRI_max) & (D1_RRI < RRI_diff)
time_RRI_rev = time_RRI[valid_indices]
time_R_rev = time_R[valid_indices]
RRI_rev = RRI[valid_indices]

# Q.RR計算
i = 0  # iの定義
Q_RR.append(len(RRI_rev) / len(RRI))

if Q_RR[i] >= Q_level:
    time_sub = pd.date_range(T1 - timedelta(minutes=5), T2 + timedelta(minutes=5), freq='5T')
    n_sub = len(time_sub) - 1
    LF = []
    HF = []
    LF_nu = []
    HF_nu = []
    LF_TP = []
    HF_TP = []
    VLF = []
    LFHF = []
    SDRR = []
    meanRR = []
    RMSSD = []
    pRR50 = []

    DATE = T1.date()

    for j in range(n_sub):
        t_loc.append(time_sub[j + 1])
        timeR_loc = time_RRI_rev[(time_R_rev >= time_sub[j]) & (time_R_rev < time_sub[j + 1])]
        RRI_loc = RRI_rev[(time_R_rev >= time_sub[j]) & (time_R_rev < time_sub[j + 1])]

        if np.sum(RRI_loc) >= 10000 * Q_level:
            VARR.append(np.var(RRI_loc))
            DRR = np.diff(RRI_loc)
            RMSSD.append(np.sqrt(np.mean(DRR**2)))
            pRR50.append(np.sum(np.abs(DRR) > 50) / len(DRR) * 100)
            SDRR.append(np.sqrt(VARR[j]))
            meanRR.append(np.mean(RRI_loc))

            # 再サンプリング
            t_resamp = np.arange(timeR_loc[0], timeR_loc[-1], dt)
            RRI_r = np.interp(t_resamp, timeR_loc, RRI_loc)

            # パワースペクトル密度の計算
            f, Pxx = signal.welch(RRI_r, fs=1/dt)

            psd_sum = np.sum(Pxx)
            psd_wo_VLF = np.sum(Pxx[f > 0.04])

            # 周波数領域指標
            HF.append(VARR[j] * np.sum(Pxx[(f > 0.15) & (f <= 0.4)]) / psd_sum)
            LF.append(VARR[j] * np.sum(Pxx[(f > 0.04) & (f <= 0.15)]) / psd_sum)
            LFHF.append(LF[j] / HF[j])

            # 追加項目
            VLF.append(VARR[j] * psd_wo_VLF / psd_sum)
            HF_nu.append(np.sum(Pxx[(f > 0.15) & (f <= 0.4)]) / psd_wo_VLF * 100)
            LF_nu.append(np.sum(Pxx[(f > 0.04) & (f <= 0.15)]) / psd_wo_VLF * 100)
            HF_TP.append(np.sum(Pxx[(f > 0.15) & (f <= 0.4)]) / psd_sum * 100)
            LF_TP.append(np.sum(Pxx[(f > 0.04) & (f <= 0.15)]) / psd_sum * 100)
        else:
            VARR.append(np.nan)
            SDRR.append(np.nan)
            RMSSD.append(np.nan)
            pRR50.append(np.nan)
            meanRR.append(np.nan)
            LF.append(np.nan)
            HF.append(np.nan)
            LFHF.append(np.nan)
            VLF.append(np.nan)
            HF_nu.append(np.nan)
            LF_nu.append(np.nan)
            HF_TP.append(np.nan)
            LF_TP.append(np.nan)

    logLF_m = np.nanmean(np.log(LF))
    logHF_m = np.nanmean(np.log(HF))
    LFHF_m = np.nanmean(LFHF)

# 結果の表示
print("logLF_m:", logLF_m)
print("logHF_m:", logHF_m)
print("LFHF_m:", LFHF_m)
