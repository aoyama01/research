#%%
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# データの出力フォルダ
DIR_out = "D:/Document/研究/緒形_睡眠/分析結果"
DIR_INFO = "D:/Document/研究/緒形_睡眠"
DIR_EEG = "D:/Document/研究/緒形_睡眠/EEG"
DIR_HR = "D:/Document/研究/緒形_睡眠/RRI"

# 情報フォルダへ移動し、データ読み込み
os.chdir(DIR_INFO)
INFO = pd.read_csv("EXP_INFO.csv").dropna()
N_exp = INFO.shape[0]

# EEGデータの読み込み
os.chdir(DIR_EEG)
FN_EEG = [f for f in os.listdir(DIR_EEG) if os.path.isfile(os.path.join(DIR_EEG, f))]

# main loop
for i in range(13):
    # SLEEP_STAGEファイルの読み込み
    os.chdir(DIR_EEG)
    TMP = pd.read_csv(INFO.loc[i, "SLEEP_STAGE"])

    # 日付と時間の結合
    TMP['Time'] = pd.to_datetime(INFO.loc[i, "DATE"] + " " + TMP['Time'])

    # 時間の変換
    TMP['Time'] = TMP['Time'] + timedelta(seconds=24*60*60) * (TMP['Time'].diff().dt.total_seconds() < 0).cumsum()
    T1 = TMP['Time'].min() + timedelta(minutes=10)
    T2 = TMP['Time'].max() - timedelta(minutes=10)

    # スコアリングの処理
    DAT_STG = TMP[TMP['Epoch'] != ""].copy()
    DAT_STG['STG'] = 0
    DAT_STG.loc[DAT_STG['Score'] == "R", 'STG'] = -1
    DAT_STG.loc[DAT_STG['Score'] == "N1", 'STG'] = -2
    DAT_STG.loc[DAT_STG['Score'] == "N2", 'STG'] = -3
    DAT_STG.loc[DAT_STG['Score'] == "N3", 'STG'] = -4

    # スコアごとの色設定
    DAT_STG['col'] = "#fde8e8"
    DAT_STG.loc[DAT_STG['Score'] == "R", 'col'] = "#d8ffd8"
    DAT_STG.loc[DAT_STG['Score'] == "N1", 'col'] = "#e6f8ff"
    DAT_STG.loc[DAT_STG['Score'] == "N2", 'col'] = "#d4ecff"
    DAT_STG.loc[DAT_STG['Score'] == "N3", 'col'] = "#b9e0ff"
    
#%%
# RRIデータの読み込みとフィルタリング (続き)
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# データの出力フォルダと情報フォルダの定義
DIR_out = "D:/Document/研究/緒形_睡眠/分析結果"
DIR_INFO = "D:/Document/研究/緒形_睡眠"
DIR_EEG = "D:/Document/研究/緒形_睡眠/EEG"
DIR_HR = "D:/Document/研究/緒形_睡眠/RRI"

# set directory
os.chdir(DIR_INFO)
INFO = pd.read_csv("EXP_INFO.csv").dropna()
N_exp = INFO.shape[0]

# set directory and get list of EEG files
os.chdir(DIR_EEG)
FN_EEG = os.listdir(DIR_EEG)

# 時系列データ処理
for i in range(13):
    os.chdir(DIR_EEG)
    TMP = pd.read_csv(INFO.iloc[i]["SLEEP_STAGE"])
    TMP['Time'] = pd.to_datetime(INFO.iloc[i]["DATE"] + ' ' + TMP['Time'])
    
    # データ変換と時間調整
    n_time = len(TMP['Time'])
    n_24 = TMP['Time'].diff().lt(timedelta(0)).cumsum()
    TMP.loc[n_24 > 0, 'Time'] += timedelta(days=1)
    
    T1 = TMP['Time'].min() + timedelta(minutes=10)
    T2 = TMP['Time'].max() - timedelta(minutes=10)
    
    DAT_STG = TMP[TMP['Epoch'].notna()].copy()
    DAT_STG['STG'] = 0
    DAT_STG.loc[DAT_STG['Score'] == "R", 'STG'] = -1
    DAT_STG.loc[DAT_STG['Score'] == "N1", 'STG'] = -2
    DAT_STG.loc[DAT_STG['Score'] == "N2", 'STG'] = -3
    DAT_STG.loc[DAT_STG['Score'] == "N3", 'STG'] = -4
    
    DAT_STG['col'] = "#fde8e8"
    DAT_STG.loc[DAT_STG['Score'] == "R", 'col'] = "#d8ffd8"
    DAT_STG.loc[DAT_STG['Score'] == "N1", 'col'] = "#e6f8ff"
    DAT_STG.loc[DAT_STG['Score'] == "N2", 'col'] = "#d4ecff"
    DAT_STG.loc[DAT_STG['Score'] == "N3", 'col'] = "#b9e0ff"
    
    os.chdir(DIR_HR)
    dt = 0.5
    Q_level = 0.5
    Q_RR, t_loc, VARR = [], [], []
    logLF_m, logHF_m, LFHF_m = [], [], []
    logLF_nu_m, logHF_nu_m, logLF_TP_m, logHF_TP_m = [], [], [], []

    DAT = pd.read_csv(INFO.iloc[i]["MyBEAT"] + ".csv", skiprows=5)
    DAT['time'] = pd.to_datetime(DAT['time'])
    DAT = DAT[(DAT['time'] >= T1) & (DAT['time'] <= T2)]
    time_R = pd.to_datetime(DAT['time'])
    RRI = DAT['RRI']
    
    # RRIの時間とフィルタリング
    time_RRI = np.cumsum(RRI / 1000.0)
    RRI_max, RRI_min, RRI_diff = 2000, 400, 200
    n_RRI = len(RRI)
    D1_RRI = np.abs(np.diff(RRI, prepend=0))
    D2_RRI = np.abs(np.diff(RRI, append=0))
    
    mask = (RRI > RRI_min) & (RRI < RRI_max) & (D1_RRI < RRI_diff) & (D2_RRI < RRI_diff)
    time_RRI_rev, time_R, RRI_rev = time_RRI[mask], time_R[mask], RRI[mask]

    Q_RR_val = len(RRI_rev) / len(RRI)
    if Q_RR_val >= Q_level:
        time_sub = pd.date_range(T1 - timedelta(minutes=5), T2 + timedelta(minutes=5), freq="5T")
        n_sub = len(time_sub) - 1
        LF, HF, LFHF, SDRR, meanRR, RMSSD, pRR50, SLP_STG = [], [], [], [], [], [], [], []

        for j in range(n_sub):
            t_loc.append(time_sub[j + 1])
            timeR_loc = time_RRI_rev[(time_R >= time_sub[j]) & (time_R < time_sub[j + 1])]
            RRI_loc = RRI_rev[(time_R >= time_sub[j]) & (time_R < time_sub[j + 1])]
            
            if DAT_STG[(DAT_STG['Time'] >= time_sub[j]) & (DAT_STG['Time'] < time_sub[j + 1])]['Score'].eq("W").sum() > 0:
                SLP_STG.append("W")
            else:
                stage_counts = DAT_STG[(DAT_STG['Time'] >= time_sub[j]) & (DAT_STG['Time'] < time_sub[j + 1])]['Score'].value_counts()
                SLP_STG.append(stage_counts.idxmax())
            
            if RRI_loc.sum() >= 10000 * Q_level:
                VARR.append(np.var(RRI_loc))
                DRR = np.diff(RRI_loc)
                RMSSD.append(np.sqrt(np.mean(DRR**2)))
                pRR50.append(np.sum(np.abs(DRR) > 50) / len(DRR) * 100)
                SDRR.append(np.sqrt(VARR[-1]))
                meanRR.append(np.mean(RRI_loc))

                # 補間と周波数分析
                t_resamp = np.arange(timeR_loc[0], timeR_loc[-1], dt)
                RRI_r = np.interp(t_resamp, timeR_loc, RRI_loc)
                psd_freq, psd_spec = signal.welch(RRI_r, fs=1/dt)
                
                # スペクトルの全面積
                psd_sum = np.sum(psd_spec)
                psd_wo_VLF = np.sum(psd_spec[psd_freq > 0.04])

                HF.append(VARR[-1] * np.sum(psd_spec[(psd_freq > 0.15) & (psd_freq <= 0.4)]) / psd_sum)
                LF.append(VARR[-1] * np.sum(psd_spec[(psd_freq > 0.04) & (psd_freq <= 0.15)]) / psd_sum)
                LFHF.append(LF[-1] / HF[-1])
                
                # 追加指標
                VLF = VARR[-1] * psd_wo_VLF / psd_sum
                HF_nu = 100 * np.sum(psd_spec[(psd_freq > 0.15) & (psd_freq <= 0.4)]) / psd_wo_VLF
                LF_nu = 100 * np.sum(psd_spec[(psd_freq > 0.04) & (psd_freq <= 0.15)]) / psd_wo_VLF
                HF_TP = 100 * np.sum(psd_spec[(psd_freq > 0.15) & (psd_freq <= 0.4)]) / psd_sum
                LF_TP = 100 * np.sum(psd_spec[(psd_freq > 0.04) & (psd_freq <= 0.15)]) / psd_sum
            else:
                VARR.append(np.nan)
                SDRR.append(np.nan)
                RMSSD.append(np.nan)
                pRR50.append(np.nan)
                meanRR.append(np.nan)
                LF.append(np.nan)
                HF.append(np.nan)
                LFHF.append(np.nan)
        
        # 指標の平均を計算
        logLF_m = np.nanmean(np.log(np.array(LF)[(np.array(LF) > 0) & (np.array(HF) > 0)]))
        logHF_m = np.nanmean(np.log(np.array(HF)[(np.array(LF) > 0) & (np.array(HF) > 0)]))
        LFHF_m = np.nanmean(np.array(LFHF)[(np.array(LF) > 0) & (np.array(HF) > 0)])
        
        # 結果をプロット
        os.chdir(DIR_out)
        with PdfPages(f"{INFO.iloc[i]['SUBJECT']}_{INFO.iloc[i]['CONDITION']}_HRV.pdf") as pdf:
            fig, axs = plt.subplots(4, 2, figsize=(12, 12.5))
            axs = axs.ravel()

            # 各プロット処理 (例: Sleep Stage)
            axs[0].plot(DAT_STG['Time'], DAT_STG['STG'], lw=1.5)
            axs[0].set_title("Sleep Stage")
            pdf.savefig(fig)
            plt.close()
