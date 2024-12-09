#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbm import FBM
from scipy.signal import savgol_filter

# Load the first file
file_path_1 = '../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv'
data_1 = pd.read_csv(file_path_1, encoding="shift-jis")

# Load the second CSV file while skipping the first 5 rows
file_path_2 = '../data/プロアシスト脳波・心拍_origin/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet.csv'
data_2 = pd.read_csv(file_path_2, encoding="shift-jis", skiprows=5)

# Filter the required rows from 1290 to 45130, selecting every 30th row
# filtered_data_2 = data_2.iloc[1290:45130:30]
filtered_data_2 = data_2.iloc[1290:41850:30] # 計測間隔は脳波の30秒に合わせる



#%%
# Extract the relevant columns for cross-correlation: Delta_Ratio from the first file and RRI from the second file
x1 = data_1['Delta_Ratio']
x2 = filtered_data_2['RRI']

print(np.mean(x1))
print(np.mean(x2))

# 平均を0にする処理
# x1 = x1 - np.mean(x1)
# x2 = x2 - np.mean(x2)

# 平均を0，標準偏差を1にする処理
# x1 = (x1 - np.mean(x1)) / np.std(x1)
# x2 = (x2 - np.mean(x2)) / np.std(x2)

n = len(x1) # 時系列の長さ

# サンプル時系列のプロット
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, color='green')
# 現在の横軸の目盛り位置とラベルを取得
current_ticks = plt.gca().get_xticks()
# 目盛りを30倍した新しいラベルを設定
new_labels = [int(tick * 30) for tick in current_ticks]
plt.xticks(ticks=current_ticks, labels=new_labels)
plt.xlim(-75, 1425)
plt.xlabel("sec")
plt.ylabel("Delta ratio (mean fixed to 0)")
plt.title("Delta ratio")

plt.subplot(2, 2, 2)
plt.plot(x2, color='blue')
# plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("sec")
plt.ylabel("RRI (mean fixed to 0)")
plt.title("RR-interval")

# DMA解析で使用するスケール
n_s = 20
s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

# DMAの解析
F1, F2, F12_sq = [], [], []
y1, y2 = np.cumsum(x1), np.cumsum(x2)  # 累積和（積分）

# 0次のDMCAを適用
for scale in s:
    y1_detrend = y1 - savgol_filter(y1, window_length=scale, polyorder=1)
    y2_detrend = y2 - savgol_filter(y2, window_length=scale, polyorder=1)
    F1.append(np.sqrt(np.mean(y1_detrend ** 2)))
    F2.append(np.sqrt(np.mean(y2_detrend ** 2)))
    F12_sq.append(np.mean(y1_detrend * y2_detrend))

rho = np.array(F12_sq) / (np.array(F1) * np.array(F2))

# クロス相関のプロット
plt.subplot(2, 2, 3)
plt.plot(np.log10(s), rho, color='red', marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='--')
plt.axhline(y=-1, color='gray', linestyle='--')
plt.ylim(-1, 1)
plt.xlabel("log10(s)")
plt.ylabel("Cross-correlation (rho)")
plt.title("Cross-correlation")

# スケーリングプロット
log10F1 = np.log10(F1)
log10F2 = np.log10(F2)
log10F12 = np.log10(np.abs(F12_sq)) / 2
y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
y_max = max(log10F1.max(), log10F2.max(), log10F12.max())

plt.subplot(2, 2, 4)
plt.plot(np.log10(s), log10F1, color='green', marker='^', linestyle='-', label='log10(F1)')
plt.plot(np.log10(s), log10F2, color='blue', marker='s', linestyle='-', label='log10(F2)')
plt.plot(np.log10(s), log10F12, color='red', marker='o', linestyle='-', label='log10(F12)')
plt.ylim(y_min, y_max)
plt.xlabel("log10(s)")
plt.ylabel("log10(F(s))")
plt.title("Scaling")
plt.legend()

plt.tight_layout()
plt.show()
# 1/f ゆらぎは数十秒から数時間あるから
# 30秒間隔のサンプリングでも問題ない？（2024/11/06）



#%%
# Extract the relevant columns for cross-correlation: Theta_Ratio from the first file and RRI from the second file
x1 = data_1['Theta_Ratio']
x2 = filtered_data_2['RRI']

print(np.mean(x1))
print(np.mean(x2))

# 平均を0にする処理
# x1 = x1 - np.mean(x1)
# x2 = x2 - np.mean(x2)

# 平均を0，標準偏差を1にする処理
x1 = (x1 - np.mean(x1)) / np.std(x1)
x2 = (x2 - np.mean(x2)) / np.std(x2)

n = len(x1) # 時系列の長さ

# サンプル時系列のプロット
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, color='green')
# 現在の横軸の目盛り位置とラベルを取得
current_ticks = plt.gca().get_xticks()
# 目盛りを30倍した新しいラベルを設定
new_labels = [int(tick * 30) for tick in current_ticks]
plt.xticks(ticks=current_ticks, labels=new_labels)
plt.xlim(-75, 1425)
plt.xlabel("sec")
plt.ylabel("Theta ratio (mean fixed to 0)")
plt.title("Theta ratio")

plt.subplot(2, 2, 2)
plt.plot(x2, color='blue')
# plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("sec")
plt.ylabel("RRI (mean fixed to 0)")
plt.title("RR-interval")

# DMA解析で使用するスケール
n_s = 20
s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

# DMAの解析
F1, F2, F12_sq = [], [], []
y1, y2 = np.cumsum(x1), np.cumsum(x2)  # 累積和（積分）

# 0次のDMCAを適用
for scale in s:
    y1_detrend = y1 - savgol_filter(y1, window_length=scale, polyorder=0)
    y2_detrend = y2 - savgol_filter(y2, window_length=scale, polyorder=0)
    F1.append(np.sqrt(np.mean(y1_detrend ** 2)))
    F2.append(np.sqrt(np.mean(y2_detrend ** 2)))
    F12_sq.append(np.mean(y1_detrend * y2_detrend))

rho = np.array(F12_sq) / (np.array(F1) * np.array(F2))

# クロス相関のプロット
plt.subplot(2, 2, 3)
plt.plot(np.log10(s), rho, color='red', marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='--')
plt.axhline(y=-1, color='gray', linestyle='--')
plt.ylim(-1, 1)
plt.xlabel("log10(s)")
plt.ylabel("Cross-correlation (rho)")
plt.title("Cross-correlation")

# スケーリングプロット
log10F1 = np.log10(F1)
log10F2 = np.log10(F2)
log10F12 = np.log10(np.abs(F12_sq)) / 2
y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
y_max = max(log10F1.max(), log10F2.max(), log10F12.max())

plt.subplot(2, 2, 4)
plt.plot(np.log10(s), log10F1, color='green', marker='^', linestyle='-', label='log10(F1)')
plt.plot(np.log10(s), log10F2, color='blue', marker='s', linestyle='-', label='log10(F2)')
plt.plot(np.log10(s), log10F12, color='red', marker='o', linestyle='-', label='log10(F12)')
plt.ylim(y_min, y_max)
plt.xlabel("log10(s)")
plt.ylabel("log10(F(s))")
plt.title("Scaling")
plt.legend()

plt.tight_layout()
plt.show()
# 1/f ゆらぎは数十秒から数時間あるから
# 30秒間隔のサンプリングでも問題ない？（2024/11/06）



# %%
# Extract columns: Alpha_Ratio from the first file and RRI from the second file
x1 = data_1['Alpha_Ratio']
x2 = filtered_data_2['RRI']

# 平均を0にする処理
x1 = x1 - np.mean(x1)
x2 = x2 - np.mean(x2)

n = len(x1) # 時系列の長さ
# H_common = 0.8  # 共通成分のHurst指数
# SD_common = 0.9  # 共通成分の標準偏差

# Fractional Gaussian Noiseの生成と標準化
# def generate_fgn(n, H):
#     fbm = FBM(n=n, hurst=H)
#     return (fbm.fgn() - np.mean(fbm.fgn())) / np.std(fbm.fgn())

# eps_common = generate_fgn(n, H_common) * SD_common

# x1 = x1 + eps_common
# x2 = x2 + eps_common

# サンプル時系列のプロット
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x1, color='green', label='x1')
# plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("i")
plt.title("Sample time series (x1)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x2, color='blue', label='x2')
# plt.plot(eps_common, color='red', linestyle='--', label='common')
plt.xlabel("i")
plt.title("Sample time series (x2)")
plt.legend()

# DMA解析で使用するスケール
n_s = 20
s = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(n / 4), n_s)) / 2) * 2 + 1).astype(int)

# DMAの解析
F1, F2, F12_sq = [], [], []
y1, y2 = np.cumsum(x1), np.cumsum(x2)  # 累積和（積分）

# 0次のDMCAを適用
for scale in s:
    y1_detrend = y1 - savgol_filter(y1, window_length=scale, polyorder=0)
    y2_detrend = y2 - savgol_filter(y2, window_length=scale, polyorder=0)
    F1.append(np.sqrt(np.mean(y1_detrend ** 2)))
    F2.append(np.sqrt(np.mean(y2_detrend ** 2)))
    F12_sq.append(np.mean(y1_detrend * y2_detrend))

rho = np.array(F12_sq) / (np.array(F1) * np.array(F2))

# クロス相関のプロット
plt.subplot(2, 2, 3)
plt.plot(np.log10(s), rho, color='red', marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='--')
plt.axhline(y=-1, color='gray', linestyle='--')
# plt.ylim(-1, 1)
plt.xlabel("log10(s)")
plt.ylabel("Cross-correlation (rho)")
plt.title("Cross-correlation")

# スケーリングプロット
log10F1 = np.log10(F1)
log10F2 = np.log10(F2)
log10F12 = np.log10(np.abs(F12_sq)) / 2
y_min = min(log10F1.min(), log10F2.min(), log10F12.min())
y_max = max(log10F1.max(), log10F2.max(), log10F12.max())

plt.subplot(2, 2, 4)
plt.plot(np.log10(s), log10F1, color='green', marker='^', linestyle='-', label='log10(F1)')
plt.plot(np.log10(s), log10F2, color='blue', marker='s', linestyle='-', label='log10(F2)')
plt.plot(np.log10(s), log10F12, color='red', marker='o', linestyle='-', label='log10(F12)')
plt.ylim(y_min, y_max)
plt.xlabel("log10(s)")
plt.ylabel("log10(F(s))")
plt.title("Scaling")
plt.legend()

plt.tight_layout()
plt.show()
