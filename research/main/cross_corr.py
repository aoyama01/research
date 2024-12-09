#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the first file
file_path_1 = '../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv'
data_1 = pd.read_csv(file_path_1, encoding="shift-jis")

# Load the second CSV file while skipping the first 5 rows
file_path_2 = '../../data/プロアシスト脳波・心拍_origin/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet.csv'
data_2 = pd.read_csv(file_path_2, encoding="shift-jis", skiprows=5)

# Filter the required rows from 1290 to 45130, selecting every 30th row
# filtered_data_2 = data_2.iloc[1290:45130:30]
filtered_data_2 = data_2.iloc[1290:41850:30] # 計測間隔は脳波の30秒に合わせる

#%%
# Extract the relevant columns for cross-correlation: Delta_Ratio from the first file and RRI from the second file
delta_ratio = data_1['Delta_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(delta_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Delta Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

print('↑↑↑ 相関があるように見えるのは非定常だから？（2024/11/05）')
# %%
# Extract the relevant columns for cross-correlation: Theta_Ratio from the first file and RRI from the second file
theta_ratio = data_1['Theta_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(theta_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Theta Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

# %%
# Extract the relevant columns for cross-correlation: Alpha_Ratio from the first file and RRI from the second file
alpha_ratio = data_1['Alpha_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(alpha_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Alpha Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

# %%
# Extract the relevant columns for cross-correlation: Beta_Ratio from the first file and RRI from the second file
beta_ratio = data_1['Beta_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(beta_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Beta Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

# %%
# Extract the relevant columns for cross-correlation: Gamma_Ratio from the first file and RRI from the second file
gamma_ratio = data_1['Gamma_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(gamma_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Gamma Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

# %%
# Extract the relevant columns for cross-correlation: Sigma_Ratio from the first file and RRI from the second file
sigma_ratio = data_1['Sigma_Ratio']
rri = filtered_data_2['RRI']

# Calculate and plot the cross-correlation
plt.figure(figsize=(10, 6))
plt.xcorr(sigma_ratio, rri, maxlags=50, usevlines=True, normed=True, lw=2)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1)) # 0 から 1 まで 0.2 刻みで目盛を設定

# Adding labels and title
plt.title('Cross-correlation between Sigma Ratio and RRI')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.show()

# %%
