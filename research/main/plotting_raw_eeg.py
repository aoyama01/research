# %%
import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルの読み込み
file_path = "../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv"  # 適切なファイルパスに変更してください
data = pd.read_csv(file_path)

# 'Time'列を時間形式に変換し、秒に変換
data["Time_in_seconds"] = pd.to_timedelta(data["Time"].astype(str)).dt.total_seconds()

# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(
    data["Time_in_seconds"], data["Delta_Power"], marker="o", linestyle="-", color="b"
)

# ラベルとタイトルを追加
plt.xlabel("Time (seconds)")
plt.ylabel("Delta Power")
plt.title("Delta Power vs Time")

# グラフの表示
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# %%
import pandas as pd

# CSVファイルの読み込み
file_path = "../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv"  # 適切なファイルパスに変更してください
data = pd.read_csv(file_path)

# 'Time'列を時間形式に変換し、秒に変換
data["Time_in_seconds"] = pd.to_timedelta(data["Time"].astype(str)).dt.total_seconds()

# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(
    data["Time_in_seconds"], data["Delta_Power"], marker=".", linestyle="-", color="b"
)
plt.plot(
    data["Time_in_seconds"], data["Theta_Power"], marker=".", linestyle="-", color="g"
)
plt.plot(
    data["Time_in_seconds"], data["Alpha_Power"], marker=".", linestyle="-", color="r"
)
plt.plot(
    data["Time_in_seconds"], data["Beta_Power"], marker=".", linestyle="-", color="c"
)
plt.plot(
    data["Time_in_seconds"], data["Gamma_Power"], marker=".", linestyle="-", color="m"
)
plt.plot(
    data["Time_in_seconds"], data["Sigma_Power"], marker=".", linestyle="-", color="y"
)

# ラベルとタイトルを追加
plt.xlabel("Time (seconds)")
plt.ylabel("Power")
plt.title("Power vs Time")

# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(
    data["Time_in_seconds"], data["Delta_Ratio"], marker=".", linestyle="-", color="b"
)
plt.plot(
    data["Time_in_seconds"], data["Theta_Ratio"], marker=".", linestyle="-", color="g"
)
plt.plot(
    data["Time_in_seconds"], data["Alpha_Ratio"], marker=".", linestyle="-", color="r"
)
plt.plot(
    data["Time_in_seconds"], data["Beta_Ratio"], marker=".", linestyle="-", color="c"
)
plt.plot(
    data["Time_in_seconds"], data["Gamma_Ratio"], marker=".", linestyle="-", color="m"
)
plt.plot(
    data["Time_in_seconds"], data["Sigma_Ratio"], marker=".", linestyle="-", color="y"
)

# ラベルとタイトルを追加
plt.xlabel("Time (seconds)")
plt.ylabel("Ratio")
plt.title("Ratio vs Time")

# グラフの表示
plt.tight_layout()
plt.show()

# %%
