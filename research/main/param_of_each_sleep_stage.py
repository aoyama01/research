# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic

# ファイルパスのリストを作成
base_path = "../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/"
file_names = [
    "filtered_wake.csv",
    "filtered_rem.csv",
    "filtered_n1.csv",
    "filtered_n2.csv",
    "filtered_n3.csv",
]

# ファイルを順に読み込む
data_files = [
    pd.read_csv(f"{base_path}{file_name}", encoding="shift-jis")
    for file_name in file_names
]

sleepStages = ["Wake", "REM", "N1", "N2", "N3"]

# 横軸(data columns)の文字列
labels = [
    "Delta_Ratio",
    "Theta_Ratio",
    "Alpha_Ratio",
    "Beta_Ratio",
    "Gamma_Ratio",
    "Sigma_Ratio",
]

for i, data_file in enumerate(data_files):
    # グラフタイトルで使うからループ回数(インデックス)も取得しとく
    # 各列のデータを抽出
    datas = [data_file[f"{label}"] for label in labels]
    # datas = [
    #     data_file["Delta_Ratio"],
    #     data_file["Theta_Ratio"],
    #     data_file["Alpha_Ratio"],
    #     data_file["Beta_Ratio"],
    #     data_file["Gamma_Ratio"],
    #     data_file["Sigma_Ratio"],
    # ]

    # 各列のデータの平均を計算
    meanList = []
    minList = []
    maxList = []
    for data in datas:
        meanList.append(np.mean(data))
        minList.append(np.min(data))
        maxList.append(np.max(data))
    ic(meanList)
    ic(minList)
    ic(maxList)

    plt.figure(figsize=(10, 6))  # グラフのサイズを変更
    # 棒グラフの作成
    plt.bar(range(len(meanList)), meanList, width=0.4)

    # 横軸のラベルを設定
    plt.xticks(range(len(meanList)), labels)

    plt.ylim(0, 1)
    # グラフのラベル設定
    plt.xlabel("Brain waves")
    plt.ylabel("PS ratio of each brain wave")
    plt.grid(axis="y")  # y軸のグリッド線のみを表示
    plt.title(f'PS ratio in the case of "{sleepStages[i]}"')

    # レイアウト自動調整
    # plt.tight_layout()

    # グラフの表示
    plt.show()

# %%
