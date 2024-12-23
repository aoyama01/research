# %%
import pandas as pd

file_path = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet.csv"

# ファイルを、最初の5行をスキップして読み込む
data_2 = pd.read_csv(file_path, encoding="shift-jis", skiprows=5)

# 必要な行をフィルタリング（1290から41820まで、30行ごとに取得）
# cutout_data = data_2.iloc[1290:41850:30]
cutout_data = data_2.iloc[1285:41845:30]

# ic(cutout_data)

output_file = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet_cutout.csv"

# CSVファイルへの書き込み
cutout_data.to_csv(
    output_file, index=False, encoding="shift-jis"
)  # 日本語に対応するエンコーディング
