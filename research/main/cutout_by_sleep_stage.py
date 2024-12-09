#%%
import csv

# 読み込み対象のCSVファイル名
input_file = '../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv'
# 切り出し条件となる値（2列目の値）
target_value = 'R'

# 出力先ファイル
output_file = '../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/filtered_rem.csv'

# フィルタリング処理
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # ヘッダー行の書き込み
    header = next(reader)
    writer.writerow(header)

    # 2列目の値に基づいて行を切り出す
    for row in reader:
        if row[2] == target_value:  # 2列目の値をチェック
            writer.writerow(row)
