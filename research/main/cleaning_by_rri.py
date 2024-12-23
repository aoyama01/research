# %%
import numpy as np
import pandas as pd


# RRIの異常値除外と該当時刻の検出
def detect_anomalous_times(
    rri_data_file, rri_column, time_column, rri_max=1250, rri_min=350, rri_diff=150
):
    """
    心拍データからRRIの異常値を検出し，該当する時刻を返す関数．

    Parameters:
        rri_data_file (str): 心拍データファイルのファイルパス．
        rri_column (str): RRIデータの列名．
        time_column (str): 時刻データの列名．
        rri_max (int): RRIの最大許容値．
        rri_min (int): RRIの最小許容値．心臓の不応期(refractory period)による．
        rri_diff (int): RRI間の許容される差分．

    Returns:
        pd.Series: 異常値が検出された時刻のリスト．
    """
    # ファイル読み込み
    rri_data = pd.read_csv(rri_data_file, encoding="shift-jis")
    RRI = rri_data[rri_column].values
    time_RRI = rri_data[time_column].values

    # 異常値の検出
    n_RRI = len(RRI)
    D1_RRI = np.zeros(n_RRI)
    D2_RRI = np.zeros(n_RRI)
    D1_RRI[1:] = np.abs(RRI[1:] - RRI[:-1])  # D1_RRIの計算（隣接するRRIの差）
    D2_RRI[:-1] = D1_RRI[1:]  # D2_RRIはD1_RRIを1つシフトしたもの

    # D1_RRIとD2_RRIを用いることで，異常なRRI_diffを生じる隣接したRRIを両方削除する
    """ 具体例
    RRI    : [800 810 790 820 800]
    RRI[1:] = [810, 790, 820] (RRIの1番目以降の全要素)
    RRI[:-1] = [800, 810, 790] (RRIの最後の要素を除いた全要素)
    D1_RRI[1:] = ... (結果をD1_RRIの1番目以降に代入)
    隣接要素の差を計算すると, 元の配列より1つ要素が少なくなるため,
    元の配列をマスクするためにはこの処理が必要
    D1_RRI : [  0  10  20  30  20]
    D2_RRI : [ 10  20  30  20   0]
    """

    mask = (RRI > rri_min) & (RRI < rri_max) & (D1_RRI < rri_diff) & (D2_RRI < rri_diff)
    anomalous_times = time_RRI[~mask]  # 異常値を持つ時刻を抽出

    return pd.Series(anomalous_times)


# 他のファイルから該当時刻のデータを削除
def remove_anomalous_times(file, time_column, anomalous_times, output_file):
    """
    指定したファイルから異常な時刻のデータを削除して新規ファイルを保存する．

    Parameters:
        file (str): 入力ファイルのパス．
        time_column (str): 時刻データの列名．
        anomalous_times (pd.Series): 異常な時刻のリスト．
        output_file (str): 出力ファイルのパス．
    """
    data = pd.read_csv(file, encoding="shift-jis")
    cleaned_data = data[~data[time_column].isin(anomalous_times)]
    cleaned_data.to_csv(output_file, index=False)
    print(f"File saved to: {output_file}")


# メイン処理
if __name__ == "__main__":
    # 入力ファイル
    eeg_data_in = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv"
    rri_data_in = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet_cutout.csv"

    # 出力ファイル
    eeg_data_out = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ_cleaned.csv"
    rri_data_out = "../../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet_cutout_cleaned.csv"

    # カラム情報
    rri_column = "RRI"  # RRIデータが
    time_column = "time"  # 時刻データが含まれる列名

    # 異常値の検出
    anomalous_times = detect_anomalous_times(rri_data_in, rri_column, time_column)

    # 心拍データファイルの異常値削除
    remove_anomalous_times(rri_data_in, time_column, anomalous_times, rri_data_out)

    # EEGデータファイルの異常値削除
    remove_anomalous_times(eeg_data_in, "Time", anomalous_times, eeg_data_out)
