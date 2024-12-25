# %%
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from icecream import ic

# file = "../../../data/心拍変動まとめ_copy/2019A自宅.csv"
file = "../../../data/プロアシスト脳波・心拍_copy/2022年度（女性・自宅・避難所・車中泊）/心拍/4.自宅.csv"
data = pd.read_csv(file, encoding="shift-jis", skiprows=5)
ic(data)

# %%
# 文字列の1つ目の半角スペースよりあとの部分を切り出す
# time_extract を 1 次元の文字列配列として生成
time_extract = np.array([elm[elm.find(" ") + 1 :] for elm in data["time"]], dtype=str)
# ind = data["time"].find(" ")
# time_extract = data["time"][ind + 1 :]

ic(time_extract)


def modify_time(time_str, delta_seconds):
    """
    時刻を加算または減算する関数
    :param time_str: 時刻を示す文字列 ("hh:mm:ss.sss")
    :param delta_seconds: 加算または減算する秒数 (正の値で加算、負の値で減算)
    :return: 加算・減算後の時刻を文字列として返す
    """
    # 入力形式を解析
    time_format = "%H:%M:%S.%f"
    base_time = datetime.strptime(time_str, time_format)

    # 秒の差分を計算
    modified_time = base_time + timedelta(seconds=delta_seconds)

    # 結果を "hh:mm:ss.sss" 形式で返す
    return modified_time.strftime("%H:%M:%S.%f")[:-3]


# rriの積分時系列を生成してtimeと比較
time_sum_of_rri = [time_extract[0]]

for i, time in enumerate(time_extract):
    time_sum_of_rri.append(modify_time(time, data["RRI"][i] * 10**-3))

ic(np.array(time_sum_of_rri))

# %%
