# %%
import matplotlib.pyplot as plt
import numpy as np

# 【重要】まず，RRIにデータを入れる
# 正常値の設定
RRI_max = 1250
RRI_min = 350  # 心臓の不応期(refractory period)
RRI_diff = 150

# RRIデータ（仮のデータを使います）
RRI = np.array([1000, 800, 950, 1100, 1300, 1150, 1050, 900, 850, 1200, 800, 900])
time_RRI = np.array(
    [i for i in range(len(RRI))]
)  # 時系列データ（単純にインデックスを使用）

# 時系列の長さ
n_RRI = len(RRI)

# 異常値の除外
D1_RRI = np.zeros(n_RRI)
D2_RRI = np.zeros(n_RRI)

# D1_RRIの計算（隣接するRRIの差）
D1_RRI[1:] = np.abs(RRI[1:] - RRI[:-1])

# D2_RRIはD1_RRIを1つシフトしたもの
D2_RRI[:-1] = D1_RRI[1:]

# D1_RRIとD2_RRIを用いることで，異常なRRI_diffを生じる隣接したRRIを両方削除する
""" 具体例
RRI    : [800 810 790 820 800]
RRI[1:] = [810, 790, 820] (RRIの1番目以降の全要素)
RRI[:-1] = [800, 810, 790] (RRIの最後の要素を除いた全要素)
D1_RRI[1:] = ... (結果をD1_RRIの1番目以降に代入)
隣接要素の差を計算すると, 元の配列より1つ要素が少なくなるため,
元の配列をマスクするためにはこの処理がひつよう
D1_RRI : [  0  10  20  30  20]
D2_RRI : [ 10  20  30  20   0]
"""

# 異常値を除外（範囲と差分が条件を満たすものだけ）
mask = (RRI > RRI_min) & (RRI < RRI_max) & (D1_RRI < RRI_diff) & (D2_RRI < RRI_diff)

time_RRI_rev = time_RRI[mask]
RRI_rev = RRI[mask]

# プロット
plt.plot(time_RRI_rev, RRI_rev, color="red")
plt.xlabel("time")
plt.ylabel("RRI [ms]")
plt.show()
