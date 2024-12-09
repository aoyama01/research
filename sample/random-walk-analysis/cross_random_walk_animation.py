#%% Cross Random Walk Simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

# パラメータ設定
p1 = 0.5  # オブジェクト1の上に進む確率
p2 = 0.5  # オブジェクト2の上に進む確率
times = 100  # 時間ステップ数

# 初期位置設定
x1 = [0]  # オブジェクト1の位置リスト
x2 = [0]  # オブジェクト2の位置リスト

# 描画領域を作成
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, times)
ax.set_ylim(-50, 50)
line1, = ax.plot([], [], color='b', label='Time series 1')  # オブジェクト1用のライン
line2, = ax.plot([], [], color='r', label='Time series 2')  # オブジェクト2用のライン
plt.xlabel('Time steps', fontsize=16)
plt.ylabel('Position', fontsize=16)
plt.title('Cross Random Walk Simulation', fontsize=20)
plt.legend(fontsize=16)

# 初期化関数
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# 各フレームを更新する関数
def animate(i):
    # オブジェクト1のランダムウォーク
    if np.random.rand() < p1:
        x1.append(x1[-1] + 1)
    else:
        x1.append(x1[-1] - 1)

    # オブジェクト2のランダムウォーク
    if np.random.rand() < p2:
        x2.append(x2[-1] + 1)
    else:
        x2.append(x2[-1] - 1)

    # ラインデータを更新
    line1.set_data(np.arange(i + 1), x1[:i + 1])
    line2.set_data(np.arange(i + 1), x2[:i + 1])
    return line1, line2

# アニメーション設定
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=times, interval=100, blit=True)

# Jupyter上で表示用の設定
rc('animation', html='jshtml')

ani.save("cross_random_walk.gif", writer="imagemagick")

# Jupyter上でアニメーション表示
ani

# %%
