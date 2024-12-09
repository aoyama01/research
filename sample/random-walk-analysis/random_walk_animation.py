#%%
### まず1点だけでシミュレーションしてみる
#%%
import numpy as np
import matplotlib.pyplot as plt

p = 0.5 # 上側に歩く確率

x = [0] # 初期値

# 時間の長さを指定
times = 100

# Adding 1 with a probability of p, or -1 with a probability of (1-p)
for i in range(times - 1):
    if np.random.rand() < p:
        x.append(x[-1] + 1)
    else:
        x.append(x[-1] - 1)

# プロット
plt.figure(figsize=(8, 6))
plt.plot(np.arange(times), x, label=r'equation of x and y', color='b')
plt.xlabel('times')
plt.ylim([-50, 50]) # y軸の範囲を指定
plt.ylabel('x')
plt.title('Title')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

#%%
# アニメーションver.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc # rc for colab
from IPython.display import HTML # for colab # Jupyter上でhtmlを表示するために必要
# import time  # 時間計測用

# 計測開始
# start_time = time.time()

p = 0.5 # ランダムウォークの挙動が依存する確率

x = [0] # 初期値

# 繰り返し回数を指定
times = 100

# fig, ax = plt.subplots() # 描画領域を作成
fig = plt.figure(figsize=(10, 6)) # 描画キャンパスを新規作成
plt.xlabel('Time steps', fontsize=16)
plt.ylabel('Position', fontsize=16)
plt.title('Random Walk Simulation', fontsize=20)
plt.xlim(0, times) # x軸の範囲を指定
plt.ylim([-50, 50]) # y軸の範囲を指定
plt.grid(True, which="both", ls="--", lw=0.5)

def animate(i):
    if np.random.rand() < p:
        x.append(x[-1] + 1) # x[i]にするとグラフがおかしくなる．インデックスがややこしくなってる？
    else:
        x.append(x[-1] - 1)
    plt.plot(np.arange(i+1), x[:i+1], color='b') # 差分更新のためのオフセットを設定
    return plt, # プロットを返す

ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=times - 1, interval=100)
# フレーム数はlen(t) - 1．
# これがanimate()に渡され，intervalだけ時間をおいてframesが終了するまで繰り返される
# フレーム切り替えは100ミリ秒
rc('animation', html='jshtml') # javascriptを動かすことができるように書式を設定

ani.save('random_walk_animation.gif', writer='imagemagick')
# plt.show()
# ↑↑↑ 処理時間を計測したかったらこっちでアニメーションを生成したらいい

# 計測終了と表示
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"一連の処理にかかった時間: {elapsed_time:.2f}秒")

ani  # アニメーションを表示
# セルを全選択するショートカットは：ctrl + alt + \

#%%
# アニメーション(ちょっと処理が効率的?)ver.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc  # rc for colab
from IPython.display import HTML  # for colab # Jupyter上でhtmlを表示するために必要
# import time  # 時間計測用

# 計測開始
# start_time = time.time()

p = 0.5  # ランダムウォークの挙動が依存する確率
times = 100  # 繰り返し回数を指定

# ランダムウォークの初期化
x = [0]  # 初期値

# 描画領域を作成
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, times)
ax.set_ylim(-times // 2, times // 2)
line, = ax.plot([], [], color='b')  # 更新用の線を定義

# フレーム初期化関数
def init():
    line.set_data([], [])
    return line,

# 各フレームを更新する関数
def animate(i):
    # ランダムウォークのステップ追加
    if np.random.rand() < p:
        x.append(x[-1] + 1)
    else:
        x.append(x[-1] - 1)

    # グラフデータを更新
    line.set_data(np.arange(i + 1), x[:i + 1])
    return line,

# アニメーションの設定
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=times, interval=100, blit=True)

# Jupyter上で表示用の設定
rc('animation', html='jshtml')

# ani.save('random_walk_animation.gif', writer='imagemagick')
# plt.show()
# ↑↑↑ 処理時間を計測したかったらこっちでアニメーションを生成したらいい

# 計測終了と表示
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"一連の処理にかかった時間: {elapsed_time:.2f}秒")

ani  # アニメーションを表示
# ↑↑↑ これはインタラクティブにアニメーションを再生するやつなので，時間計測できない

#%%
# 時間計測用のテンプレ
import time  # 時間計測用

# 計測開始
start_time = time.time()

'''
処理内容

'''

# 計測終了と表示
end_time = time.time()
elapsed_time = end_time - start_time
print(f"一連の処理にかかった時間: {elapsed_time:.2f}秒")

#%%
# 100個のオブジェクトを生成

# 各オブジェクトに対応する色を格納する配列を生成

# 繰り返し回数を指定

# 繰り返し
    # 各点について次の時点での位置を求める
