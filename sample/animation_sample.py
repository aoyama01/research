#%%
# animationでプロット点を並行に動かしていくプログラム
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc # rc for colab
from IPython.display import HTML # for colab # Jupyter上でhtmlを表示するために必要

fig, ax = plt.subplots() # 描画領域を作成
ax.set_xlim([0, 10]) # x軸の範囲を指定

scat = ax.scatter(1, 0) # 散布図を作成
x = np.linspace(0, 10) # [0, 10]の範囲で等間隔の数値を生成

def animate(i):
  scat.set_offsets((x[i], 0)) # 差分更新のためののオフセットを設定
  return scat # オフセットを返す

# To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me')), bitrate=1800
# ani.save('scatter.gif', writer=writer)
ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=len(x) - 1, interval=50)
# フレーム数はlen(x) - 1．
# これがanimate()に渡され，時間intervalおいてframesが終了するまで繰り返される
# フレーム切り替えは50ミリ秒
rc('animation', html='jshtml')  # javascriptを動かすことができるように書式を設定
                                # https://atatat.hatenablog.com/entry/colab_python8_animation
ani # アニメーションを表示

#%%
# animationで関数f(t)を連続的に表示させるプログラム
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc # rc for colab
from IPython.display import HTML # for colab # Jupyter上でhtmlを表示するために必要

def f(t):
  return np.exp(-t/8) * np.cos(4*np.pi*t)

t = np.arange(0.0, 16.0, 0.05) # [0.0, 16.0) の範囲で0.05刻みの値を生成

# fig, ax = plt.subplots() # 描画領域を作成
fig = plt.figure() # 描画キャンパスを新規作成
# plt.xlim([0, 10]) # x軸の範囲を指定
plt.ylim([-1.1, 1.1]) # y軸の範囲を指定

def animate(i):
  if t[i] < 1:
    plt.xlim([0, 1])
  elif t[i] < 2:
    plt.xlim([0, 2])
  elif t[i] < 4:
    plt.xlim([0, 4])
  elif t[i] < 8:
    plt.xlim([0, 8])
  else:
    plt.xlim([0, 16])
  plt.plot(t[0:i], f(t[0:i]), color='b') # 差分更新のためのオフセットを設定
  return plt, # プロットを返す

ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=len(t) - 1, interval=100)
# フレーム数はlen(t) - 1．
# これがanimate()に渡され，intervalだけ時間をおいてframesが終了するまで繰り返される
# フレーム切り替えは100ミリ秒
rc('animation', html='jshtml') # javascriptを動かすことができるように書式を設定
ani # アニメーションを表示
