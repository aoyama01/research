# %%
# 3sinπx + 10sin4πx をフーリエ変換して絶対値2乗してみる
import matplotlib.pyplot as plt
import numpy as np

# 定義する信号のパラメータ
A, B = 2, 3  # 振幅
k, l = np.pi, 4 * np.pi  # 周波数
T = 10  # 観測時間
N = 1000  # サンプル数

# 時間軸
x = np.linspace(0, T, N)
y = A * np.sin(k * x) + B * np.sin(l * x)

# フーリエ変換
fft_y = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=x[1] - x[0])  # 周波数軸

print(y.shape[-1])


# パワースペクトル
power_spectrum = np.abs(fft_y) ** 2

# パワースペクトル密度の計算
psd = (np.abs(fft_y) ** 2) / T

# プロット
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(x, y)
plt.title(f"Original Signal: {A}sin({k:.2f}x) + {B}sin({l:.2f}x)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(3, 1, 2)
plt.plot(
    freqs[: len(freqs) // 2], power_spectrum[: len(freqs) // 2]
)  # 正の周波数のみ表示
plt.title(f"Power Spectrum: {A}sin({k:.2f}x) + {B}sin({l:.2f}x)")
plt.xlabel("Frequency")
plt.ylabel("Power")

plt.subplot(3, 1, 3)
plt.plot(freqs[: N // 2], psd[: N // 2])  # 正の周波数部分だけ表示
plt.title("Power Spectral Density")
plt.xlabel("Frequency")
plt.ylabel("PSD")
plt.grid()

plt.tight_layout()
plt.show()

# ピーク検出
peak_index = np.argmax(power_spectrum[: N // 2])  # 正の周波数部分のみ
peak_frequency = freqs[peak_index]
peak_amplitude = np.sqrt(power_spectrum[peak_index])

# 結果表示
print(f"Peak Frequency: {peak_frequency} Hz")
print(f"Peak Power Spectrum: {power_spectrum[peak_index]}")
print(f"Peak Amplitude: {peak_amplitude}")

# %%
# サンプル数とデータの作成
N = 8  # サンプル数
n = np.arange(N)  # サンプルインデックス
f_n = np.sin(n)  # 時系列データ

# DFTの計算
F_k = np.fft.fft(f_n)  # NumPyのFFTを使用（離散フーリエ変換）
frequencies = np.fft.fftfreq(N)  # 周波数の対応関係

# 振幅スペクトルと位相スペクトルの計算
amplitude = np.abs(F_k)  # 振幅
phase = np.angle(F_k)  # 位相

# 結果のプロット
plt.figure(figsize=(12, 6))

# 元の時系列データ
plt.subplot(1, 3, 1)
plt.stem(n, f_n, basefmt=" ")
plt.title("Time Series Data")
plt.xlabel("n")
plt.ylabel("f[n]")

# 振幅スペクトル
plt.subplot(1, 3, 2)
plt.stem(frequencies, amplitude, basefmt=" ")
plt.title("Amplitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# 位相スペクトル
plt.subplot(1, 3, 3)
plt.stem(frequencies, phase, basefmt=" ")
plt.title("Phase Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()

# %%
# それぞれのサンプル数における振幅と位相のピーク
# スペクトル: 光や信号などの波を成分に分解し，成分毎の強度を見やすく配列したもの
# 光を波長毎に分解した「分光スペクトル」や，信号の波形を周波数成分に分解した「周波数スペクトル」などがある
peak_ampSpec = []
peak_phaseSpec = []

start = 8
step = 1001

for N in range(start, step):
    n = np.arange(N)  # サンプルインデックス
    f_n = np.sin(n)  # 時系列データ

    # DFTの計算
    F_k = np.fft.fft(f_n)  # NumPyのFFTを使用（離散フーリエ変換）
    frequencies = np.fft.fftfreq(N)  # 周波数の対応関係

    # 振幅スペクトルと位相スペクトルの計算
    amplitude = np.abs(F_k)  # 振幅
    phase = np.angle(F_k)  # 位相

    # ピークの取得
    peak_ampSpec.append(max(amplitude))
    peak_phaseSpec.append(max(phase))

plt.figure(figsize=(12, 6))

# 振幅スペクトル
plt.subplot(1, 2, 1)
plt.stem(range(start, step), peak_ampSpec, basefmt=" ")
plt.title("Peak Amplitude")
plt.xlabel("Number of Samples")
plt.ylabel("Amplitude")

# 位相スペクトル
plt.subplot(1, 2, 2)
plt.stem(range(start, step), peak_phaseSpec, basefmt=" ")
plt.title("Peak Phase")
plt.xlabel("Number of Samples")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()

# %%
