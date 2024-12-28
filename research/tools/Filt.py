#%%
import numpy as np
import matplotlib.pyplot as plt

def savgol(x, y, N, m, d=0):
    dx = x[1] - x[0]
    X = (np.c_[-N:N+1] * dx) ** np.r_[:m+1]
    print(np.c_[-N:N+1])
    print(np.r_[:m+1])
    print(X)
    C = np.linalg.pinv(X) # (X.T X)^-1 X.T
    print(C)
    x_ = x[N:-N]
    y_ = np.convolve(y[::-1], C[d], 'valid')[::-1]
    return x_, y_

# x = np.arange(-4, 5)
# y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# x_, y_,  = savgol(x, y, len(y), 1)

# print(x)

# plt.plot(y)
# plt.show()
# plt.plot(y_)
# plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
x = np.r_[-1:1+dx:dx]
coef = np.random.randn(5)
y = np.poly1d(coef)(x) + np.sin(abs(coef).sum() * x)
y_ = y + 0.1 * np.random.randn(*x.shape)

plt.plot(x, y, zorder=10, label='without noise')
plt.plot(x, y_, label='with noise')
plt.legend()
plt.show()


# %%
plt.plot(x, np.gradient(y, x), zorder=10, label='without noise')
plt.plot(x, np.gradient(y_, x), label='with noise')
plt.legend()
plt.show()


# %%
plt.plot(x, np.gradient(y, x), zorder=10, label='without noise')
plt.plot(x, np.gradient(y_, x), label='with noise')
plt.plot(*savgol(x, y_, 10, 2, d=1), label='with noise + SG(N=5, m=2)')
plt.legend()
plt.show()

plt.plot(x, y_, label='with noise + SG(N=5, m=2)')
plt.show()
# %%
