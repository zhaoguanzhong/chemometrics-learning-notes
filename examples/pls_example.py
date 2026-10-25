"""PLS 回归示例：在合成光谱数据上预测浓度。

生成方式：两个高斯峰的线性组合 + 噪声，浓度决定峰高比例。
运行：pip install numpy scikit-learn && python pls_example.py
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

n_samples, n_wavelengths = 80, 200
wavelength = np.linspace(400, 1000, n_wavelengths)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# 真实浓度（待预测量）
concentration = rng.uniform(0, 10, n_samples)

# 合成光谱：浓度越高，650 nm 的峰越强，550 nm 的峰越弱
spectra = (
    concentration[:, None] * gaussian(wavelength, 650, 40)[None, :]
    + (10 - concentration)[:, None] * gaussian(wavelength, 550, 30)[None, :]
    + rng.normal(0, 0.05, (n_samples, n_wavelengths))
)

# SNV 预处理：每条光谱减均值、除标准差
spectra_snv = (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(
    axis=1, keepdims=True
)

for n_components in (2, 3, 5, 8):
    pls = PLSRegression(n_components=n_components)
    y_pred = cross_val_predict(pls, spectra_snv, concentration, cv=5)
    rmse = mean_squared_error(concentration, y_pred) ** 0.5
    print(f"n_components={n_components}: CV RMSE = {rmse:.3f}")
