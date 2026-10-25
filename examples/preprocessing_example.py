"""预处理对比示例：原始光谱 vs SNV，看 PLS 交叉验证误差的变化。

合成数据中混入了每条光谱整体的基线偏移与尺度差异，
SNV 应当能消除这部分干扰、降低预测误差。
运行：pip install numpy scikit-learn && python preprocessing_example.py
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(2)

n_samples, n_wavelengths = 80, 200
wavelength = np.linspace(400, 1000, n_wavelengths)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


concentration = rng.uniform(0, 10, n_samples)
clean = (
    concentration[:, None] * gaussian(wavelength, 650, 40)[None, :]
    + (10 - concentration)[:, None] * gaussian(wavelength, 550, 30)[None, :]
)
# 混入基线偏移与尺度差异（模拟真实测量干扰）
baseline = rng.normal(0, 0.5, (n_samples, 1))
scale = rng.normal(1, 0.2, (n_samples, 1))
spectra = clean * scale + baseline + rng.normal(0, 0.05, (n_samples, n_wavelengths))
spectra_snv = (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(
    axis=1, keepdims=True
)

for name, X in (("raw", spectra), ("SNV", spectra_snv)):
    pls = PLSRegression(n_components=3)
    y_pred = cross_val_predict(pls, X, concentration, cv=5)
    rmse = mean_squared_error(concentration, y_pred) ** 0.5
    print(f"{name}: CV RMSE = {rmse:.3f}")
