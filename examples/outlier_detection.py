"""异常样本检测示例：PCA 的 Hotelling T² 与 Q 残差。

正常样本是两个高斯峰的组合；混入 5 个异常样本（峰形整体偏移）。
用 PCA 建模后，计算每个样本的 T² 与 Q，异常样本应当在 Q 上显著偏大。
运行：pip install numpy scikit-learn && python outlier_detection.py
"""

import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(3)

n_normal, n_outlier, n_wavelengths = 80, 5, 200
wavelength = np.linspace(400, 1000, n_wavelengths)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


base = (
    rng.uniform(0, 10, n_normal)[:, None] * gaussian(wavelength, 650, 40)[None, :]
    + rng.uniform(0, 10, n_normal)[:, None] * gaussian(wavelength, 550, 30)[None, :]
    + rng.normal(0, 0.05, (n_normal, n_wavelengths))
)
# 异常样本：峰位整体偏移 60 nm，模拟仪器波长漂移或污染样品
outliers = (
    rng.uniform(0, 10, n_outlier)[:, None] * gaussian(wavelength, 710, 40)[None, :]
    + rng.uniform(0, 10, n_outlier)[:, None] * gaussian(wavelength, 610, 30)[None, :]
    + rng.normal(0, 0.05, (n_outlier, n_wavelengths))
)
spectra = np.vstack([base, outliers])

pca = PCA(n_components=3)
scores = pca.fit_transform(spectra)
reconstructed = pca.inverse_transform(scores)

# Hotelling T²：得分按各主成分方差归一化后的平方和
t2 = np.sum((scores / np.sqrt(pca.explained_variance_)) ** 2, axis=1)
# Q 残差：重建误差的平方和
q = np.sum((spectra - reconstructed) ** 2, axis=1)

print(f"正常样本 Q 均值：{q[:n_normal].mean():.3f}")
print(f"异常样本 Q 均值：{q[n_normal:].mean():.3f}")
print(f"正常样本 T² 均值：{t2[:n_normal].mean():.2f}")
print(f"异常样本 T² 均值：{t2[n_normal:].mean():.2f}")
print(f"Q 最大的 5 个样本编号：{np.argsort(q)[-5:]}（真实异常编号：{list(range(n_normal, n_normal + n_outlier))}）")
