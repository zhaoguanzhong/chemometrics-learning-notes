"""LDA 分类示例：用合成光谱区分两类样本。

两类样本的光谱峰位不同（A 类 600 nm 强，B 类 700 nm 强）。
高维光谱先 PLS 降维再接 LDA，避免协方差矩阵不可逆。
运行：pip install numpy scikit-learn && python lda_example.py
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(1)

n_samples, n_wavelengths = 100, 200
wavelength = np.linspace(400, 1000, n_wavelengths)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


labels = np.array([0] * 50 + [1] * 50)
spectra = np.where(
    labels[:, None] == 0,
    gaussian(wavelength, 600, 40)[None, :] + rng.normal(0, 0.05, (n_samples, n_wavelengths)),
    gaussian(wavelength, 700, 40)[None, :] + rng.normal(0, 0.05, (n_samples, n_wavelengths)),
)

# 先 PLS 降维到 3 个潜变量，再 LDA 分类
pls = PLSRegression(n_components=3)
scores = pls.fit_transform(spectra, labels)[0]
pred = cross_val_predict(LinearDiscriminantAnalysis(), scores, labels, cv=5)
print(f"LDA on PLS scores: accuracy = {accuracy_score(labels, pred):.3f}")
