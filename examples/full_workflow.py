"""完整工作流示例：从原始光谱到验证报告，一条龙跑通。

流程：合成含基线漂移与异常样本的光谱
  → SNV 预处理
  → PCA 做异常检测并剔除 Q 最大的样本
  → 交叉验证选 PLS 主成分数
  → 独立测试集上报告 RMSEP / R² / bias

运行：pip install numpy scikit-learn && python full_workflow.py
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(4)

n_samples, n_wavelengths = 120, 200
wavelength = np.linspace(400, 1000, n_wavelengths)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


concentration = rng.uniform(0, 10, n_samples)
clean = (
    concentration[:, None] * gaussian(wavelength, 650, 40)[None, :]
    + (10 - concentration)[:, None] * gaussian(wavelength, 550, 30)[None, :]
)
spectra = (
    clean * rng.normal(1, 0.2, (n_samples, 1))
    + rng.normal(0, 0.5, (n_samples, 1))
    + rng.normal(0, 0.05, (n_samples, n_wavelengths))
)
# 混入 5 个峰位偏移的异常样本
spectra[-5:] = (
    concentration[-5:][:, None] * gaussian(wavelength, 710, 40)[None, :]
    + rng.normal(0, 0.05, (5, n_wavelengths))
)

# 1. SNV 预处理
spectra_snv = (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(
    axis=1, keepdims=True
)

# 2. PCA 异常检测：剔除 Q 残差最大的 5 个样本
pca = PCA(n_components=3)
scores = pca.fit_transform(spectra_snv)
q = np.sum((spectra_snv - pca.inverse_transform(scores)) ** 2, axis=1)
keep = np.argsort(q)[: n_samples - 5]
print(f"剔除样本编号：{sorted(set(range(n_samples)) - set(keep))}")
X, y = spectra_snv[keep], concentration[keep]

# 3. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 4. 交叉验证选主成分数
best_n, best_rmse = None, float("inf")
for n_components in (2, 3, 4, 5, 6):
    pls = PLSRegression(n_components=n_components)
    y_cv = cross_val_predict(pls, X_train, y_train, cv=5)
    rmse = mean_squared_error(y_train, y_cv) ** 0.5
    print(f"n_components={n_components}: CV RMSE = {rmse:.3f}")
    if rmse < best_rmse:
        best_n, best_rmse = n_components, rmse

# 5. 测试集报告
pls = PLSRegression(n_components=best_n).fit(X_train, y_train)
y_pred = pls.predict(X_test).ravel()
print(f"\n选定主成分数：{best_n}")
print(f"测试集 RMSEP = {mean_squared_error(y_test, y_pred) ** 0.5:.3f}")
print(f"测试集 R² = {r2_score(y_test, y_pred):.3f}")
print(f"测试集 bias = {np.mean(y_pred - y_test):.3f}")
