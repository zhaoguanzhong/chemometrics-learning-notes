"""随机森林示例：合成光谱上的非线性定量。

构造两个组分的光谱：目标组分的吸收峰强度随浓度饱和式增长
（类似强吸收下的偏离线性），再叠加一个弱干扰组分、
基线漂移与噪声。目标值主要由目标组分决定，
干扰组分只贡献一小部分。用随机森林建模，并与线性 PLS 对比，
展示非线性关系下集成方法的优势；最后按波段汇总变量重要性，
说明相关波长会把重要性分摊成一片区域。

运行：python examples/random_forest_example.py
依赖：numpy, scikit-learn
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def make_spectra(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    # 波长轴 200 个点，两处高斯吸收峰
    x = np.linspace(0, 10, 200)
    peak1 = np.exp(-0.5 * ((x - 3.0) / 0.8) ** 2)
    peak2 = np.exp(-0.5 * ((x - 7.0) / 1.2) ** 2)
    conc1 = rng.uniform(0, 10, size=n_samples)
    conc2 = rng.uniform(0, 10, size=n_samples)
    spectra = conc1[:, None] * peak1[None, :] + conc2[:, None] * peak2[None, :]
    # 非线性目标：目标组分饱和响应 + 弱干扰组分的线性贡献
    target = 10.0 * (1.0 - np.exp(-conc1 / 4.0)) + 0.3 * conc2
    # 基线漂移与噪声
    spectra += rng.normal(0, 0.05, size=(n_samples, 1)) * x[None, :]
    spectra += rng.normal(0, 0.05, size=spectra.shape)
    return spectra, target, x


def report(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")
    return rmse, r2


def main():
    spectra, target, x = make_spectra(300)
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, target, test_size=0.25, random_state=42
    )

    # 线性对照：PLS（5 个主成分）
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, y_train)
    report("PLS (linear)", y_test, pls.predict(X_test).ravel())

    # 随机森林：300 棵树，袋外分数作为内部泛化估计
    rf = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, oob_score=True
    )
    rf.fit(X_train, y_train)
    print(f"Random forest OOB R2 = {rf.oob_score_:.3f}")
    report("Random forest", y_test, rf.predict(X_test))

    # 变量重要性按波段汇总：相邻相关波长会分摊重要性，
    # 因此看区域合计比看单个波长更有意义
    importance = rf.feature_importances_
    regions = [
        ("peak1 region (x in [2, 4])", (x >= 2) & (x <= 4)),
        ("peak2 region (x in [6, 8])", (x >= 6) & (x <= 8)),
        ("background region", (x < 1.5) | (x > 8.5)),
    ]
    for label, mask in regions:
        print(
            f"Importance {label}: "
            f"total = {importance[mask].sum():.3f}, "
            f"mean per wavelength = {importance[mask].mean():.4f}"
        )


if __name__ == "__main__":
    main()
