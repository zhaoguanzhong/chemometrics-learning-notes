"""支持向量回归（SVR）示例：合成光谱上的非线性定量。

构造一个浓度与光谱呈非线性关系的数据集：目标组分的吸收峰
强度随浓度饱和式增长（类似强吸收下的偏离线性），再叠加
基线漂移与噪声。用 RBF 核 SVR 建模，并与线性 PLS 对比，
展示非线性关系下 SVR 的优势。

运行：python examples/svr_example.py
依赖：numpy, scikit-learn
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_spectra(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    # 波长轴 200 个点，两处高斯吸收峰
    x = np.linspace(0, 10, 200)
    peak1 = np.exp(-0.5 * ((x - 3.0) / 0.8) ** 2)
    peak2 = np.exp(-0.5 * ((x - 7.0) / 1.2) ** 2)
    conc = rng.uniform(0.5, 10.0, size=n_samples)
    # 非线性响应：峰高随浓度饱和式增长 1 - exp(-c/4)，而非正比于 c
    response = 1.0 - np.exp(-conc / 4.0)
    spectra = response[:, None] * peak1[None, :] + 0.5 * peak2[None, :]
    # 基线漂移与噪声
    spectra += rng.normal(0, 0.05, size=(n_samples, 1)) * x[None, :]
    spectra += rng.normal(0, 0.02, size=spectra.shape)
    return spectra, conc


def report(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")
    return rmse, r2


def main():
    spectra, conc = make_spectra(240)
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, conc, test_size=0.25, random_state=42
    )

    # 线性对照：PLS（5 个主成分）
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, y_train)
    report("PLS (linear)", y_test, pls.predict(X_test).ravel())

    # SVR：先标准化，再用网格交叉验证选 C / epsilon / gamma
    pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    grid = GridSearchCV(
        pipe,
        param_grid={
            "svr__C": [1, 10, 100],
            "svr__epsilon": [0.1, 0.5],
            "svr__gamma": ["scale", 0.01],
        },
        cv=5,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    print("SVR best params:", grid.best_params_)
    report("SVR (RBF)", y_test, grid.predict(X_test))


if __name__ == "__main__":
    main()
