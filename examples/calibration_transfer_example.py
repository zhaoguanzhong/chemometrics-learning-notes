"""模型转移示例：主仪器模型拿到从仪器上用。

模拟两台仪器测同一批样品：
- 主仪器：两个高斯吸收峰，峰位在 3.0 和 7.0；
- 从仪器：峰位漂移 +0.2，增益 1.15，再叠加基线偏移。

先在主仪器校正集上训练 PLS，然后对比三种做法
在从仪器测试集上的表现：
1. 直接预测（不做转移）；
2. 斜率 / 截距修正（用转移样品的参考值修正预测值）；
3. 直接标准化 DS（用转移样品拟合从仪器光谱到主仪器
   光谱的岭回归变换，再用旧 PLS 预测）。

转移样品与验证样品严格分开：转移矩阵和斜率/截距
都在转移样品上拟合，效果只在独立测试集上报告。

运行：python examples/calibration_transfer_example.py
依赖：numpy, scikit-learn
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def make_spectra(conc1, conc2, x, peak_shift=0.0, gain=1.0,
                 baseline=0.0, seed=0):
    rng = np.random.default_rng(seed)
    peak1 = np.exp(-0.5 * ((x - (3.0 + peak_shift)) / 0.8) ** 2)
    peak2 = np.exp(-0.5 * ((x - (7.0 + peak_shift)) / 1.2) ** 2)
    spectra = conc1[:, None] * peak1[None, :] + conc2[:, None] * peak2[None, :]
    spectra = gain * spectra + baseline
    spectra += rng.normal(0, 0.05, size=spectra.shape)
    return spectra


def report(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{name}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")
    return rmse, r2


def main():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 200)

    def draw(n):
        conc1 = rng.uniform(0, 10, size=n)
        conc2 = rng.uniform(0, 10, size=n)
        target = 2.0 * conc1 + 0.5 * conc2
        return conc1, conc2, target

    # 主仪器校正集：训练 PLS 模型
    c1, c2, y_cal = draw(200)
    X_master_cal = make_spectra(c1, c2, x, seed=1)
    pls = PLSRegression(n_components=5)
    pls.fit(X_master_cal, y_cal)

    # 转移样品：两台仪器都测过（拟合转移用，不参与效果验证）
    c1_t, c2_t, y_transfer = draw(40)
    X_master_t = make_spectra(c1_t, c2_t, x, seed=2)
    X_slave_t = make_spectra(c1_t, c2_t, x, peak_shift=0.2, gain=1.15,
                             baseline=0.3, seed=3)

    # 独立测试集：主、从仪器各一份
    c1_te, c2_te, y_test = draw(100)
    X_master_te = make_spectra(c1_te, c2_te, x, seed=4)
    X_slave_te = make_spectra(c1_te, c2_te, x, peak_shift=0.2, gain=1.15,
                              baseline=0.3, seed=5)

    report("Master test (reference)", y_test,
           pls.predict(X_master_te).ravel())
    pred_slave_raw = pls.predict(X_slave_te).ravel()
    report("Slave test, no transfer", y_test, pred_slave_raw)

    # 做法 2：斜率 / 截距修正（在转移样品上拟合）
    pred_slave_transfer = pls.predict(X_slave_t).ravel()
    slope_bias = LinearRegression().fit(
        pred_slave_transfer.reshape(-1, 1), y_transfer)
    pred_corrected = slope_bias.predict(pred_slave_raw.reshape(-1, 1))
    report("Slave test, slope/bias correction", y_test, pred_corrected)

    # 做法 3：直接标准化 DS（岭回归拟合从仪器光谱到主仪器光谱）
    ds = Ridge(alpha=10.0)
    ds.fit(X_slave_t, X_master_t)
    X_slave_te_mapped = ds.predict(X_slave_te)
    report("Slave test, direct standardization (DS)", y_test,
           pls.predict(X_slave_te_mapped).ravel())


if __name__ == "__main__":
    main()
