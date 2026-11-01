"""OPLS 示例：把与目标无关的正交变异过滤掉，再做 PLS。

合成数据：50 个波长点，训练集 80 条、测试集 40 条光谱。
- 预测信号：峰在约 15 处的高斯峰，强度由潜变量 u 决定，
  目标 y = 2*u + 小噪声；
- 正交变异：峰在约 38 处的宽峰，强度由与 y 无关的潜变量 v
  决定，且幅度比预测信号大（v 的标准差为 1.5）；
- 再加小幅高斯噪声。

方法（OPLS-1，见 notes/16-opls.md）：
- 先算 PLS 权重 w 与载荷 p，把 p 中与 w 平行的部分去掉，
  得到正交权重 wo、正交得分 to 与正交载荷 po；
- 从 X 中减去 to * po，得到过滤后的 X，再做 PLS1，
  得到预测成分；测试集用同样的 wo、po 先过滤再预测；
- 对照组是直接在原始 X 上做的 PLS1。

注意：这里的过滤参数是在训练集上算的；正式做交叉验证时，
过滤必须放在每一折的训练集内部重算，不能先过滤全集再分组，
否则验证集信息会漏进训练集（见笔记）。

只依赖 numpy，可直接运行：python examples/opls_example.py
"""

import numpy as np

rng = np.random.default_rng(7)


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


n_wl = 50
wl = np.arange(n_wl)
peak_predictive = gauss(wl, 15, 4.0)   # 预测峰：与 y 有关
peak_orthogonal = gauss(wl, 38, 7.0)   # 正交峰：与 y 无关，幅度更大


def make_spectra(n):
    """生成 n 条合成光谱及其目标值。"""
    u = rng.normal(size=n)              # 预测潜变量
    v = rng.normal(size=n) * 1.5        # 正交潜变量，幅度更大
    y = 2.0 * u + rng.normal(scale=0.3, size=n)
    X = (np.outer(u, peak_predictive) + np.outer(v, peak_orthogonal)
         + rng.normal(scale=0.05, size=(n, n_wl)))
    return X, y


X_train, y_train = make_spectra(80)
X_test, y_test = make_spectra(40)


def pls1_fit_predict(X_tr, y_tr, X_te):
    """一个成分的 PLS 回归，返回预测值、权重与载荷。"""
    x_mean, y_mean = X_tr.mean(axis=0), y_tr.mean()
    Xc, yc = X_tr - x_mean, y_tr - y_mean
    w = Xc.T @ yc
    w = w / np.linalg.norm(w)
    t = Xc @ w
    q = (t @ yc) / (t @ t)
    p_load = (Xc.T @ t) / (t @ t)
    y_pred = y_mean + (X_te - x_mean) @ w * q
    return y_pred, w, p_load


def opls1_fit_predict(X_tr, y_tr, X_te):
    """一个正交成分 + 一个预测成分的 OPLS 回归。"""
    x_mean, y_mean = X_tr.mean(axis=0), y_tr.mean()
    Xc, yc = X_tr - x_mean, y_tr - y_mean
    # 先算 PLS 权重与载荷，用来找正交方向
    w = Xc.T @ yc
    w = w / np.linalg.norm(w)
    t = Xc @ w
    p_load = (Xc.T @ t) / (t @ t)
    # 正交权重：载荷中去掉与 w 平行的部分
    w_orth = p_load - (w @ p_load) / (w @ w) * w
    w_orth = w_orth / np.linalg.norm(w_orth)
    t_orth = Xc @ w_orth
    p_orth = (Xc.T @ t_orth) / (t_orth @ t_orth)
    # 过滤：从训练集 X 中减去正交成分
    X_filtered = Xc - np.outer(t_orth, p_orth)
    # 在过滤后的 X 上做 PLS1，得到预测成分
    w_pred = X_filtered.T @ yc
    w_pred = w_pred / np.linalg.norm(w_pred)
    t_pred = X_filtered @ w_pred
    q_pred = (t_pred @ yc) / (t_pred @ t_pred)
    p_pred = (X_filtered.T @ t_pred) / (t_pred @ t_pred)
    # 测试集：用同样的正交权重与载荷先过滤，再预测
    X_te_c = X_te - x_mean
    t_orth_te = X_te_c @ w_orth
    X_te_filtered = X_te_c - np.outer(t_orth_te, p_orth)
    y_pred = y_mean + X_te_filtered @ w_pred * q_pred
    return y_pred, w_pred, p_pred, w_orth, p_orth, t_orth, t_pred


def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2_score(y_pred, y_true):
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


y_pls, _, p_pls = pls1_fit_predict(X_train, y_train, X_test)
(y_opls, _, p_opls, _, p_orth,
 t_orth, t_pred) = opls1_fit_predict(X_train, y_train, X_test)

print(f"PLS1  测试集 RMSE {rmse(y_pls, y_test):.3f}  R2 {r2_score(y_pls, y_test):.3f}")
print(f"OPLS  测试集 RMSE {rmse(y_opls, y_test):.3f}  R2 {r2_score(y_opls, y_test):.3f}")
print(f"训练集：预测得分与 y 的相关系数 {corr(t_pred, y_train):.3f}")
print(f"训练集：正交得分与 y 的相关系数 {corr(t_orth, y_train):.3f}")
print(f"PLS1 载荷与真实预测峰的相关系数 {corr(p_pls, peak_predictive):.3f}")
print(f"OPLS 预测载荷与真实预测峰的相关系数 {corr(p_opls, peak_predictive):.3f}")
print(f"OPLS 正交载荷与真实正交峰的相关系数 {corr(p_orth, peak_orthogonal):.3f}")
print(f"训练集得分方差：正交成分 {np.var(t_orth):.2f}，预测成分 {np.var(t_pred):.2f}")
print("结论：正交变异幅度更大却与 y 无关；OPLS 精度只略好于 PLS1，"
      "真正的收益是预测载荷更干净、正交载荷单独对应无关峰。")
