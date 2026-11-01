"""PLS-DA 示例：用偏最小二乘做两类光谱分类。

合成数据：60 个波长点，训练集 100 条、测试集 50 条光谱。
- 两类共享一个位于约 30 处的背景峰；
- A 类在约 18 处多一个特征峰，B 类在约 44 处多一个特征峰，
  峰强度在类内有波动；
- 再加与类别无关的小幅基线漂移与高斯噪声。

方法（见 notes/17-pls-da.md）：
- 把类别编成哑变量 y（A 类记 0，B 类记 1），做 PLS1 回归；
- 潜变量个数先用训练集 5 折交叉验证选（看分类准确率），
  再在全部训练集上重拟合、到测试集上预测；
- 预测值以 0.5 为阈值判类：低于 0.5 判 A 类，高于 0.5 判 B 类。

注意：哑变量回归的预测值不是概率，0.5 只是本例两类
均衡时的常用阈值；类别不平衡或错判代价不对称时，
阈值要按实际需要另选（见笔记）。合成数据演示，
真实分类问题须用独立样品检验。

只依赖 numpy，可直接运行：python examples/pls_da_example.py
"""

import numpy as np

rng = np.random.default_rng(11)


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


n_wl = 60
wl = np.arange(n_wl)
peak_shared = gauss(wl, 30, 6.0)   # 两类共有的背景峰
peak_a = gauss(wl, 18, 4.0)        # A 类特征峰
peak_b = gauss(wl, 44, 4.0)        # B 类特征峰


def make_spectra(n_per_class):
    """每类生成 n_per_class 条光谱，返回 X、类别标签（0=A，1=B）。"""
    xs, ys = [], []
    for label, peak in ((0, peak_a), (1, peak_b)):
        n = n_per_class
        strength = rng.normal(loc=1.0, scale=0.25, size=n)
        drift = rng.normal(scale=0.10, size=n)
        X = (np.outer(np.ones(n), peak_shared)
             + np.outer(strength, peak)
             + np.outer(drift, np.ones(n_wl))
             + rng.normal(scale=0.08, size=(n, n_wl)))
        xs.append(X)
        ys.append(np.full(n, label))
    return np.vstack(xs), np.concatenate(ys)


X_train, y_train = make_spectra(50)
X_test, y_test = make_spectra(25)


def pls1_fit(X_tr, y_tr, n_comp):
    """NIPALS 式 PLS1，返回回归系数与均值，供预测使用。"""
    x_mean, y_mean = X_tr.mean(axis=0), y_tr.mean()
    Xc, yc = X_tr - x_mean, y_tr - y_mean
    W, P, Q = [], [], []
    X_work, y_work = Xc.copy(), yc.copy()
    for _ in range(n_comp):
        w = X_work.T @ y_work
        w = w / np.linalg.norm(w)
        t = X_work @ w
        q = float((t @ y_work) / (t @ t))
        p = (X_work.T @ t) / (t @ t)
        W.append(w)
        P.append(p)
        Q.append(q)
        X_work = X_work - np.outer(t, p)
        y_work = y_work - t * q
    W = np.column_stack(W)
    P = np.column_stack(P)
    q_vec = np.array(Q)
    coef = W @ np.linalg.inv(P.T @ W) @ q_vec
    return x_mean, y_mean, coef


def pls1_predict(model, X):
    x_mean, y_mean, coef = model
    return y_mean + (X - x_mean) @ coef


def accuracy(y_pred_label, y_true):
    return float(np.mean(y_pred_label == y_true))


def confusion(y_pred_label, y_true):
    """返回 [[A 判对，A 判错]，[B 判错，B 判对]]（行=真实类）。"""
    tp_a = int(np.sum((y_true == 0) & (y_pred_label == 0)))
    fn_a = int(np.sum((y_true == 0) & (y_pred_label == 1)))
    fn_b = int(np.sum((y_true == 1) & (y_pred_label == 0)))
    tp_b = int(np.sum((y_true == 1) & (y_pred_label == 1)))
    return [[tp_a, fn_a], [fn_b, tp_b]]


# 1）训练集 5 折交叉验证：比较 1~4 个潜变量的分类准确率
folds = np.array_split(rng.permutation(len(y_train)), 5)
print("5 折交叉验证（训练集）各潜变量数的准确率：")
cv_scores = {}
for n_comp in (1, 2, 3, 4):
    fold_acc = []
    for fold in folds:
        mask = np.ones(len(y_train), dtype=bool)
        mask[fold] = False
        model = pls1_fit(X_train[mask], y_train[mask], n_comp)
        pred = (pls1_predict(model, X_train[fold]) >= 0.5).astype(int)
        fold_acc.append(accuracy(pred, y_train[fold]))
    cv_scores[n_comp] = float(np.mean(fold_acc))
    print(f"  潜变量数 {n_comp}：{cv_scores[n_comp]:.3f}")
best_comp = max(cv_scores, key=lambda k: (cv_scores[k], -k))
print(f"交叉验证选出的潜变量数：{best_comp}")

# 2）全部训练集重拟合，到测试集预测
model = pls1_fit(X_train, y_train, best_comp)
y_score = pls1_predict(model, X_test)
y_pred = (y_score >= 0.5).astype(int)
print(f"测试集（每类 25 条）：准确率 {accuracy(y_pred, y_test):.3f}")
print(f"混淆矩阵 [[A 判对，A 判错]，[B 判错，B 判对]]：{confusion(y_pred, y_test)}")
print(f"测试集预测值范围：{y_score.min():.3f} ~ {y_score.max():.3f}")
print("结论：两类特征峰位置不同，PLS-DA 在潜变量空间里把类别"
      "分开；但预测值不是概率，阈值 0.5 只适用于本例两类均衡的情形。")
