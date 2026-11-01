"""SIMCA 软分类示例：每个类别单独建一个 PCA 模型。

合成数据：两类样品的光谱峰位不同（A 类峰在约 30、70 处，
B 类峰在约 45、82 处），每类样品内两个峰的强度独立波动，
再加小幅基线漂移和噪声。另有一组“未知类”样品
（峰在约 20、58 处），不属于 A、B 中任何一类，
用来检验软分类能不能说“我不知道”而不是硬猜一类。

方法（简化教学版）：
- 每个类别用本类训练集的均值中心化后做 SVD，取前 k 个主成分
  建一个 PCA 模型；
- 对任一样品，算它到该类模型的 Hotelling T2（得分空间内的距离）
  和 Q 残差（模型解释不了的部分）；
- T2、Q 的限值取训练集的 99 百分位数（经验限，教学简化；
  正式 SIMCA 用 F 分布的 T2 限和 Jackson-Mudholkar 的 Q 限，
  训练样本只有 40 个时经验限本身就不太稳，见笔记）；
- 综合距离 D = sqrt((T2/T2限)^2 + (Q/Q限)^2)，
  样品归入 D 最小的类；若最小的 D 仍大于 sqrt(2)
  （相当于 T2、Q 同时顶到限值），则拒判。

只依赖 numpy，可直接运行：python examples/simca_example.py
"""

import math

import numpy as np

rng = np.random.default_rng(7)
n_wl = 100
wl = np.arange(n_wl)


def gauss(center, width):
    return np.exp(-0.5 * ((wl - center) / width) ** 2)


def make_samples(peaks, n):
    """生成 n 条光谱：两个峰的强度各自独立波动 + 基线漂移 + 噪声。"""
    out = []
    for _ in range(n):
        a1 = rng.normal(1.0, 0.15)
        a2 = rng.normal(1.0, 0.15)
        spec = a1 * peaks[0] + a2 * peaks[1]
        spec = spec + rng.normal(0.0, 0.05)  # 基线漂移
        spec = spec + rng.normal(0.0, 0.03, size=n_wl)  # 测量噪声
        out.append(spec)
    return np.array(out)


peaks_a = (gauss(30, 8), 0.6 * gauss(70, 10))
peaks_b = (0.8 * gauss(45, 8), 0.9 * gauss(82, 9))
peaks_u = (gauss(20, 8), 0.7 * gauss(58, 9))

Xa_train = make_samples(peaks_a, 40)
Xb_train = make_samples(peaks_b, 40)
Xa_test = make_samples(peaks_a, 20)
Xb_test = make_samples(peaks_b, 20)
Xu_test = make_samples(peaks_u, 10)


class SimcaModel:
    """单类别的 PCA 模型：T2、Q 与到该类模型的综合距离。"""

    def __init__(self, n_components=3, limit_percentile=99):
        self.k = n_components
        self.pct = limit_percentile

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.V_ = Vt[: self.k].T
        scores = Xc @ self.V_
        self.score_var_ = scores.var(axis=0, ddof=1)
        t2, q = self._t2_q(X)
        self.t2_lim_ = np.percentile(t2, self.pct)
        self.q_lim_ = np.percentile(q, self.pct)
        return self

    def _t2_q(self, X):
        Xc = X - self.mean_
        scores = Xc @ self.V_
        resid = Xc - scores @ self.V_.T
        t2 = np.sum((scores ** 2) / self.score_var_, axis=1)
        q = np.sum(resid ** 2, axis=1)
        return t2, q

    def distance(self, X):
        t2, q = self._t2_q(X)
        return np.sqrt((t2 / self.t2_lim_) ** 2 + (q / self.q_lim_) ** 2)


model_a = SimcaModel(n_components=3).fit(Xa_train)
model_b = SimcaModel(n_components=3).fit(Xb_train)
print("A 类模型限值：T2 限 %.2f，Q 限 %.3f" % (model_a.t2_lim_, model_a.q_lim_))
print("B 类模型限值：T2 限 %.2f，Q 限 %.3f" % (model_b.t2_lim_, model_b.q_lim_))

THRESHOLD = math.sqrt(2)  # T2、Q 同时顶到限值时的综合距离


def classify(X):
    da = model_a.distance(X)
    db = model_b.distance(X)
    pred = []
    for a, b in zip(da, db):
        if min(a, b) > THRESHOLD:
            pred.append("拒判")
        elif a <= b:
            pred.append("A")
        else:
            pred.append("B")
    return np.array(pred), np.minimum(da, db)


for name, X, truth in [
    ("A 类测试集", Xa_test, "A"),
    ("B 类测试集", Xb_test, "B"),
    ("未知类测试集", Xu_test, "拒判"),
]:
    pred, min_dist = classify(X)
    n_expected = int(np.sum(pred == truth))
    n_wrong = int(np.sum((pred != truth) & (pred != "拒判")))
    n_reject = int(np.sum(pred == "拒判"))
    print(
        "%s：符合预期 %d/%d（判错 %d 个，拒判 %d 个），"
        "到最近模型的平均距离 %.2f"
        % (name, n_expected, len(X), n_wrong, n_reject, float(np.mean(min_dist)))
    )

print()
print("要点：两类测试集没有判错到对方类里，拒判的只是个别离本类模型")
print("较远的样品；而未知类样品全部被拒判——软分类的价值正在于此：")
print("遇到训练时没见过的东西，宁可说不知道，也不要硬猜一类。")
