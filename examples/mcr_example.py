"""MCR-ALS 示例：从三组分混合光谱中拆出纯组分。

合成数据（非实测数据，仅用于演示算法）：
  - 200 个波长点上的 3 条高斯纯光谱，峰位分别在 50、100、150；
  - 60 个混合样品，浓度来自 Dirichlet 分布（各组分含量之和为 1）；
  - 混合光谱 = 浓度 @ 纯光谱 + 标准差 0.01 的高斯噪声。

算法：最纯变量法（SIMPLISMA 思路）给出初始浓度估计，
然后交替非负最小二乘（NNLS）迭代，并把纯光谱归一化到
最大值为 1（消除尺度模糊）。评价只用合成数据已知的
真实纯光谱做相关系数对照——真实数据里没有这份“答案”，
必须改用已知纯光谱或加标样品做外部验证。

运行：pip install numpy scipy
      python examples/mcr_example.py
"""

import numpy as np
from scipy.optimize import nnls

rng = np.random.default_rng(42)
N_SAMPLES, N_WAVELENGTHS, N_COMPONENTS = 60, 200, 3


def gaussian(centre, width, height, x):
    """单峰高斯曲线，用作一条合成纯光谱。"""
    return height * np.exp(-0.5 * ((x - centre) / width) ** 2)


def purest_variables(X, n_components, min_gap=20):
    """最纯变量法：纯度 = 标准差 /（均值 + 偏移）。

    只属于一个组分的波长上，样品间的相对变化最大，纯度最高。
    挑出的波长彼此至少间隔 min_gap，避免重复选中同一组分。
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    offset = 0.05 * mean.max()
    purity = std / (mean + offset)
    chosen = []
    for j in np.argsort(-purity):
        if all(abs(int(j) - q) > min_gap for q in chosen):
            chosen.append(int(j))
        if len(chosen) == n_components:
            break
    return chosen


def solve_nnls_rows(A, B):
    """对 B 的每一行，用 NNLS 解 A @ x ≈ b，返回系数矩阵。"""
    out = np.zeros((B.shape[0], A.shape[1]))
    for i, b in enumerate(B):
        out[i], _ = nnls(A, b)
    return out


def main():
    wv = np.arange(N_WAVELENGTHS)
    # 真实纯光谱（仅用于生成数据和事后对照，算法本身不知道它们）
    S_true = np.column_stack(
        [
            gaussian(50, 12, 1.0, wv),
            gaussian(100, 18, 0.9, wv),
            gaussian(150, 10, 1.1, wv),
        ]
    )
    C_true = rng.dirichlet([2, 2, 2], size=N_SAMPLES)
    X = C_true @ S_true.T + rng.normal(0, 0.01, size=(N_SAMPLES, N_WAVELENGTHS))
    X = np.clip(X, 0, None)

    # 1. 组分数判断：SVD 奇异值在第 3 个之后出现明显断点
    singular_values = np.linalg.svd(X, compute_uv=False)
    print("SVD 前 6 个奇异值：", np.round(singular_values[:6], 2))

    # 2. 最纯变量法初始化浓度矩阵
    pure_idx = purest_variables(X, N_COMPONENTS)
    print("最纯变量波长点：", pure_idx)
    C = np.clip(X[:, pure_idx], 0, None)
    C = C / C.sum(axis=1, keepdims=True)

    # 3-4. 交替非负最小二乘，直到浓度相对变化小于 1e-4
    for iteration in range(200):
        S = solve_nnls_rows(C, X.T)  # 固定 C 解 S（逐波长，结果已是 p x k）
        scale = S.max(axis=0)
        S = S / scale  # 纯光谱归一化到最大值 1，尺度由 C 吸收
        C = C * scale
        C_new = solve_nnls_rows(S, X)  # 固定 S 解 C（逐样品）
        delta = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-9)
        C = C_new
        if delta < 1e-4:
            print(f"第 {iteration + 1} 次迭代后收敛（相对变化 {delta:.2e}）")
            break
    else:
        print("达到最大迭代次数 200，仍未收敛（需检查初始估计或组分数）")

    # 5. 评价：残差指标 + 拆出的纯光谱与真实纯光谱的相关系数
    residual = X - C @ S.T
    lof = 100 * np.sqrt((residual**2).sum() / (X**2).sum())
    r2 = 1 - (residual**2).sum() / ((X - X.mean()) ** 2).sum()
    print(f"LOF = {lof:.2f}%  R² = {r2:.3f}  （LOF 低不等于拆对，见笔记）")
    for a in range(N_COMPONENTS):
        cors = [np.corrcoef(S[:, a], S_true[:, b])[0, 1] for b in range(N_COMPONENTS)]
        best = int(np.argmax(cors))
        print(f"拆出组分 {a + 1}：与真实组分 {best + 1} 的相关系数 {cors[best]:.3f}")


if __name__ == "__main__":
    main()
