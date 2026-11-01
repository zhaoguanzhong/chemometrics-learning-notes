"""实验设计（DoE）示例：中心复合设计 + 二次响应面模型。

合成数据（非实测数据，仅用于演示设计与拟合）：
  - 两个编码因素 x1（温度）、x2（pH），范围均为 -1.414 ~ +1.414；
  - 真实响应面：y = 60 + 10*x1 + 6*x2 - 8*x1**2 - 4*x2**2 + 5*x1*x2；
  - 噪声为标准差 1.0 的高斯噪声，随机种子固定为 42。

设计：中心复合设计（CCD），共 11 次实验
  - 4 个析因角点 (±1, ±1)；
  - 4 个轴向点 (±1.414, 0)、(0, ±1.414)；
  - 3 个中心点 (0, 0) 重复，用于估计纯误差。

拟合：对 [1, x1, x2, x1*x2, x1**2, x2**2] 做最小二乘，
报告估计系数与真实系数的对照、R²、中心点纯误差，
并在网格上搜索预测响应最大的点，再用真实模型验证
该点的真实响应（真实实验里这一步必须换成验证实验）。

另附 OFAT 对照：只沿坐标轴变化时，交互作用项 x1*x2
恒为 0，无法估计——这正是笔记里说的 OFAT 看不到交互作用。

运行：pip install numpy
      python examples/doe_example.py
"""

import numpy as np

rng = np.random.default_rng(42)
ALPHA = np.sqrt(2)  # 可旋转设计的轴向点距离


def true_response(x1, x2):
    """合成真实响应面（仅用于生成数据和事后验证，拟合时不知道它）。"""
    return 60 + 10 * x1 + 6 * x2 - 8 * x1**2 - 4 * x2**2 + 5 * x1 * x2


def design_matrix(x1, x2):
    """二次模型的设计矩阵：[1, x1, x2, x1*x2, x1**2, x2**2]。"""
    return np.column_stack(
        [np.ones_like(x1), x1, x2, x1 * x2, x1**2, x2**2]
    )


def main():
    # 1. 中心复合设计：角点 + 轴向点 + 中心点重复
    factorial = np.array(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float
    )
    axial = np.array(
        [[-ALPHA, 0], [ALPHA, 0], [0, -ALPHA], [0, ALPHA]], dtype=float
    )
    centre = np.zeros((3, 2))
    design = np.vstack([factorial, axial, centre])
    x1, x2 = design[:, 0], design[:, 1]

    y = true_response(x1, x2) + rng.normal(0, 1.0, size=len(design))
    print(f"设计实验数：{len(design)}（角点 4 + 轴向点 4 + 中心点 3）")

    # 2. 最小二乘拟合二次模型
    X = design_matrix(x1, x2)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    names = ["截距", "x1", "x2", "x1*x2", "x1^2", "x2^2"]
    true_coef = [60, 10, 6, 5, -8, -4]
    print("估计系数 vs 真实系数：")
    for name, est, true in zip(names, coef, true_coef):
        print(f"  {name:>5}：估计 {est:7.2f}  真实 {true:7.2f}")
    print(f"拟合 R² = {r2:.3f}")

    # 3. 中心点纯误差：3 个中心点重复的波动就是实验噪声的估计
    centre_y = y[-3:]
    print(
        f"中心点响应：{np.round(centre_y, 2)}，"
        f"标准差 {centre_y.std(ddof=1):.2f}（真实噪声标准差 1.0）"
    )

    # 4. 网格搜索预测最优点，再用真实模型验证（真实实验里换成验证实验）
    grid = np.linspace(-ALPHA, ALPHA, 57)
    gx1, gx2 = np.meshgrid(grid, grid)
    pred = design_matrix(gx1.ravel(), gx2.ravel()) @ coef
    best = int(np.argmax(pred))
    opt_x1, opt_x2 = gx1.ravel()[best], gx2.ravel()[best]
    print(
        f"预测最优点：x1 = {opt_x1:.2f}，x2 = {opt_x2:.2f}，"
        f"预测响应 {pred[best]:.2f}，"
        f"真实响应 {true_response(opt_x1, opt_x2):.2f}"
    )
    print(
        "真实最优点（求导可得 x1 约 1.07、x2 约 1.42，"
        "恰在轴向点边界附近；超出设计范围不可盲目外推）。"
    )

    # 5. OFAT 对照：只沿坐标轴走，交互作用项恒为 0，无法估计
    ofat = np.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float
    )
    print(
        "OFAT 对照：5 个沿轴点上的 x1*x2 全为 "
        f"{np.unique(ofat[:, 0] * ofat[:, 1])}，交互作用不可识别；"
        "且 OFAT 从未同时往两个方向走，找不到带交互的峰值。"
    )


if __name__ == "__main__":
    main()
