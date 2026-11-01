# 化学计量学学习笔记 | Chemometrics Learning Notes

面向初学者的化学计量学笔记：从线性模型、偏最小二乘（PLS）到线性判别分析（LDA），

以及光谱预处理与模型验证的常见做法。内容以概念讲解和可运行示例为主，持续补充中。

> 免责：本仓库为学习笔记，不是教科书或论文引用来源；公式与结论请以教材和原始文献为准。

## 目录

### 笔记

- `notes/01-linear-models.md` — 线性模型：y = Xβ + ε
- `notes/02-pls.md` — PLS 与 PLS-DA：为光谱而生的回归方法
- `notes/03-lda.md` — LDA：线性判别分析
- `notes/04-preprocessing.md` — 光谱预处理：SNV、MSC、一阶导数
- `notes/05-validation.md` — 模型验证：交叉验证与常见坑
- `notes/06-pca-and-pcr.md` — PCA 与主成分回归（PCR）
- `notes/07-evaluation-metrics.md` — 评价指标：RMSEP、R² 与偏差
- `notes/08-variable-selection.md` — 变量选择：VIP、CARS、SPA
- `notes/09-outlier-detection.md` — 异常样本检测：杠杆值、Q 残差、Hotelling T²
- `notes/10-svr.md` — 支持向量回归（SVR）：ε-不敏感损失与 RBF 核
- `notes/11-random-forest.md` — 随机森林与集成方法：Bagging、OOB、变量重要性与外推短板
- `notes/12-calibration-transfer.md` — 模型转移与校正维护：斜率/截距修正、DS、PDS 与定期复核
- `notes/13-mcr.md` — 多元曲线分辨（MCR）：不分离样品，直接拆出纯组分
- `notes/14-doe.md` — 实验设计（DoE）：析因设计、中心点、响应面与随机化
- `notes/15-simca.md` — SIMCA 软分类：每类一个 PCA 模型、T² 与 Q、拒判
- `notes/16-opls.md` — OPLS：正交过滤、预测成分与正交成分的分离与解读
- `notes/17-pls-da.md` — PLS-DA：哑变量编码、阈值、混淆矩阵与置换检验

### 示例（均可直接运行）

- `examples/pls_example.py` — 合成光谱上的 PLS 回归
- `examples/lda_example.py` — PLS 降维后接 LDA 的分类示例
- `examples/preprocessing_example.py` — 原始光谱 vs SNV 的预处理对比
- `examples/outlier_detection.py` — PCA 的 T² 与 Q 残差异常检测
- `examples/full_workflow.py` — 完整工作流：预处理 → 异常剔除 → 选主成分数 → 测试集报告
- `examples/svr_example.py` — RBF 核 SVR 回归（饱和响应光谱 + 网格选参与 PLS 对照）
- `examples/random_forest_example.py` — 随机森林回归（饱和响应光谱 + PLS 对照 + 按波段汇总变量重要性）
- `examples/calibration_transfer_example.py` — 模型转移示例（主、从仪器漂移模拟 + 斜率/截距修正 + 直接标准化 DS 对照）
- `examples/mcr_example.py` — MCR-ALS 示例（合成三组分混合光谱 + 最纯变量初始化 + 交替非负最小二乘）
- `examples/doe_example.py` — DoE 示例（中心复合设计 11 次实验 + 二次响应面拟合 + OFAT 对照）
- `examples/simca_example.py` — SIMCA 示例（合成两类光谱 + 未知类 + 每类 PCA 模型 + 综合距离判别）
- `examples/opls_example.py` — OPLS 示例（合成光谱 + 正交变异过滤 + PLS1 对照）
- `examples/pls_da_example.py` — PLS-DA 示例（合成两类光谱 + 5 折交叉验证选潜变量数 + 混淆矩阵）

### 其他

- `CHANGELOG.md` — 维护记录

## 快速开始

```
pip install numpy scikit-learn scipy

python examples/pls_example.py
python examples/lda_example.py
python examples/preprocessing_example.py
python examples/outlier_detection.py
python examples/full_workflow.py
python examples/svr_example.py
python examples/random_forest_example.py
python examples/calibration_transfer_example.py
python examples/mcr_example.py
python examples/doe_example.py
python examples/simca_example.py
python examples/opls_example.py
python examples/pls_da_example.py
```

## 一句话速查

- PLS 预测数值，LDA 区分种类，本质都是线性的。
- PLS 主成分数别贪多，靠交叉验证选。
- 光谱先做预处理（SNV / 一阶导 / MSC），再建模。
- 样本数少于变量数时，LDA 直接用会崩，先降维。
- 异常检测看 T² 和 Q 两个量，剔除样本要有理由并记录。
- 关系接近线性时优先 PLS；明显弯曲时再考虑 SVR，且 SVR 必须标准化并用交叉验证选参。
- 随机森林能拟合非线性与交互作用、不用标准化，但不能外推：校正集浓度范围必须覆盖未来样品。
- 模型换仪器要先做转移：先试斜率/截距修正，波长漂移明显时用 DS / PDS，效果必须在独立样品上验证。
- 混合物峰重叠、分不开时用 MCR：C 和 S 都未知，靠非负等约束收窄旋转模糊，残差低不等于拆对，相对浓度须标定才成绝对浓度。
- 做实验先排设计：一次只变一个因素看不到交互作用，先析因筛因素、中心点验弯曲，再用响应面找最优点，最优点必须经真实验证实验确认。
- 未来可能出现未知类别时用 SIMCA：每类单独建 PCA 模型，T² 与 Q 一起看，离所有模型都远就拒判，不要硬猜。
- 想要干净的载荷解释时用 OPLS：先滤掉与目标无关的正交变异再建模，精度收益通常有限，解释收益才是重点；过滤必须在交叉验证的每一折内重算。
- 分类时用 PLS-DA：y 换成类别编码即可判类，但它没有拒判、输出不是概率，评价要看混淆矩阵与置换检验，不能只报总准确率。

## 开源协议

MIT License，见 `LICENSE`。
