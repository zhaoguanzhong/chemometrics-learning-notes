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

### 示例（均可直接运行）

- `examples/pls_example.py` — 合成光谱上的 PLS 回归
- `examples/lda_example.py` — PLS 降维后接 LDA 的分类示例
- `examples/preprocessing_example.py` — 原始光谱 vs SNV 的预处理对比
- `examples/outlier_detection.py` — PCA 的 T² 与 Q 残差异常检测
- `examples/full_workflow.py` — 完整工作流：预处理 → 异常剔除 → 选主成分数 → 测试集报告

### 其他

- `CHANGELOG.md` — 维护记录

## 快速开始

```bash
pip install numpy scikit-learn
python examples/pls_example.py
python examples/lda_example.py
python examples/preprocessing_example.py
python examples/outlier_detection.py
python examples/full_workflow.py
```

## 一句话速查

- PLS 预测数值，LDA 区分种类，本质都是线性的。
- PLS 主成分数别贪多，靠交叉验证选。
- 光谱先做预处理（SNV / 一阶导 / MSC），再建模。
- 样本数少于变量数时，LDA 直接用会崩，先降维。
- 异常检测看 T² 和 Q 两个量，剔除样本要有理由并记录。

## 开源协议

MIT License，见 `LICENSE`。
