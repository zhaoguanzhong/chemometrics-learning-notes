# 化学计量学学习笔记 | Chemometrics Learning Notes

面向初学者的化学计量学笔记：从线性模型、偏最小二乘（PLS）到线性判别分析（LDA），
以及光谱预处理与模型验证的常见做法。内容以概念讲解和可运行示例为主，持续补充中。

> 免责：本仓库为学习笔记，不是教科书或论文引用来源；公式与结论请以教材和原始文献为准。

## 目录

- `notes/01-linear-models.md` — 线性模型：y = Xβ + ε
- `notes/02-pls.md` — PLS 与 PLS-DA：为光谱而生的回归方法
- `notes/03-lda.md` — LDA：线性判别分析
- `notes/04-preprocessing.md` — 光谱预处理：SNV、MSC、一阶导数
- `notes/05-validation.md` — 模型验证：交叉验证与常见坑
- `examples/pls_example.py` — 用 scikit-learn 在合成光谱数据上跑 PLS 回归

## 快速开始

```bash
pip install numpy scikit-learn
python examples/pls_example.py
```

## 一句话速查

- PLS 预测数值，LDA 区分种类，本质都是线性的。
- PLS 主成分数别贪多，靠交叉验证选。
- 光谱先做预处理（SNV / 一阶导 / MSC），再建模。
- 样本数少于变量数时，LDA 直接用会崩，先降维。
