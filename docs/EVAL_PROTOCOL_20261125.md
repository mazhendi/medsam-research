# 息肉评估协议（2026-11-25 决定）

本文档固定后续所有实验的评估口径，避免不同批次的数字混用。

## 决定

| 项 | 取值 |
|---|---|
| 测试清单 | **完整清单** `data/polyp_test_full`，100 / 62 / 60 / 380 / 196 |
| Dice / IoU | **固定阈值 0.5**（论文惯例） |
| Sα / Eφ / Fβω / MAE | 概率图指标（阈值无关），同一批完整清单 |
| HD95 | 0.5 阈值二值掩码，表面距离 95 分位，单位原图像素 |
| 概率图 | sigmoid 原始输出，8-bit 量化，`--normalization raw` |
| 辅助口径 | 256 阈值平均（HSNet 惯例）仅作附表，不用于主表 |

## 为什么

1. 完整清单与文献一致（此前去重版为 58/346/193，无法直接比较）
2. 固定 0.5 是 PraNet / Polyp-PVT / MEGANet / DSCNet 等表格的口径
3. 已实测两种口径的差：**阈值 +0.0033~0.0039**、**去重 +0.0014**，方向一致，不改变相对排名

## 历史结果的处理

- `*_dedup.csv` 及其对应的所有历史结果保留，但**不再作为主表来源**
- 之前的分析（尺寸分桶、失败样本、模块消融）来自去重清单，结论方向不变，引用时需注明清单版本

## 复现命令

```bash
# 完整清单（源数据 60/62/380/196/100）
python tools/prepare_polyp_test.py --output data/polyp_test_full

# 概率图
medsam predict-polyp-pvt --config CONFIG --checkpoint CKPT \
  --manifest data/polyp_test_full/ETIS-LaribPolypDB.csv --output OUT --normalization raw

# 固定 0.5 的 Dice / IoU
python tools/eval_fixed_threshold.py --prefix PREFIX_ --test-root data/polyp_test_full \
  --manifest-suffix .csv --output reports/.../fixed05

# 概率图指标（Sα / Eφ / Fβω / MAE）
OMP_NUM_THREADS=1 python tools/eval_hsnet_parallel.py --pred-root outputs --pred-prefix PREFIX_ \
  --test-root data/polyp_test_full --manifest-suffix .csv --soft-maps --output reports/.../hsm

# HD95
python tools/eval_hd95.py --prob-root outputs --prefix PREFIX_ \
  --test-root data/polyp_test_full --manifest-suffix .csv --output reports/.../hd95
```

## 当前主结果（三头版，完整 + 0.5）

| 测试集 | Dice | IoU | HD95 | Sα | Eφ | Fβω | MAE |
|---|---|---|---|---|---|---|---|
| Kvasir | 0.9289 | 0.8817 | 待补 | 0.9313 | 0.9633 | 0.9188 | 0.0220 |
| CVC-ClinicDB | 0.9446 | 0.8990 | 待补 | 0.9532 | 0.9856 | 0.9421 | 0.0074 |
| CVC-300 | 0.8895 | 0.8177 | 待补 | 0.9255 | 0.9619 | 0.8579 | 0.0076 |
| CVC-ColonDB | 0.8178 | 0.7350 | 待补 | 0.8674 | 0.9173 | 0.7942 | 0.0279 |
| ETIS | 0.7597 | 0.6827 | 待补 | 0.8538 | 0.8727 | 0.7189 | 0.0186 |
| **宏平均** | **0.8681** | **0.8032** | 待补 | **0.9062** | **0.9402** | **0.8464** | **0.0167** |
