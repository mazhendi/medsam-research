# 验证集错误诊断与高分辨率重读对照

这一阶段验证一个具体问题：保持已训练权重不变，重新读取原图局部细节能否修复血管错误？固定网格裁剪是常规对照实验，不是已确立的新方法。只有验证收益、代价和失败模式之后，才有依据设计自适应读取策略。

## 使用已有权重，不需要重新训练

在仓库根目录、原来的 Python 环境中运行：

```bash
pip install -e ".[dev]"
medsam diagnose --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_diagnose_grid2
```

Windows PowerShell 也可以直接使用上面的一行命令。权重和配置必须来自同一轮实验，原始 SAM 2 权重仍需保留。输出目录不能已经存在。该开发命令只接受配置中的验证集路径，避免反复用测试标签调整策略。

## 两个对照

- `global`：整图缩放至模型输入尺寸，单次前向，logits 恢复到原图尺寸。
- `reread`：复用整图结果，再从原始 RGB 图像取固定 2×2 重叠窗口，各自缩放并前向。重叠区域用边缘非零的 Hann 权重融合，随后将局部 logits 与整图 logits 按 0.5 比例融合，最后以原先固定的 0.5 阈值二值化。窗口由图像大小决定，不读取标注，不更新参数。

默认每张图共五次模型前向；这不是免费提升。记录 batch=1、FP32 推理的实际耗时，CUDA 同步计时，首图预热不计入耗时。重读耗时包含整图和局部路径、缩放、传输及拼接，不含指标计算或文件写入；仅供当前设备上的初步比较。`--grid`、`--overlap`、`--blend` 可在验证集探索，不能根据测试表现选值。

## 输出与解释

- `summary.json`：两种方式的图像宏平均指标、耗时、调用次数、数据及权重哈希。
- `per_image.csv`、`paired_deltas.csv`：逐图指标与重读减整图的差值。
- `per_group_deltas.csv`：存在 group 时，先在每个受试者内平均差值。同一儿童双眼不应当作两个独立受试者做显著性检验。
- `*_mask.png`、`*_errors.png`：原图分辨率预测及叠加图；红色漏分、蓝色误分、绿色正确血管，FOV 外变暗。
- `images.json`：编号和原始样本 ID 的映射、固定裁剪坐标。

新增 Precision/Recall、clDice、细/中/粗血管中心线召回率。clDice 用预测中心线落在真值内的比例与真值中心线落在预测内的比例计算，不能证明血管拓扑完全正确。宽度按真值中心线处 `2 × EDT` 估计，默认 ≤4、(4,8]、>8 原图像素，不是物理直径，不适合直接跨分辨率比较。每个分组同时报告中心线像素数；没有该组血管时指标留空，宏平均排除空值。FOV 同时约束预测与真值，FOV 边缘会影响形态学指标。

重点检查：Dice 和 clDice 是否同时改善；细血管召回上升时 Precision 是否下降；收益是否集中在个别受试者；成本增加是否值得。若固定重读没有稳定收益，应检查局部视野缺少上下文或训练与裁剪输入存在分布差异，不能据此声称高分辨率信息无用，也不应直接堆叠更多模块。

请回传 summary.json、per_image.csv、paired_deltas.csv、per_group_deltas.csv（如有）及代表性 errors.png。当前实现本身不保证提升；需要真实训练权重在验证集运行后判断。
