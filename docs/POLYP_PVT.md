# Polyp-PVT：当前主基线

采用作者 [DengPingFan/Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT)，固定提交
`cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4`。保留现有SAM2-UNet/原生SAM2代码、配置与权重作对照。
本文档描述可运行接入，不代表已经复现论文指标。

## 结构

输入352×352 → PVTv2-B2四级特征 → CIM伪装识别、CFM级联融合、SAM相似性聚合 → 两个预测头。
这里的SAM是Similarity Aggregation Module，不是Segment Anything。
全部骨干及解码参数训练，使用ImageNet PVTv2-B2初始化；不加载SAM权重。
损失为两个头各自的边界加权BCE+加权IoU之和。
推理为 `sigmoid(上采样(logit1 + logit2))`，不能只取第二头或平均两个概率。

## 安装与准备

在仓库根目录运行（Windows也可使用相同Python命令）：

```bash
python -m pip install -e ".[polyp-pvt]"
python tools/prepare_polyp_pvt.py
```

第二条命令将官方代码检出至忽略目录`vendor/Polyp-PVT`，不会改动其他模型。
适配器检查两个算法源文件SHA256，只在内存中替换导入命名空间和硬编码预训练加载；不改前向算子。
源码保持独立，不在本仓库重新分发作者算法文件。

按[官方README预训练权重链接](https://github.com/DengPingFan/Polyp-PVT#43-pretrained-model)
取得`pvt_v2_b2.pth`，放入`checkpoints/pvt_v2_b2.pth`。这是PVTv2-B2 ImageNet骨干，
不是已完成息肉训练的权重。加载会严格检查骨干键与形状，仅去掉分类器head两个参数。
不自动下载大权重；预测本项目last.pt时不再需要ImageNet权重。

训练清单沿用`data/polyp_full/train_all.csv`，必须为900张Kvasir+550张ClinicDB组成的1450张训练池。
代码检查总数、文件配对和重复图像文件；来源组成及跨划分内容重复仍需按已有数据审计核查。
不要把任何测试清单合入训练池。

## 训练

```bash
medsam train-polyp-pvt --config configs/polyp_pvt_1450.yaml
```

配置：352，batch16，AdamW lr1e-4 / weight_decay1e-4，FP32，三尺度
[0.75,1,1.25]，每个尺度单独反向与更新，逐元素梯度裁剪±0.5。
ImageNet归一化，RGB和灰度标签均双线性缩放，标签保留软边缘。
官方默认未启用随机旋转/翻转，本配置同样不启用；已有light配置不受影响。

官方`--epoch=100`使用`range(1,100)`，实际99轮；本配置明确写99轮。
其调度器调用200轮才触发衰减，99轮内相当于恒定学习率。
每轮drop_last：1450/16取90批，三尺度270次更新；99轮共26730次更新。
这与SAM2-UNet的20轮单尺度预算不同，不能把全部差异归于架构。

短预算对照可用`configs/polyp_pvt_20ep.yaml`，共5400次更新；
同为20轮也不等于与单尺度SAM2-UNet计算量相同。99轮才是默认主实验。
输出目录必须不存在，避免覆盖已有实验。记录配置、清单/预训练哈希、步数、损失和源码版本。
仅末轮保存`last.pt`，不会每轮生成大量权重或访问五测试集。

## 预测与评估

例如Kvasir（清单名按本机实际文件调整）：

```bash
medsam predict-polyp-pvt --config configs/polyp_pvt_1450.yaml --checkpoint runs/polyp_pvt_1450_352_99ep/last.pt --manifest data/polyp_test/Kvasir_dedup.csv --output outputs/polyp_pvt/Kvasir --normalization raw
```

其余四集分别运行同一命令。按原图尺寸输出uint8概率图及predictions.csv，复用项目现有评估工具。
`raw`用于与项目原始概率对照；`--normalization per-image`对应作者逐图min-max处理。
两种口径分别报告；两边模型必须使用相同口径。不要混用固定0.5 Dice、soft Dice和256阈值平均Dice。

论文常用100/62/60/380/196共798张，而目前去重版本100/62/58/346/193共759张。
保持现有759张作项目内部对照，若补标准798张则另立清单并重评所有模型；不能直接等同论文成绩。

## 与官方训练脚本的差别

作者Train.py每轮评估五测试集，并按test_path/test的成绩保存best。
本项目明确不复用该选模过程，使用预先固定末轮，不读取测试GT选轮或选参数。
因此称为“官方结构和默认训练设置接入，固定末轮协议”，不声称与作者所有实验步骤完全相同。
如需要验证选模，应另设训练池内验证划分，并对所有模型统一协议；不能仍称1450张全部训练。
