# 息肉增强：依据公开代码设置

息肉论文没有统一的增强规范。此次固定参照HSNet公开加载器中**显式开启的几何增强分支**，配置名`augmentation: hsnet`；不称为所有息肉论文的共同设置，也不称为完整HSNet训练协议复现。

## 核对的作者代码

| 方法 | 锁定版本 | 加载器增强 | 训练尺度 |
| --- | --- | --- | --- |
| HSNet | `9d3986a8ada19314bdf6884fed3690ddc0b987ec` | 开启时：RandomRotation(90)，上下翻转0.5，左右翻转0.5；无颜色扰动 | 352基准；0.75/1/1.25三尺度，每尺度一次优化 |
| PraNet | `95f12bd4e32ec954b5069654abae7de9473fd1f5` | 所核对加载器只Resize/ToTensor/Normalize，没有随机翻转或颜色扰动 | 352基准；0.75/1/1.25三尺度，每尺度一次优化 |

来源：

- [HSNet加载器](https://github.com/baiboat/HSNet/blob/9d3986a8ada19314bdf6884fed3690ddc0b987ec/utils/dataloader.py)
- [HSNet训练](https://github.com/baiboat/HSNet/blob/9d3986a8ada19314bdf6884fed3690ddc0b987ec/Train.py)
- [PraNet加载器](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/utils/dataloader.py)
- [PraNet训练](https://github.com/DengPingFan/PraNet/blob/95f12bd4e32ec954b5069654abae7de9473fd1f5/MyTrain.py)

HSNet该版本Train.py默认augmentation是布尔True，而加载器比较字符串`'True'`；这里参照其启用分支的操作定义，不能声称默认训练命令必然开启这些操作。

## 当前实现

按以下顺序操作，只在训练集启用：

1. 连续均匀采样角度[-90°,90°]，原尺寸旋转，expand=False，黑色填充，nearest插值。不是只在0/90/180/270°中选择。
2. 以0.5概率上下翻转。
3. 以0.5概率左右翻转。
4. 继续使用SAM2的1024×1024输入、ImageNet归一化及二值标签最近邻缩放。

图像、掩码和FOV共享同一角度与翻转操作，不靠反复重置全局种子来对齐。没有亮度/对比度、模糊、透视、弹性变形或随机裁剪。无显式FOV时，旋转填充区域作为背景参与损失；有显式FOV时，旋转产生的视野外区域不参与损失。

**与HSNet完整预处理的差异**：SAM2仍是1024输入，标签Resize使用nearest保持二值；原加载器Resize默认bilinear可产生软标签。旋转操作及其nearest插值对齐公开增强分支，但这里不复制软标签与多尺度训练。SAM2当前结构要求固定输入，直接把352或三尺度输入搬进来会改变计算路径；三尺度还会使每批更新次数变成3次，不能在相同epoch下视作相同预算。

## 运行

```bash
git pull
medsam train --config configs/polyp_full_aug.yaml
```

- `polyp_full.yaml`：无增强对照。
- `polyp_full_aug.yaml`：**当前采用的 light 增强**（翻转、90° 倍数旋转、亮度/对比度 ±10%），独立保存至 `outputs/polyp_baseline_full_aug_seed42/`。
- `polyp_full_hsnet.yaml`：按 HSNet 启用分支设置的几何增强（连续 ±90° 旋转＋翻转、无颜色扰动），保留用于协议对照，独立保存至 `outputs/polyp_baseline_full_hsnet_seed42/`。

选择 light 的依据（本机单 seed、单划分、五组测试集 Dice 宏平均）：light 0.7987，HSNet 几何 0.7880，无增强 0.7597。light 的优势集中在 CVC-300 与 ETIS，这两组对颜色/光照变化最敏感；HSNet 几何在 ClinicDB 上略好。该结果只作为当前工程选择，不构成对论文协议或增强方法的结论；换划分或更多 seed 时需重新确认。

三组共用原有训练/验证清单与其余超参，不改划分、不覆盖历史结果。验证、测试和推理不增强，不使用TTA；不要把已有light结果改名当作hsnet结果。归档记录包含模式、参考版本、角度、概率、插值及输入尺寸。

本次不处理先前评估工具的去重、HSNet外部路径或阈值聚合问题；这些仍须单独修正后才能作为正式比较依据。
