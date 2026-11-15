# 视网膜血管数据入口

## 已接入的数据集

| 数据集 | 血管基准图像数 | 当前获取方式 | 标注与视野策略 |
| --- | --- | --- | --- |
| CHASE_DB1 | 28，来自 14 位儿童双眼 | **完整原始压缩包随仓库提供**，约 2.4 MiB | 两位观察者分别保留；按儿童划分；未提供官方 FOV 时全图评价 |
| DRIVE | 40，官方训练/测试各 20 | 用户从官方入口取得 ZIP，本项目负责转换 | 支持 manual1/manual2；使用官方 FOV；若测试标注未公开，只生成测试推理清单 |
| STARE | 经典血管子集 20，不是整个 STARE 图像库 | 官方下载命令或本地 TAR/目录导入 | AH/VK 分开保留；默认全部作为外部测试，不虚构官方划分；全图评价 |

官方来源：[CHASE_DB1](https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/)、[DRIVE](https://drive.grand-challenge.org/)、[STARE 血管子集](https://cecas.clemson.edu/~ahoover/stare/probing/)。

2026-11-15：CHASE_DB1 官方记录明确标示 CC BY 4.0，故可保留署名后随库提供。当前环境官方文件端点返回 403，已使用固定提交镜像取得完整数据并核查数量、尺寸及标签。
来源、镜像和校验值见 [署名与许可](../assets/datasets/CHASE_DB1/ATTRIBUTION.md)。不把镜像宣称为官方托管。

当前环境 DRIVE 下载入口返回 403，STARE 官方站点/文件返回 503。因此这两套数据**尚未实际下载验证**，只提供来源和导入实现；没有以第三方门户的许可代替原始数据使用条款，也没有把不明确可再分发的数据复制进 Git。

## CHASE_DB1：拉取后即可准备

完成 README 中 Python 环境安装后，在仓库根目录执行：

```bash
medsam prepare-retina --dataset CHASE_DB1 --output data/retina/CHASE_DB1
medsam audit data/retina/CHASE_DB1/train.csv data/retina/CHASE_DB1/val.csv data/retina/CHASE_DB1/test.csv
medsam train --config configs/retina_chase.yaml
```

默认使用第一位观察者。准备脚本保留第二位观察者标注于 `masks2/`，不会将两份标注合并或当成两个独立样本。`--observer 2` 可生成以第二观察者为目标的新目录；请使用新的实验目录，不得在测试后选择更有利的观察者。

固定种子 42，将 14 位儿童按 8/2/4 分为训练、验证和测试，即 16/4/8 张图像。左右眼从不分到不同集合。**这是项目自定义开发协议，不是官方协议，也不等同于论文常见的 20/8 设置。** 划分 ID、分组、种子和源数据指纹保存在 `provenance.json`。

仓库同时记录 [固定划分 ID](../protocols/CHASE_DB1_seed42.json)，便于在不同机器核对；准备脚本会按同一规则生成对应 CSV。

```bash
medsam evaluate --config configs/retina_chase.yaml \
  --checkpoint outputs/chase_observer1_seed42/best.pt \
  --manifest data/retina/CHASE_DB1/test.csv --output outputs/chase_test_seed42
```

## DRIVE：保留官方训练/测试边界

从官方页面完成其要求的下载流程后，将训练和测试 ZIP 放在本地。不要把测试集用于调参。

```bash
medsam prepare-retina --dataset DRIVE \
  --source /path/to/training.zip /path/to/test.zip --output data/retina/DRIVE
medsam train --config configs/retina_drive.yaml
```

也支持已解压的 `DRIVE/` 根目录。识别原始命名 `21_training.tif`、`21_manual1.gif`、`21_training_mask.gif` 等，将 GIF/TIFF 规范化为 PNG。官方训练 20 张中，固定种子选出 4 张验证、16 张训练；官方测试 20 张全部保留。

官方测试 ZIP 若不含真值，不会从训练标签推测，也不会伪造标签：输出 `test_inference.csv`，只用于 `medsam predict`，分数应通过官方评价渠道获取。若已有合法取得的完整测试标注，则输出 `test.csv`，可本地评价。

原图 FOV 由 CSV 的 `fov` 列传入，训练损失与评价都排除视野外像素。所有指标恢复到原图网格后计算。训练/验证内部划分仅基于可用图像 ID；没有额外患者元数据时，不声称额外证明了患者级独立。

## STARE：默认独立外部测试

官方下载命令：

```bash
medsam download-retina --dataset STARE --output data/raw/STARE
medsam prepare-retina --dataset STARE \
  --source data/raw/STARE/stare-images.tar data/raw/STARE/labels-ah.tar data/raw/STARE/labels-vk.tar \
  --output data/retina/STARE
```

支持 TAR 中的 `.ppm.gz`，无需自行解压。若源站仍不可用，可将你已合法下载的三个原始归档路径传给 `--source`；程序不会自动切换未经核实的镜像。

默认第一观察者为 AH、第二观察者为 VK。仅识别经典 20 图像血管子集，导出 `external.csv`。例如用 CHASE 模型直接测试 STARE：

```bash
medsam evaluate --config configs/retina_chase.yaml \
  --checkpoint outputs/chase_observer1_seed42/best.pt \
  --manifest data/retina/STARE/external.csv --output outputs/chase_to_stare_seed42
```

不得用 STARE 标签选择 CHASE 训练的阈值、epoch 或方法。未提供官方 FOV 时，全图评价并在报告中明确，不把自动估计的视野掩码称为官方掩码。

## 来源论文

- DRIVE: Staal et al., “Ridge-based vessel segmentation in color images of the retina,” IEEE TMI, 2004. DOI: 10.1109/TMI.2004.825627.
- STARE: Hoover, Kouznetsova, Goldbaum, “Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response,” IEEE TMI, 2000. DOI: 10.1109/42.845178.
- CHASE_DB1: Fraz et al., “An ensemble classification-based approach applied to retinal blood vessel segmentation,” IEEE TBME, 2012. DOI: 10.1109/TBME.2012.2205687.

## 数据版本与评价边界

当前血管支持 Dice、IoU、boundary F1；AUC、灵敏度、特异度、clDice、HD95 尚未实现，不应在论文表格中填入这些未计算的指标。准备脚本不裁剪、不增强、不改尺寸。下载、解包、格式转换和固定划分不构成算法创新。
