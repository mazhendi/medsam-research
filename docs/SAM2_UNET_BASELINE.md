# SAM2 Hiera-L 息肉分割基线

这是独立接入的第三方 **SAM2-UNet** 基线，不是新提出的模块。现有SAM2.1 Small、region-shape及其权重保持独立。

## 结构与可训练参数

352 RGB → SAM2 Hiera-L四级特征[144,288,576,1152] → 四个RFB各压缩至64通道 → 三层跳接上采样解码 → 主输出及两个深监督输出。

Hiera原参数冻结，每个编码block前的32维Adapter可训练；RFB、解码器和输出头可训练。冻结block不能包在no_grad里，否则前置Adapter无法通过这些block获得梯度。没有使用原SAM提示、掩码解码器或视频记忆。

来源：https://github.com/WZH0120/SAM2-UNet ，锁定架构来源提交 `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e`。论文：https://doi.org/10.1007/s44267-025-00106-w 。署名与许可证见 `third_party/sam2_unet/`。

## 1. 环境与预训练权重

沿用仓库README的PyTorch/torchvision及官方SAM2安装步骤，安装本项目：

```bash
pip install -e '.[dev]'
mkdir -p checkpoints
curl -fL --retry 3 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -o checkpoints/sam2_hiera_large.pt
```

用原始 **SAM2 Hiera-L** 权重，不能用当前Small的best.pt，也不要换成SAM2.1 Large后仍称原版复现。程序严格提取并加载 `image_encoder.trunk.*`，不下载或构造无关的SAM解码和记忆组件。权重不会进Git。训练需要用户本机4090；本轮云端验证不包含预训练权重加载或GPU精度复现。

## 2. 合并训练池为1450张

若现有train.csv为1305、val.csv为145，执行：

```bash
medsam merge-polyp-training --manifests data/polyp_full/train.csv data/polyp_full/val.csv --output data/polyp_full/train_all.csv
```

程序检查计数、ID重复、解码后图像完全重复及原始图像/掩码合法性，输出来源计数。输出使用绝对路径，移动原数据目录后需重新生成。它不证明划分就是公开基准，也不能自动判断900 Kvasir+550 ClinicDB，应核对来源计数和原始分割。不能为了凑1450加入测试图。

准备好测试清单后，使用现有审计命令检查跨集合重复：

```bash
medsam audit data/polyp_full/train_all.csv /path/to/Kvasir.csv /path/to/ClinicDB.csv /path/to/CVC-300.csv /path/to/ColonDB.csv /path/to/ETIS.csv
```

审计发现重叠时先追查来源，不能静默删除或混入训练。历史dedup清单和完整公开基准应分别记录。

## 3. 训练

```bash
medsam train-sam2-unet --config configs/sam2_unet_polyp1450.yaml
```

默认1450张、352、batch16、20轮、FP32、AdamW lr.001/wd.0005、cosine至1e-7；水平/垂直翻转各.5；ImageNet归一化；三个输出的weighted BCE+weighted IoU相加。标签与作者代码一致采用bicubic软目标，不沿用历史light增强。

Batch16是本项目对比协议，参考多篇息肉分割论文采用的批大小；SAM2-UNet作者默认是12。因此本配置属于基于作者实现的协议调整，不能标为完全相同超参的原版复现。尚未在4090上验证batch16的完整反向显存。

配置路径相对YAML位置。输出目录必须不存在，以免覆盖旧实验。日志记录optimizer steps，隔5轮保存快照；`last.pt`为固定末轮，无外部测试选模，也不将已合并进训练的145张继续作为验证。当前没有断点续训；保存文件包含完整模型与训练来源记录，不包含可恢复优化器状态。

**论文/源码差异：** 作者发布的train.py是单尺度，而论文另列[1,1.25]多尺度。默认复现发布代码的固定352；若验证论文尺度方案，将scales改为[1.0,1.25]且更换输出目录，1.25对齐32后为448。此变体每batch两次优化，不能与单尺度同轮数称作相同计算预算。20轮来自SAM2-UNet设置，不照搬Polyp-PVT的100轮。

## 4. 导出原图分辨率概率图

```bash
medsam predict-sam2-unet --config configs/sam2_unet_polyp1450.yaml --checkpoint runs/sam2_unet_polyp1450_352/last.pt --manifest /path/to/ETIS.csv --output outputs/sam2_unet_etis_raw --normalization raw
```

`raw`导出sigmoid概率；作者处理方式另用新目录：

```bash
medsam predict-sam2-unet --config configs/sam2_unet_polyp1450.yaml --checkpoint runs/sam2_unet_polyp1450_352/last.pt --manifest /path/to/ETIS.csv --output outputs/sam2_unet_etis_author --normalization per-image
```

先将logits双线性恢复原图尺寸，再sigmoid；`per-image`额外逐图min–max。两个模式均记录8-bit截断，输出 `predictions.csv` 与PNG，字段兼容现有概率图评估工具；不能将raw与归一化结果混入同一对比列。既有HSNet指标脚本默认读取dedup测试清单，正式全量比较前须统一相应清单及指标口径。

预测只读取图像，不需要GT或原始预训练文件；项目训练checkpoint已含完整编码器权重。

## 5. 验证范围

测试覆盖Adapter梯度、冻结参数不变、三输出深监督、训练清单重复拒绝、软标签与输出映射。另用真实Hiera-L随机初始化验证352前向；这与加载预训练权重、完整训练及论文结果复现是不同层级。

2026-11-21验证：全套89项测试通过，新文件ruff检查通过；CPU真实Hiera-L随机初始化前向三个输出均为[1,1,352,352]，总参数216,529,891，可训练参数4,380,595。尚未下载/验证官方Large预训练文件，也未执行1450张GPU训练。
