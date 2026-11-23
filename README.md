> **最新无CFM配置**：`configs/polyp_pvt_slots_no_cfm.yaml`。原CFM已从该计算图移除，三层对齐拼接后仅用1×1压通道；A/B/C启用、A/B外层残差关闭。见[说明](docs/POLYP_PVT_SLOTS.md)。

> **新增原拓扑接口版本**：保留原CFM和44网格双头，提供A/B/C单项替换。见[结构与运行顺序](docs/POLYP_PVT_SLOTS.md)。优先验证单项，尚无训练成绩。

> **A/B去外层残差变体**：使用`configs/abc_pvt_no_residual.yaml`，训练/预测均须配套该配置；原full配置保留旧行为。见[说明](docs/ABC_PVT.md)。

> **最新A/B/C候选已实现**：STPNet-inspired A、局部卷积前置Mamba B、SSFM-inspired通道—空间选择C。见[ABC运行说明与Mamba后端检查](docs/ABC_PVT.md)。先运行448尺寸探针；尚无完整训练成绩。

> **最新待验证设计：SDS＋ADF＋RTR**，已接入独立配置，最终头单独推理。见[训练、预测与三模块对照](docs/SDS_ADF_RTR.md)。旧SCSI/BCSC实现保留作历史实验。

> **新增待验证架构**：SCSI＋BCSC＋区域—边界解码器，完整替换原解码部分。见[运行与消融说明](docs/SUBBAND_PVT.md)。原版Polyp-PVT继续作为主基线。

> **当前主基线：Polyp-PVT（PVTv2-B2）**。SAM2-UNet和原生SAM2保留为对照。
> 安装、训练和预测命令见 [Polyp-PVT运行说明](docs/POLYP_PVT.md)。用户已报告352基线五集宏平均Dice为0.8606；新架构尚无完整训练成绩。

# medsam-research

基于 **SAM 2.1 Hiera-Small** 的自动二值医学图像分割研究，另提供独立的 **SAM2 Hiera-L / SAM2-UNet** 息肉强基线。

新增：[SAM2-UNet基线：1450张、352输入、安装与训练命令](docs/SAM2_UNET_BASELINE.md)。使用原始SAM2 Large权重，独立训练Adapter、RFB与U形解码器；不加载旧Small分割权重。

实验分支：[原生SAM2 Mask Decoder＋SCDS/SBCA/RCSR](docs/NATIVE_SAM2_MODULES.md)。提供原生对照、三组单模块与完整模型配置；先运行真实权重探针，再训练。尚未证明其精度优于SAM2-UNet。

[尺度覆盖诊断对照](docs/NATIVE_SCALE_CONTROL.md)：原ZIP诊断脚本与独立缩小增强配置；不修改现有默认增强，不作为创新模块。

[局部正样本与随机／难背景训练对照](docs/LOCAL_BACKGROUND_CONTROL.md)：自动裁剪、纯背景检查、计算预算匹配；推理仍为整图，未验证精度收益。

## 当前实现与边界

- 已实现：可学习任务 token、冻结图像编码器、训练掩码解码器；CSV 数据配对、数据检查、训练、原始分辨率评估、无标签推理、检查点与运行记录。
- 当前维护单任务基线、region-shape研究原型与SAM2-UNet独立基线实现；不包含文本分支。SAM2-UNet尚未完成GPU精度复现。视频记忆、多任务联合训练、断点续训未实现。
- 这是可运行的**基线**，不将其本身声称为新方法。已用于视网膜实验，当前也用于息肉分割；不同任务的权重和输出需独立保存。
- 第一版为单任务训练；输入为 8-bit RGB/灰度图像，目标为单通道二值掩码。DICOM、16-bit X-ray、多类别标签需先明确转换规则。
- 使用官方默认 1024×1024 输入，直接方形缩放；采用 ImageNet 均值/标准差归一化。标签最近邻缩放；评估先将 logits 恢复到原图尺寸，再以固定阈值二值化。
- 增强模式默认`none`；息肉当前采用`light`（翻转、90° 倍数旋转、亮度/对比度 ±10%），`hsnet`（连续±90°旋转、翻转、无颜色扰动）保留作协议对照。只在训练集增强，图像/标签/FOV同步变换；详细依据与选择记录见[增强说明](docs/POLYP_AUGMENTATION.md)。

## Region–shape：检索后的可选扩展

在原有 region-shape 上新增整图先验锚定的窗口竞争与候选诊断；不恢复已撤下的 BC，不更换原 SAM 解码器。默认仍是原整图版本。

- [相关论文检索与创新边界](docs/research/REGION_SHAPE_LITERATURE_20261119.md)
- [窗口竞争原理、诊断与运行命令](docs/REGION_SHAPE_WINDOW.md)
- 先用 `diagnose-region-shape` 检查已有模型，再用 `configs/region_shape_window.yaml` 做独立同预算实验；尚未证明精度收益。

## 1. 安装（Linux / RTX 4090）

建议 Python 3.11 或 3.12；需要 Git、NVIDIA 驱动及网络。以下 CUDA 12.4 wheel 需要兼容驱动；其他驱动请按 PyTorch 官方安装器选择对应版本。

```bash
git clone https://github.com/mazhendi/medsam-research.git
cd medsam-research
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/facebookresearch/sam2.git vendor/sam2
git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e vendor/sam2
pip install -e '.[dev]'
```

此基线不使用 SAM 的 CUDA 后处理扩展。依赖采用上游可编辑 Git checkout，程序会核验提交号；不要直接升级 SAM 2 后继续使用旧实验记录。

下载官方 Small 权重：

```bash
mkdir -p checkpoints
curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  -o checkpoints/sam2.1_hiera_small.pt
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

来源：[SAM 2 官方仓库](https://github.com/facebookresearch/sam2)。模型权重不进入 Git；数据的内置许可例外见下文。

### 视网膜血管数据：已内置 CHASE_DB1

仓库现包含许可为 CC BY 4.0 的完整 CHASE_DB1 数据压缩包及署名（28 张原图、两套标注）。
安装环境后无需另找数据路径：

```bash
medsam prepare-retina --dataset CHASE_DB1 --output data/retina/CHASE_DB1
medsam train --config configs/retina_chase.yaml
```

DRIVE/STARE 下载来源、导入命令、观察者选择、分组与 FOV 策略见
[视网膜数据说明](docs/RETINAL_DATASETS.md)。CHASE_DB1 是本项目明确许可的 Git 数据例外；
其他原始数据、生成的 PNG、检查点和实验结果保持忽略。

## 2. 准备数据清单

可读取 Kvasir、ClinicDB、血管、皮肤等已下载图像；不自动下载，也不自行猜测患者/视频划分。

```bash
medsam manifest --images /path/to/images --masks /path/to/masks \
  --dataset Kvasir --output data/all.csv
```

图像与掩码按相对目录和文件主名严格配对，可以是 JPG 对 PNG。若掩码名有后缀，例如 `case_segmentation.png`，增加 `--mask-suffix _segmentation`。

CSV 可选 `fov` 列指定二值视野掩码；存在时训练损失和指标仅使用视野内像素。
没有此列或路径为空时使用全图，不自动估算 FOV。

CSV 示例：

```csv
id,image,mask,dataset,group
frame001,images/frame001.png,masks/frame001.png,ClinicDB,video01
frame002,images/frame002.png,masks/frame002.png,ClinicDB,video01
```

文件路径相对 **CSV 所在目录**。`group` 应填写患者或视频 ID；`dataset` 为分组命名空间。请依据官方划分/分组信息产生 `data/train.csv`、`data/val.csv`、`data/test.csv`。**不要将同一患者或视频的相邻帧随机拆分。** 分组未知时可以留空，但检查程序无法替你证明组间独立。

```bash
medsam audit data/train.csv data/val.csv data/test.csv
```

审计检查配对尺寸、二值标签、解码后完全相同的图像，以及已提供 group 的跨划分重叠；不检测近重复图像。默认掩码 `> 0` 为前景，支持 `{0,1}`、`{0,255}`；RGB 标签、抗锯齿标签及多类别标签不会被悄悄当作二值数据。

## 3. 训练

修改 `configs/baseline.yaml` 中的数据和权重路径。文件系统路径相对 **YAML 所在目录**；SAM 配置名称是上游包资源，不是本地文件路径。

```bash
medsam train --config configs/baseline.yaml
```

4090 起步批量大小为 1，启用混合精度（支持时BF16，否则FP16）；实际显存与速度需在你的机器测量。只训练任务 token 和掩码解码器，不生成真实标注点/框，也不做多掩码真值选择。所有数据仍通过冻结的图像编码器。

输出到 `outputs/baseline_seed42/`：

- `config.yaml`、`provenance.json`：解析后的配置、依赖版本、Git 版本、权重/清单/解码数据校验值。
- `history.jsonl`、`best_validation.csv`：每轮训练损失、验证指标及最佳轮的逐图结果。
- `best.pt`、`last.pt`：仅任务 token 与掩码解码器权重；恢复时仍需相同官方预训练权重。
- `training_summary.json`：时间及 CUDA 峰值已分配显存（若使用 CUDA）。

同名输出目录已经存在时会报错，避免覆盖实验。`best.pt` 仅按验证集平均 Dice 选择；同时保存末轮，不能用外部测试集选择检查点。

## 4. 独立评估与推理

```bash
medsam evaluate --config configs/baseline.yaml \
  --checkpoint outputs/baseline_seed42/best.pt \
  --manifest data/test.csv --output outputs/test_seed42

medsam predict --config configs/baseline.yaml \
  --checkpoint outputs/baseline_seed42/best.pt \
  --manifest data/inference.csv --output outputs/predictions_seed42
```

推理清单只需要 `id,image` 两列。输出 PNG 保留原始尺寸，通过 `predictions.csv` 对应样本 ID，文件名不使用原始 ID，避免路径字符导致写出目录。

评估输出逐图 Dice、IoU、boundary F1，以及前景面积比例。boundary F1 使用原图网格上的欧氏距离容差，默认 2 **像素**，不是毫米。Dice/IoU 为逐图宏平均；两张空掩码记 1，只有一张为空记 0。另报告阳性样本 Dice 和空目标图像的假阳性率，避免阴性比例掩盖结果。常规evaluate不包含clDice、HD95或目标级漏检分析；血管clDice与中心线指标由diagnose提供。

独立评估会同时审计源训练、验证与测试清单，因此需保留源数据。评估阈值、标签阈值和边界容差必须与训练记录一致，不允许根据测试集修改。推理不需要源数据或目标标注。

## 5. 验证代码

```bash
ruff check .
pytest -q -m 'not integration'
pytest -q -m integration
```

普通测试以微型测试网络检查训练/保存/恢复/评估/预测闭环，不代表 SAM 的实验效果。集成测试使用**真实 SAM 2.1 Hiera-Small 架构、随机权重、64×64 小输入**验证批量行为、任务 token 和解码器梯度、编码器冻结；不需要下载预训练权重，不代表默认分辨率的 GPU 性能。

GitHub Actions 执行 CPU 静态检查和普通测试。真实 4090、默认 1024 输入、医学数据性能仍需实测。

## 息肉分割

沿用同一个SAM 2.1基线，从官方预训练权重开始训练。使用独立的`configs/polyp.yaml`和`outputs/polyp_baseline_seed42/`，不覆盖视网膜的`outputs/chase_observer1_seed42/best.pt`。

```bash
medsam train --config configs/polyp.yaml
```

先准备显式训练/验证清单；掩码格式、患者/视频划分及外部测试要求见[息肉数据与运行说明](docs/POLYP.md)。

### 全量训练池、增强与 HSNet 口径指标

`tools/` 提供与 `docs/POLYP.md` 配套的实验工具，均只读写显式路径，不自动下载或改划分：

```bash
# PraNet/HSNet 风格 1450 对训练图：复制、掩码 >=128 二值化、ID 哈希 90/10 划分
python tools/prepare_polyp.py --limit 0 --val-fraction 0.1 --output data/polyp_full
# 五个标准测试集：转换并生成去重清单（<dataset>_dedup.csv）
python tools/prepare_polyp_test.py
```

- `configs/polyp_full.yaml`：1305 train / 145 val，无增强
- `configs/polyp_full_aug.yaml`：**当前采用的 light 增强**（翻转、90° 倍数旋转、亮度/对比度 ±10%）
- `configs/polyp_full_hsnet.yaml`：按HSNet启用分支设置连续±90°旋转＋上下/左右翻转，无颜色扰动，保留作协议对照；详见[增强依据与差异](docs/POLYP_AUGMENTATION.md)
- `tools/predict_probs.py`：输出原图分辨率概率图（HSNet 指标的必要输入）
- `tools/eval_hsnet_metrics.py`：调用 HSNet 作者的 `eval_functions.py` 计算 Dice/IoU/Sm/Em/加权 F（256 阈值）
- `tools/eval_hd95.py`：0.5 阈值边界对称距离的 95 分位

这些脚本不读取测试标签用于模型选择；测试集去重与测试集使用范围需在报告中说明。

## 基线诊断

`medsam diagnose`提供已有基线的验证集错误图、血管指标和固定局部重读对照，不训练额外分支，详见[诊断说明](docs/DIAGNOSTICS.md)。

## 仓库维护范围

当前main维护我们自己的SAM 2.1基线、数据工具、诊断与测试；区域—轮廓原型使用独立命令和权重。已移除外部论文复现、文本/VLM分支、纠错头、区域监督和额外适配实验。历史代码仍可从Git历史找回。`data/`、`checkpoints/`、`outputs/`不随本次清理删除，也不提交训练权重。已有基线best.pt/last.pt的模型结构和加载格式保持不变。

## 区域—轮廓交替推断原型（V3）

新增共享SAM2特征、区域/隐式轮廓双向交互和联合prompt解码，最终不做原预测logits的残差相加。保留当前SAM2＋light基线，默认冻结已有权重，仅训练独立机制；先运行两图拟合探针，再跑1轮检查。

[结构图、损失、运行命令、消融与验证限制](docs/REGION_SHAPE.md)。新配置为`configs/region_shape.yaml`，与`configs/polyp_full_aug.yaml`配合使用。实现和接口检查不代表精度提升或创新已成立。

### Experimental scale/detail feature modules

Optional B/C modules preserve the existing region-shape prompt path and SAM decoder.
See [implementation, warm start and ablations](docs/SCALE_DETAIL.md).
Old configurations remain unchanged; no accuracy gain has been established.

### Coarse-to-fine B + existing RS (current proposed architecture)

Use `configs/region_shape_multiscale_b.yaml` with `--initialize-from` pointing to
a trained original RS artifact. Stage one trains only B. The enhanced pyramid
feeds both C and final SAM decoding; all three resolutions are preserved.
See [architecture, training and checkpoint details](docs/MULTISCALE_B.md).
The older scale-detail experiment is a separate optional configuration.
