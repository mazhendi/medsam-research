# medsam-research

基于 **SAM 2.1 Hiera-Small** 的自动二值医学图像分割研究。

## 当前实现与边界

- 已实现：可学习任务 token、冻结图像编码器、训练掩码解码器；CSV 数据配对、数据检查、训练、原始分辨率评估、无标签推理、检查点与运行记录。
- **尚未实现**：局部高分辨率重读、LoRA、无标签测试时适配、动作收益预测器、视频记忆、多任务联合训练、断点续训。
- 这是可运行的**基线**，不是已验证的创新模型。尚无真实医学数据实验结果。
- 第一版为单任务训练；输入为 8-bit RGB/灰度图像，目标为单通道二值掩码。DICOM、16-bit X-ray、多类别标签需先明确转换规则。
- 使用官方默认 1024×1024 输入，直接方形缩放；采用 ImageNet 均值/标准差归一化。标签最近邻缩放；评估先将 logits 恢复到原图尺寸，再以固定阈值二值化。
- 当前无随机训练增强；后续公平对照需单独增加并固定增强策略，不能将此初始配置称为最优基线。

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

4090 起步批量大小为 1，启用 FP16 混合精度；实际显存与速度需在你的机器测量。只训练任务 token 和掩码解码器，不生成真实标注点/框，也不做多掩码真值选择。所有数据仍通过冻结的图像编码器。

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

评估输出逐图 Dice、IoU、boundary F1，以及前景面积比例。boundary F1 使用原图网格上的欧氏距离容差，默认 2 **像素**，不是毫米。Dice/IoU 为逐图宏平均；两张空掩码记 1，只有一张为空记 0。另报告阳性样本 Dice 和空目标图像的假阳性率，避免阴性比例掩盖结果。尚未提供 clDice、HD95 或目标级漏检分析。

独立评估会同时审计源训练、验证与测试清单，因此需保留源数据。评估阈值、标签阈值和边界容差必须与训练记录一致，不允许根据测试集修改。推理不需要源数据或目标标注。

## 5. 验证代码

```bash
ruff check .
pytest -q -m 'not integration'
pytest -q -m integration
```

普通测试以微型测试网络检查训练/保存/恢复/评估/预测闭环，不代表 SAM 的实验效果。集成测试使用**真实 SAM 2.1 Hiera-Small 架构、随机权重、64×64 小输入**验证批量行为、任务 token 和解码器梯度、编码器冻结；不需要下载预训练权重，不代表默认分辨率的 GPU 性能。

GitHub Actions 执行 CPU 静态检查和普通测试。真实 4090、默认 1024 输入、医学数据性能仍需实测。

## 后续研究顺序

先跑通单任务真实基线，再独立加入原图局部重读、LoRA，验证不同动作收益是否互补。只有现象成立，才训练收益预测器。无标签适配目标尚未确定，受监督微调不得冒充测试时适配。

### 下一步：验证集错误诊断与局部重读对照

已有训练权重可直接运行，无需重新训练：

```bash
pip install -e ".[dev]"
medsam diagnose --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_diagnose_grid2
```

输出原图尺寸的漏分/误分图、血管中心线诊断、固定网格重读前后的配对指标和计算成本。详细设置与限制见 [诊断说明](docs/DIAGNOSTICS.md)。固定网格重读属于对照实验，不代表已提出创新方法或已验证精度提升。

### 训练局部纠错分支（研究原型）

```bash
medsam train-correction --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_correction_v1 --epochs 20
```

冻结已有基线，训练共享的细节/上下文纠错头，输出验证集整图和逐区域对照；尚无动作选择器。详见 [结构、损失和实验限制](docs/CORRECTION.md)。

### 区域监督精度控制器（独立研究原型）

```bash
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_meta --mode meta --optimizer adam --learning-rate 0.001 --epochs 50
```

冻结原基线，新增轻量分割适配头，以训练集内部不同受试者的元目标学习区域占比/像素BCE的混合权重。支持pixel、fixed、curriculum、meta、reweight五组对照；独立验证集不更新控制器，推理不需要控制器。先阅读[实验步骤与限制](docs/GRANULARITY.md)，与pixel对照一起运行；本模块尚无已验证的创新或性能结论。

如果 `pixel` 对照的训练损失和预测几乎不变，先暂停控制器对照，运行训练集小样本拟合诊断：

```bash
medsam probe-adapter --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_adapter_probe --steps 100 --samples 2
```

比较相同初始化下不同SGD步长与Adam的拟合、梯度、实际参数更新及预测残差。它不会修改正式训练优化器，解释方式见[诊断说明](docs/GRANULARITY.md#先决检查pixel几乎不学习时暂停控制器结论)。

### 当前优先实验：普通解码器继续训练 vs 多尺度残差头

暂停监督控制器，先建立可学习的强对照：

```bash
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_decoder --mode decoder --epochs 50
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_multiscale --mode multiscale --epochs 50
```

另有deep和参数量匹配的pooled对照。所有组用纯BCE+Dice，报告原图训练/验证指标；详见[结构、比较边界与评估命令](docs/BACKBONE_CONTROLS.md)。多尺度头是诊断性改进，不代表已经证明创新或提升。

### 新增：文本定位与方向性修复（两阶段研究原型）

```bash
pip install -e ".[dev,text,captions]"
medsam prepare-biomedclip --output checkpoints/biomedclip
medsam generate-image-prompts --manifests data/retina/CHASE_DB1/train.csv data/retina/CHASE_DB1/val.csv --task "retinal blood vessels" --output outputs/chase_image_prompts
medsam train-text-localizer --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --prompt "retinal blood vessels" --prompt-map outputs/chase_image_prompts/prompts.json --output outputs/chase_text_localizer --epochs 30
```

先检查visual/text/text_null与错误互补性，再决定是否训练第二阶段。恢复/抑制头、五组修复消融、固定评估和无标注预测均已实现；详见[安装、完整命令与验证边界](docs/TEXT_VERIFICATION.md)。采用真实官方BiomedCLIP，但不是Simple-ViLMedSAM的IPP/BID复现；尚未验证精度提升或新颖性。

逐图描述生成、人工检查、图文错配对照及新图预测见[逐图文本说明](docs/IMAGE_CAPTIONS.md)。现在训练必须明确选择`--prompt-map`或统一文本对照`--task-only`，不会缺失时静默补同一句话。

高分辨率重叠裁块响应已加入：`--evidence-mode tiled --tile-grid 2`，详见[实验命令、采样结构与成本](docs/TILED_EVIDENCE.md)。
