# Simple-ViLMedSAM：作者组件运行复现

本目录不是我们此前的SAM2文本残差头。这里使用作者的SAM ViT-H、rank=4 LoRA、信息瓶颈归因和原版亲和力细化/双向解码器。论文已在CVPR 2026正式论文集公开。

- [正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Qian_Simple-ViLMedSAM_Simple_Text_Prompts_Meet_Vision-Language_Models_for_Medical_Image_CVPR_2026_paper.pdf)
- [补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Qian_Simple-ViLMedSAM_Simple_Text_CVPR_2026_supplemental.pdf)
- [作者仓库](https://github.com/qcc001/Simple-ViLMedSAM)，锁定`76d95d68fff037182cbdaf37fffb2d30bda2a9ec`。

## 已知边界与代码审计

不能宣称已复现论文精度。作者仓库未提供文件级划分、固定10-shot样本或训练好的分割权重；[issue #1](https://github.com/qcc001/Simple-ViLMedSAM/issues/1)也在询问这些信息。补充材料给出Kvasir 700/100/200等数量，不能据此推断具体图像名单；ISIC版本/子集同样存在未澄清处。论文zero-shot是使用其他数据集训练后迁移到未见目标，不等于完全不训练。

锁定版本中发现：

- `scripts/generate_map.py:308`有未闭合字符串，且入口变量与参数不一致。
- 数据加载器缺少Dataset、glob、torch等导入，使用未定义的vmap、k；空前景标签会在max()处失败。
- train.py中net未定义；训练和验证读取同一个data_root；未处理尾部不足8步的累积梯度；恢复路径是未展开的shell表达式。
- `Segment_Anything`与实际目录`segment_anything`不一致。
- 主论文写AdamW，补充材料写Adam；我们默认作者train.py的AdamW，也提供`--optimizer adam`，不把两者混同。
- 公开InformationBottleneck存储mean/std，但forward未使用它们，因此传入var不会影响输出。这和补充材料对σ的描述不一致。本次保留公开算法，不擅自补公式。

`prepare.py`只做导入路径、无关导入及安全权重加载兼容修补，记录前后SHA256。原版model/model.py、scripts/iba.py、SAM图像编码器和注意力代码保持不变并核验哈希。源代码下载到忽略的vendor目录，不复制为我们的模型。

原版生成器假设Hugging Face CLIP接口，而官方BiomedCLIP bundle是OpenCLIP/timm。我们用本地已有BiomedCLIP，在第ℓ个block输出挂钩，调用作者InformationBottleneck；保持Adam lr=1、10步、β×KL−图文余弦目标、通道容量求和、去CLS及归一化。只适配tuple/tensor接口。这是有明确API改动的组件复现，不是未经修改的官方端到端运行。

## 1. 准备环境与作者源码

在仓库根目录运行，保留当前CUDA版torch：

```bash
git pull
pip install -e ".[dev,text]"
python experiments/simple_vilmedsam/prepare.py
```

不直接安装作者requirements.txt：它包含大量无关包及与当前环境不同的torch/CUDA固定版本。此处使用现有文本环境的open_clip 2.32和transformers 4.49，不声称环境逐项相同。

准备官方权重：

```bash
# 已有完整BiomedCLIP bundle则跳过
medsam prepare-biomedclip --output checkpoints/biomedclip
curl -L --fail -o checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

ViT-H权重约2.4 GiB。本次下载SHA256：`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`。不使用SAM2权重替代。

## 2. 先生成真正的信息瓶颈归因图

```bash
python experiments/simple_vilmedsam/run.py maps --manifest data/retina/CHASE_DB1/train.csv --prompt "retinal blood vessels" --layer 8 --beta 0.1 --var 2 --steps 10 --device cuda --output outputs/vilmed_chase_train_maps
python experiments/simple_vilmedsam/run.py maps --manifest data/retina/CHASE_DB1/val.csv --prompt "retinal blood vessels" --layer 8 --beta 0.1 --var 2 --steps 10 --device cuda --output outputs/vilmed_chase_val_maps
```

CHASE不在论文中，以上超参数只是探索设置，不能称为论文对血管的设定。补充材料的Kvasir参数是文本`polyp`、β=0.1、σ=2、ℓ=8；ISIC是`skin melanoma`、0.1、2、10；肺部X片/CT是`lung`、0.1、1、8。层索引沿用公开代码传值，按零起始block解释。

生成仅读取原图，不读取掩码或FOV。输出maps.json记录图像/归因图哈希、模型版本、指令及设置，每图另存损失、α梯度、原始容量与响应跨度。min-max归一化会放大微小差异，响应看起来不平不代表定位正确；必须结合原始跨度和后续对照判断。

归因图是按图优化瓶颈得到的，不是之前直接计算patch–文本余弦。BiomedCLIP权重仍冻结。生成失败不会导出complete索引；换新输出目录重跑，当前入口没有续跑功能。

## 3. 原论文数据的组件检查（可选）

```bash
python experiments/simple_vilmedsam/sample_kvasir.py
python experiments/simple_vilmedsam/run.py maps --manifest data/kvasir_reproduction_examples/samples.csv --prompt polyp --device cuda --output outputs/vilmed_kvasir_maps
```

只下载3张固定Kvasir图及标签，来自锁定的JPEG镜像；官方Simula源本次返回503。记录镜像版本和原始哈希，JPEG掩码按灰度≥128转二值PNG。这些样本用于组件检查，不构成论文划分或性能基准。

## 4. 真实ViT-H前向与BID反向检查

```bash
python experiments/simple_vilmedsam/run.py smoke --manifest data/retina/CHASE_DB1/val.csv --maps outputs/vilmed_chase_val_maps --device cuda --output outputs/vilmed_smoke
```

只用清单第一张图：官方ViT-H+零初始化LoRA前向，随后释放编码器，使用真实特征对原版BID做一次参数更新。该命令不是完整LoRA训练。输出smoke.json和decoder_smoke.pt，后者只是接口检查产物，不能作为正式训练checkpoint。

## 5. 完整训练入口：先1轮，后决定预算

```bash
python experiments/simple_vilmedsam/run.py train --train-manifest data/retina/CHASE_DB1/train.csv --val-manifest data/retina/CHASE_DB1/val.csv --train-maps outputs/vilmed_chase_train_maps --val-maps outputs/vilmed_chase_val_maps --device cuda --epochs 1 --activation-checkpointing --output outputs/vilmed_chase_runcheck
```

只有LoRA和BID更新；冻结SAM其余参数，SAM原mask decoder不参与预测。BID有3,267,522参数；ViT-H rank=4 LoRA有655,360参数。损失为BCE+MONAI式squared-prediction Dice，物理batch=1、累积8，尾组按实际数量归一化。默认AdamW lr：LoRA 2e-4、BID 1e-4，weight_decay=1e-4，5% warmup+cosine。默认FP32；激活检查点不改模型结构。

`--amp`启用CUDA BF16，会改变数值设置，应单列记录。4090显存/速度不能从CPU验证推断。不要因论文用了4090就假定本实现已完成同等GPU验证。

修复过的训练入口使用显式互斥清单，数据审计拒绝重复图像/组重叠。图像用RGB，修正原加载器的cv2 BGR；保持最长边缩放与右/下补零。归因图和掩码同步翻转，暂不使用原代码中k未定义的旋转。原图评估恢复缩放/补边，报告native指标并额外记录原加载网格上的padded_grid_dice；二者不能与论文指标不加区分地比较。若有官方FOV，native评估使用它，训练损失按原版在完整补边网格计算。

正式预算不能机械照搬：作者跨数据集阶段为100轮，10-shot适配为50轮。直接在CHASE训练16张不是该跨数据集zero-shot协议；即使跑100轮，也只是迁移实验。

## 6. 必要对照与固定评估

至少同种子、同初始化、同预算训练两组：

- `--map-mode full`：信息瓶颈归因＋LoRA＋BID。
- `--map-mode zero`：归因置零，保留相同LoRA和BID参数。`constant`可作附加对照。

若要检查语义，应另用错误目标词生成归因图；同图同索引使用相同随机种子，再比较固定checkpoint下两组归因。不能只比较热图是否好看。移除归因同时改变的是整条空间条件，不能单独证明文本语义贡献。

```bash
python experiments/simple_vilmedsam/run.py evaluate --manifest data/retina/CHASE_DB1/val.csv --maps outputs/vilmed_chase_val_maps --init outputs/vilmed_chase_runcheck/best.pt --device cuda --output outputs/vilmed_chase_evaluation
```

输出逐图Dice、边界指标、clDice、细目标召回等。`--init`是权重初始化/评估，不恢复优化器训练状态；best/last保存LoRA和BID，不重复保存ViT-H主干。评估不得把测试集成绩用于挑epoch。

如果没有独立训练的纯视觉对照与可靠划分，即便完整方法超过旧SAM2基线，也不能归因为文本；骨干、可训练参数和解码器都变了。

## 本次实际执行记录（2026-11-18）

- 锁定作者源码并执行兼容修补；完整测试88项通过，Ruff、diff检查和pip check通过。
- 官方BiomedCLIP在CHASE 16张训练图、4张验证图完成每图10步信息瓶颈优化；验证图归一化响应标准差约0.153–0.178，响应不平坦不能证明定位正确。
- 3张镜像Kvasir样本完成论文polyp/β=0.1/ℓ=8设置的归因生成。σ对应var参数沿用2，但公开瓶颈实现未使用它。
- 真实SAM ViT-H在CHASE及Kvasir图均输出[1,256,64,64]，原版BID输出[1,1,256,256]，反向及一次AdamW更新通过；解码器末层最大参数变化约1e-4。
- 尝试真实ViT-H+LoRA+BID完整1步训练，启用激活检查点，进程因本环境8GB CPU内存OOM被终止（exit 137）；未产生完整训练结果。4090完整训练/速度/显存尚未验证。
- LoRA零初始化等价、梯度、更新和重载在缩小的作者SAM编码器结构上单独测试通过；这不能替代完整ViT-H训练。

摘要另见`experiments/simple_vilmedsam/validation_report.json`。本次没有可以用于与论文表格比较的Dice结果。
