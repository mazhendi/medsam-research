# 内外对照边界与边界约束跨尺度融合（当前研究分支）

状态：**V2 修正为在已有基线上添加模块**。沿用原 SAM2 掩码解码器和 task tokens，零初始化特征门控保证初始预测与基线一致。没有新版完整数据集训练结果或新颖性证明。

V1 曾替换为随机初始化的自定义解码器；该版本用户 30 轮最佳验证 Dice 0.8208，低于基线 0.8633，不能将全部差距归因于某一个结构因素。该结果不是“在原基线上添加模块”的公平对照。V1 保留只为重载/评估旧实验；不能将它的权重当作 V2 继续训练。
可编辑 PPTX 通过 [cell_su7](https://github.com/yrui-cmd/cell_su7) 从自绘 SVG 导出；未使用付费识图或去水印 API。

旧的区域—形状方案保留在 `region_shape.py`，二者互不替代检查点；本方案不使用隐式形状候选、文本或 SAM 稠密/稀疏提示适配。

![实现结构](figures/boundary_contrast.png)

[矢量 SVG](figures/boundary_contrast.svg) · [矢量 PDF](figures/boundary_contrast.pdf) · [三页可编辑 PPTX](figures/boundary_contrast_all.pptx)

## 独立模块图

![内外对照边界模块](figures/boundary_contrast_module_b.png)

![边界约束跨尺度融合模块](figures/boundary_contrast_module_c.png)

绘图源文件：`tools/draw_boundary_contrast.py`。可选安装 `cairosvg` 后运行，重新生成三组 SVG/PNG/PDF，不改变训练依赖。图中使用同名端口减少长线交叉。

## 1. 问题、模块与可检验的贡献

### 模块 B：内外对照边界模块（Inner–Outer Contrast Boundary Module）

问题假设：息肉与邻近黏膜外观相似时，单点特征不足以区分目标边缘两侧。

- 冻结 SAM2 产生三尺度特征，分别用 1×1 卷积投影至 32 通道。
- 深层投影特征经过可训练粗分割头生成 logit，再上采样并 sigmoid 得到 `p0`。
- 浅层与上采样中层投影特征取均值，作为边界模块输入 `F_s`。
- 局部分支：深度 3×3 卷积 → 1×1 卷积 → GroupNorm → GELU。
- 几何分支：由于 p0 在目标内部较大，**外法向取负梯度** `n = -∇p0 / sqrt(|∇p0|² + 1e-12)`。
- 在特征网格上以 r∈{1,2,4} 像素，配对采样 `F_s(x-rn)` 与 `F_s(x+rn)`；其差值拼接后经 1×1 卷积、GN、GELU 编码。
- 可靠性 `rho = |∇p0|² / (|∇p0|² + s²)`，乘各半径有效区域的均值；用固定保守半径边缘范围抑制图外采样，避免浮点误差引发边缘有效性跳变。
- `a = rho * sigmoid(Conv([local, contrast, rho]))`，`B = (1-a)*local + a*contrast`。方向不可靠时退回局部分支；没有 GT 法向或 GT 边界输入。

**可写入方法草稿的贡献候选：**设计一种预测几何条件化的内外成对采样模块，在多个距离上比较边界两侧特征，并以方向可靠性调节其与局部细节的融合。

限制：粗预测错误会让方向错误；低梯度的平滑假阴性区域可能根本没有可靠方向。本模块不能凭空恢复所有漏检目标，方向可靠性也不是经过校准的不确定性。

### 模块 C：边界约束跨尺度融合（Boundary-Constrained Cross-Scale Fusion）

问题假设：深层特征上采样后缺少边界细节，直接融合可能跨越目标边缘混合特征。

按 F3→F2→F1 逐级融合；每对尺度内部**并行**计算：

1. 语义引导细节：`D = F_shallow * sigmoid(Conv1x1(U(F_deep)))`。
2. 边界约束语义：将 B 投影为标量门控 `e = sigmoid(Conv1x1(B))`。对上采样深层特征执行 3×3 邻域归一化聚合：
   `a_xy = exp(-lambda*|p0(x)-p0(y)| - |e(x)-e(y)|)`，
   `A(x) = Σ_y a_xy U(F_deep)(y) / Σ_y a_xy`；之后 Conv3×3 → GN → GELU 得到 S。
3. 竞争融合：`g = sigmoid(Conv1x1([D,S,B]))`，`F_out = g*D + (1-g)*S`。

边界约束来自**预测概率差和学习的边界门控差**，不是对原始特征做硬切割。此处“语义引导”表示深层视觉特征，不涉及文本。

**可写入方法草稿的贡献候选：**在跨尺度融合中构建语义引导细节与边界约束语义两条并行路径，以预测边界相关的邻域亲和度控制特征传播，并通过互补门控融合两路结果。

限制：跨边界抑制也可能保留粗预测错误；需与普通局部平滑、无约束融合以及参数/计算匹配结构比较。

### 与所读 WeakPolyp-SAM 的关系

[WeakPolyp-SAM, KBS 2025](https://doi.org/10.1016/j.knosys.2025.113701) 使用 CFM/LCD 和涂鸦监督、SAM 在线伪标签协作训练。本分支只是借鉴“具体问题对应结构模块”的研究组织方式；没有移植 CFM/LCD，也没有实现其弱监督协议。我们的当前任务仍是完整掩码监督。不能用该论文的结果为本分支的新颖性或性能背书。

### 在原基线上接入：零初始化特征门控

原始三尺度 decoder-interface 特征 Fi 始终保留，包含已训练的 conv_s0/conv_s1 投影，以及深层的 no-memory embedding。

边界增强分支输出 Ri，各自通过 Pi(1×1 卷积) 恢复到原始通道数/网格，并独立接回：

`Fi* = Fi + alpha_i * Pi(Ri)`，`alpha_i` 初始为 0。

- F1/F2/F3 三个旁路独立，不将不同尺度连接为一条总线。
- 最终直接调用**已有、已训练的 SAM2 掩码解码器**，使用原 task tokens、no-mask embedding 和 positional encoding。没有新建自定义分割解码器，也没有最终 mask-logit 相加。
- `alpha=0` 时，特征和最终输出应与加载的基线一致。训练入口在首次更新前执行检查，写入 `identity_check.json`；失败则停止。
- 初始 Pi 保持随机初始化，只有 alpha 为零。若二者同时为零，会阻断主要学习路径。第一个更新中，alpha 和粗分割头可以学习；模块 B/C 通常要等 alpha 离开零后才获得最终分割损失的非零梯度。不要用第 1 步 B/C 梯度为零判断断链。
- 默认 `train_decoder: false`：冻结整个原基线，训练新模块、Pi 和 alpha。
- 可单独设计 `train_decoder: true` 对照：保留并微调原掩码解码器，继续冻结编码器、conv_s0/conv_s1 和 task tokens。配对评估使用一份原解码器的冻结副本，确保基线指标不随微调漂移。此设置会增加显存，不能与冻结解码器实验混报。
- 训练/预测每图 1 次编码 + 1 次原 SAM 掩码解码；配对评估同一编码下增加 1 次原基线参考解码。
- 损失仍为最终 BCE+Dice，加 `0.2 ×` 粗分割 BCE+Dice。粗 logit 上采样到标签网格计算损失，遵守 valid/FOV。没有 GT 边界或方向输入。
- Radii 单位为最细特征网格像素。默认 1024 输入/256 特征网格下 r=1,2,4 约对应模型输入的 4,8,16 像素。

此处的特征门控是接入与初始化方式，不单独包装成主要创新点。主要贡献候选仍是模块 B 的内外对照与模块 C 的边界约束融合。

## 2. 在 4090 上运行

继续使用已有 `light` 配置和自己的**息肉**基线权重。相同权重/划分/评估协议下，第 0 轮新版 Dice 应与原基线一致（你的当前报告为 0.8633），不应从 0.69 重新学习。使用下方 `_v2_` 新目录，保留旧 30 轮结果。不同任务权重不要混用。

先验证小样本可学习性（只取训练数据，不作为正式模型或性能证据）：

```bash
medsam probe-boundary-contrast \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --mechanism-config configs/boundary_contrast.yaml \
  --output outputs/polyp_bc_v2_probe_seed42 --steps 100 --samples 2 --learning-rate 0.001
```

检查 `probe.json` 中训练损失、训练 Dice、`head.alpha`、`head.coarse`，以及后续步骤的 `head.boundary`/`head.fusions`/`head.output_projections` 梯度。尚未提供固定的通过阈值，不应只看单个指标微升。

正式先跑 30 轮筛查：

```bash
medsam train-boundary-contrast \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --mechanism-config configs/boundary_contrast.yaml \
  --output outputs/polyp_bc_v2_full_seed42 --epochs 30
```

继承 YAML 中训练 LR、batch size、AMP、增强和种子；probe 的 `0.001` **不会**自动覆盖正式训练 LR。若显存不足，降低复制后的配置里的 batch size。当前环境只做 CPU/API 检查，尚未测量 4090 显存。

按验证集决定是否追加独立、预先设定的 100 轮实验，不使用五个测试集选轮数或超参。不支持断点续训，延长轮数需要新输出目录重新训练，不能把两次运行冒充同一次连续训练。

```bash
medsam evaluate-boundary-contrast \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --checkpoint outputs/polyp_bc_v2_full_seed42/best_boundary_contrast.pt \
  --manifest data/your_held_out_test.csv --output outputs/polyp_bc_v2_test_seed42

medsam predict-boundary-contrast \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --checkpoint outputs/polyp_bc_v2_full_seed42/best_boundary_contrast.pt \
  --manifest data/your_unlabeled.csv --output outputs/polyp_bc_v2_predictions_seed42
```

V2 `best_boundary_contrast.pt` 的 kind 是 `boundary_contrast_sam2_v2`。V1 kind `boundary_contrast_sam2_v1` 明确重载为 custom_decoder 架构；旧检查点不会被静默解释成新模型。检查点绑定 baseline SHA256、结构配置、评估阈值和训练数据审计指纹；不能用基线 `evaluate` 误加载。输出配对 Dice/IoU/Boundary F1/precision/recall、逐图改正/新增错误统计、粗预测方向与可靠性诊断。指标遵循项目固定阈值协议；并非自动复现每篇论文的指标工具。

## 3. 必须补齐的对照

复制 `configs/boundary_contrast.yaml`，只改变 `mode`，其他数据/预算/初始化种子相同：

| mode | 移除什么 | 检验什么 |
|---|---|---|
| full | 无 | 完整原型 |
| no_contrast | B 仅使用局部分支 | 定向内外对照是否有贡献 |
| no_constraint | 语义路径不做边界亲和度聚合 | 邻域约束是否有贡献 |
| equal_fusion | g 固定 0.5 | 学习互补门控是否必要 |
| no_boundary | B=0，同时关闭语义邻域约束 | 相同原 SAM 解码器与特征门控下的无边界对照 |

这些消融没有严格匹配实际计算，部分参数会不参与 forward，不能称为参数/算力匹配对照。还需普通 3×3 平滑、非定向同预算采样、相同 Pi/alpha 的普通多尺度适配器等对照；当前尚未实现。完成多种子与跨域测试前，不写“首次”“显著提高泛化”“解决边界模糊”或“SOTA”。

论文的合理结构是“问题假设 → 模块 → 计算规则 → 对照结果”，不是用模块名称替代新颖性证据。需进一步检索定向边界采样、双侧特征对比、边缘感知亲和度/双边滤波及跨尺度门控相关工作。

## 4. 实现检查记录（2026-11-19）

- V2 全量 71 项测试通过，Ruff 和 diff 检查通过。新增原解码器身份、门控梯度、CPU bf16、旧 checkpoint 兼容和微调参考不漂移检查。
- V2 真实 1024 输入首次更新前 `max_logit_difference = 0.0`，完成训练/重载/评估/无标签预测/探针；这仅是 API 和梯度检查，不是精度证据。
- 检查内容包括原特征/输出初始一致性、alpha 首步梯度与模块后续梯度、旧检查点显式重载和微调时冻结参考基线。当前 CUDA/4090 性能尚未验证。
- 检查了法向符号与特征网格采样距离、低梯度回退、保守边缘有效域、邻域约束的跨边界混合、梯度链和无 logit 残差。
- 流程测试覆盖训练、绑定检查点重载、配对评估、无标签预测、训练集探针、无效标签忽略。
- 真实官方 SAM2 Small 64 输入检查了基线参考复现、批量/逐图一致性和冻结参数。
- 真实官方 Small 预训练权重、1024 输入、CPU、3 张仓库自带 Kvasir 示例，完成 1 轮训练、重载、评估、无标签预测和 2 步探针。此处基线是带随机 task token 的 API 检查权重，**不是用户训练好的息肉基线**；不报告其 Dice 作为研究效果证据。
- 尚未进行完整数据集训练、4090 CUDA/bf16 显存和吞吐测试、跨数据集评估、多种子实验或新颖性证明。

最初讨论的概念图保存在 [initial_concept.png](figures/boundary_contrast_initial_concept.png)；图中的“not yet implemented”是绘制当时的状态，部分算子为概念简写。该图还代表早期 V1 概念而非 V2 接入方式。当前实现以本页公式、代码和上方重新生成的 V2 结构图为准。
