# Region–shape 窗口竞争与候选诊断

当前研究在 region-shape 上继续，基于远端 `e5a5953`；尊重该提交移除 BC 的决定。本次没有恢复 BC，也没有更换已训练的 SAM2 解码器或 task tokens。

先读[本次检索记录](research/REGION_SHAPE_LITERATURE_20261119.md)。窗口路由已有先例，本分支是未验证的受控扩展，不是已成立的创新或精度结论。

## 1. 只改变候选聚合粒度

原默认 `configs/region_shape.yaml` 保持整图竞争（`routing: global`）。新增：

- `configs/region_shape_window.yaml`：`routing: window`、`window_size: 8`、`local_strength: 0.5`。
- `configs/region_shape_window_control.yaml`：同一路径，`local_strength: 0`，精确退回整图竞争。

默认机制网格 64×64，每个窗口 8×8，即 8×8 个窗口；不是裁原图 patch，不重新编码。

对于候选 k、窗口 u：

1. 保留原整图能量 `Dk` 和整图权重 `pi_k=softmax(-Dk)`。
2. 在窗口内重新池化候选内部/外部视觉特征，并计算区域一致性、面积；复用**同一个**兼容性 MLP 得到 `Dku`。
3. `w_ku = softmax_k(-[(1-beta)Dk + beta Dku])`。beta 固定在 [0,1]，默认 0.5，表示整图先验与局部证据的折中，不是自动学到的校准概率。
4. 将窗口权重双线性上采样至机制网格，归一化后计算 `mu(x)=Σ_k w_k(x)H_k(x)`、`v(x)=Σ_k w_k(x)(H_k(x)-mu(x))²`。
5. 其余区域更新、场生成、辅助损失与联合提示不变。稀疏提示仍使用 pi_k，避免同时改动两个选择路径。

**没有新增可训练参数，参数名/形状与原版一致；增加了窗口内评分计算，不能声称相同 FLOPs 或相同时间。** `uniform_competition` 仍覆盖窗口权重，使其均匀。K=1 时方差为 0，窗口竞争无选择空间。

旧 checkpoint 缺少这三个字段时，明确采用 global 默认值。普通 evaluate/predict 仍按保存配置重建；不会把旧 checkpoint 静默切换到窗口模式。训练仍从同一冻结基线重新初始化机制，**不自动接着旧 region-shape best.pt 微调**，以便做同预算对照。

## 2. 先诊断已有 region-shape

路径替换为本机实际文件，优先用验证集。下面的 manifest 需与配置中的验证清单对应：

```bash
medsam diagnose-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --checkpoint outputs/polyp_rs_full_seed42/best_region_shape.pt \
  --manifest data/polyp_full/val.csv \
  --window-size 8 \
  --output outputs/polyp_rs_hypothesis_diagnosis_seed42
```

检查 `summary.json` 和 `per_image.json`：

| 字段 | 含义 | 不能怎么解释 |
|---|---|---|
| mean_pairwise_iou / probability_l1 | 候选重叠与概率差异 | 差异大不自动代表候选合理 |
| best_candidate_dice_oracle | 使用 GT 选整图最佳候选 | 不是可部署选择器，也不是最终 SAM 解码器性能 |
| fixed_window_oracle_dice | 给定二值候选、非重叠窗口网格下，最大化整图 Dice 的硬选择参考 | 不是软融合、重新训练或最终解码器的理论上界 |
| window_oracle_gain_over_best_candidate | 局部选择相比整图候选选择的空间 | 没有空间时不宜继续堆路由器 |
| global_consensus_dice / spatial_consensus_dice | 固定当前候选下，两种已定义聚合的占据图指标 | 不等于最终输出指标 |
| disagreement_error_auroc | 当前空间方差能否定位最终预测错误；0.5 类似随机排序 | 不是不确定性校准或临床风险评估 |
| mean_spatial_weight_std | 权重是否随位置改变 | 变化大不等于选择正确 |

候选占据概率先双线性恢复到**原始图像网格**，再使用固定配置阈值；只在 valid/FOV 内计算。没有把粗网格 max-pool 标签当精细边界评价。全对/全错时 AUROC 无定义，记录 null；候选全空的两两 IoU 可为 1，需结合前景量与 Dice 阅读。

硬窗口参考采用分式优化选择每个窗口的候选，优化整图 Dice，而非贪心最大化各窗口 Dice。窗口数按机制网格确定、映射到原图非重叠区域；不模拟双线性软权重的全部行为。

**这项命令读取标签，所有 oracle 只存在于独立诊断文件；forward/predict 不读标签或 oracle。** 诊断保留 source 数据审计；不要在测试集看 oracle 后选择结构再声称盲测。

## 3. 同预算对照

原 global 版本可以继续运行，旧权重不覆盖。先验证新分支可训练：

```bash
medsam probe-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --mechanism-config configs/region_shape_window.yaml \
  --output outputs/polyp_rs_window_probe_seed42 --steps 100 --samples 2
```

再用相同的 30 轮预算比较 global、window 和 beta=0 控制：

```bash
medsam train-region-shape \
  --config configs/polyp_full_aug.yaml \
  --baseline-checkpoint outputs/polyp_baseline_full_aug_seed42/best.pt \
  --mechanism-config configs/region_shape_window.yaml \
  --output outputs/polyp_rs_window_seed42 --epochs 30
```

global 使用 `configs/region_shape.yaml`；控制使用 `configs/region_shape_window_control.yaml`，分别指定新输出目录。除路由设置外保持增强、LR、种子、K、轮数、标签预算一致。先用默认 `train_decoder: false`；不要混入解码器微调，否则旧实现的配对“baseline”也会随同一个解码器改变，不能当原始基线固定参考。

仍保留原提示的零初始化：第 0 轮最终输出应与加载基线一致。正式效果看验证集 native Dice、bF1、逐图变化、运行时间和 corrected/damaged pixels，不看 oracle 分数选最好测试集。多种子和真正的测试集泛化尚待本机完成。

`states.json` 增加平均空间权重和其标准差；`*_states.npz` 增加 `spatial_weights`。其余 CLI、检查点名与数据协议不变。

## 4. 下一步决策

- 候选趋同且窗口 oracle 没空间：先研究候选生成，停止加复杂路由。
- 候选有互补而窗口模型不利用：检查权重与一致性评分是否被错误区域证据主导。
- 分歧与错误几乎无关系：暂停分歧驱动细化，尤其检查共同漏检。
- 只有验证集微升：继续多种子/逐图对照，不写显著改善或新颖性结论。

本次暂不加入多支撑候选头或高分辨率细化头，避免无法归因。局部选择可能破坏整体形状一致性；整图锚定和插值仅缓解，不保证拓扑。

## 5. 本次验证范围（2026-11-19）

- 全量 70 项测试通过；Ruff 和 diff 检查通过。
- 窗口权重的空间选择、单纯形约束、候选顺序置换、单窗口退化和 beta=0 精确退化均有检查；同一参数布局下 global/window 参数数相同。
- 原 checkpoint（没有 routing 字段）明确恢复为 global；训练/保存/重载/诊断闭环通过。
- 窗口硬选择参考在多个微型案例中与穷举整图最优 Dice 一致；AUROC 的并列分数、无错误类别和候选完全相同情形有检查。
- CPU bf16 的新窗口评分前后向通过，不等同于 4090 CUDA 验证。
- 官方 SAM2 Small 预训练权重、1024 输入、3 张仓库公开 Kvasir 示例，完成新窗口分支 1 轮训练、重载、评估、无标签预测、2 步探针和标签诊断；也实际加载了扩展前的 region-shape checkpoint。此处基线带随机 task tokens，仅用于 API 检查，**不是用户已训练的息肉基线**。
- 该 smoke 的验证选模仍为 epoch 0；它不证明新路由有精度收益，也不用于支持优于原版的结论。完整数据集、GPU 预算和多种子结果需要本机实验。
