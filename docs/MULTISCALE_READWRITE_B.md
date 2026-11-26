# 候选B：多尺度读写条件化Mamba

配置 `configs/polyp_pvt_slots_three_head_readwrite_b.yaml`；同时替换B₃/B₄为独立实例。原A₁/A₂、原C、深层Cat＋1×1融合和三头监督保持不变。原B与历史变体不覆盖。本版本尚未完成GPU精度验证或创新查重。

## 实际结构与参数来源

输入X为32×H×W，先通过1×1投影得到P，然后分两条路径：

- 内容路径：P展平为B×HW×32 → Linear输入投影产生内容x和门控z → 因果DWConv1D(k=4)+SiLU得到u。u通过低秩投影产生delta；沿用Mamba的delta_bias和Softplus。
- 多尺度条件路径：P分别经过DW1×1、DW3×3、DW5×5、DW7×7，每路32通道；Concat128通道 → 1×1 Conv128→32 → 展平＋LayerNorm32 → Linear32→2·d_state，分成B_t与C_t。

默认d_state=16，条件输出为B×HW×32，再分成两个B×16×HW张量。内容路径expand=2，内部64通道；delta为B×64×HW。B/C按状态维度生成，并在内部通道间广播，沿用标准选择性扫描的张量约定，不是每通道单独生成64×16的B/C。

A=-exp(A_log)和D保持原Mamba参数化。递推采用现有选择性扫描：

```
step_t = softplus(delta_t + delta_bias)
h_t = exp(step_t * A) * h_(t-1) + step_t * B_t * u_t
y_t = C_t h_t + D * u_t
```

SSM输出与SiLU(z)相乘，经Linear输出投影恢复32通道，恢复二维网格后经1×1输出卷积，再与原始X相加。默认`b_residual: true`。没有组间递进相加、尺度Softmax或额外辅助头。

内容和条件路径虽然职责分离，但都来自P；不能据此声称统计独立。改变条件输入时，内容u、z、delta计算不变；B/C发生变化。没有沿用原Mamba的联合x_proj产生B/C，该未使用层已删除，避免无梯度参数。

不改变扫描顺序，仍为单向光栅扫描；不是SS2D或轴向扫描。多尺度B/C依赖二维邻域，因此整个模块不是严格因果图像模型，虽内部序列卷积与SSM仍为因果形式。这里B_t/C_t是状态读写系数，不是边界概率或分割置信度。

## 运行

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_readwrite_b.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_readwrite_b.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_readwrite_b.yaml --checkpoint runs/polyp_pvt_slots_three_head_readwrite_b/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_readwrite_b_raw --sizes 352 --normalization raw
```

独立输出：`runs/polyp_pvt_slots_three_head_readwrite_b`。从同一PVTv2-B2预训练重新训练，不加载旧结构的完整权重。新配置使用`b_variant: multiscale_readwrite`，预测已有b_variant/b_residual校验，必须配套训练配置。

1450训练图、batch16、99轮、seed42、AdamW lr与weight_decay均1e-4、梯度裁剪0.5；基础352，保留256/352/448三尺度训练；无随机增强；辅助权重[0.5,0.5]。评估必须与Full使用同一完整798张清单、概率图口径、阈值及HD95坐标规则。

## 后端与验证

继续支持auto/torch/fused，读取既有mamba_ssm选择性扫描接口，输入相关B/C的形状保持该接口要求。CPU使用可微分并行前缀扫描；CUDA auto在扩展可用时使用fused，否则明确警告后回退。fused在CPU或缺少扩展时拒绝静默回退。

CPU检查包含：与显式逐步SSM递推的输出及内容/条件梯度一致、条件改变不改变delta、四尺度22/11网格所有参数梯度、外层残差恒等测试、三个预测头、旧变体兼容。此环境未进行GPU/fused数值测试或完整训练；先运行4090探针再长训，不能保证效果提升。

状态条件化并不自动构成原创性；应进一步比较标准Mamba、多尺度融合后标准Mamba、该读写条件化方案，并控制参数量和训练协议。
