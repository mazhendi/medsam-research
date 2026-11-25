# 候选C：双向局部交互＋原通道—空间联合选择

独立实验配置：`configs/polyp_pvt_slots_three_head_local_c.yaml`。只替换原三头模型的C；原A₁/A₂、单向B₃/B₄、深层Cat＋1×1融合、主头及两个辅助头保持不变。原Full仍为主对照，不覆盖原配置或训练结果。

## 对应模块图的实际计算

1. E、U通过原C的独立1×1投影得到E′、U′。U使用现有resize方法对齐到E网格。
2. 双向局部交叉注意力：E′生成Q，U′生成K/V，得到M(U→E)；另一个独立实例交换来源得到M(E→U)。每组Q/K/V均为1×1 Conv→DWConv3×3，维度32；两个方向与各投影不共享参数。
3. 每个位置只在另一路的3×3邻域计算单头空间注意力。logits为Q·K/sqrt(32)，沿9个邻居Softmax，然后聚合V。图像边界以外邻居在Softmax前屏蔽。Q/K/V卷积本身使用普通零填充，不进行跨边界循环。半精度时相似度与Softmax使用FP32。
4. 两路各用独立DWConv3×3提取本路局部结构，与对应跨路消息Concat，送入独立通道MLP：1×1(64→64)→GELU→1×1(64→32)，得到E*、U*。
5. 保留原C的selector结构：Concat(E*,U*)→3×3 Conv+BN+ReLU，送入全局通道支路与局部空间支路。两路logits广播相加，对两个来源Softmax。最终仍为1×1(W_E·E* + W_U·U*)。

352输入时，E、U、输出Z均为32×44×44。没有新增外层残差、边界预测损失、第四个头、频域或可变形采样。这是对跨层局部关系的候选建模；它不保证几何对齐，也不保证ETIS提升。

继承原C投影、选择器与最终投影的**结构**，新实验从头训练这些任务组件，并非自动载入旧C的已训练参数。原C作为默认`c_variant: original`保留；新配置使用`c_variant: bidirectional_local`。w/o C不能同时启用该变体。检查点加载检查c_variant，旧权重缺省按original解释。

## 训练与推理

与原三头一致：1450张训练图、352基础输入、batch16、99轮、seed42、AdamW lr=1e-4和weight_decay=1e-4、梯度裁剪0.5。`augmentation: none`，保留256/352/448三尺度更新，辅助权重[0.5,0.5]；推理仅主头、352输入。32通道时，C参数由22,786增至44,162（增加21,376）；这是C模块参数，不是全网络参数。额外模块会增加计算，需要本机探针确认实际显存和耗时。

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_local_c.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_local_c.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_local_c.yaml --checkpoint runs/polyp_pvt_slots_three_head_local_c/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_local_c_raw --sizes 352 --normalization raw
```

不要用原三头last.pt当作这个变体的完整权重；从相同PVTv2-B2预训练重新训练。独立输出 `runs/polyp_pvt_slots_three_head_local_c`。测试使用与Full相同的完整798张清单；如果清单有_full等后缀，按当前评估协议采用相同后缀/清单目录，不得混入去重版本。

## 证据与边界

设计受跨层特征交互、边界信息融合论文启发，例如[CFA-Net (PR 2023)](https://github.com/taozh2017/CFANet)、[MEGANet (WACV 2024)](https://github.com/UARK-AICV/MEGANet)；本模块不是这些网络的原模块复现，双向局部交叉注意力本身亦不能据此宣称原创。待查重与实验验证。

当前CPU测试核对显式逐位置邻域参考、边界掩码、44网格接口、双向梯度、三头训练/单头推理及旧模型兼容。没有在此环境完成GPU精度训练。后续用同一协议对比Full、w/o C和新C，并记录参数量与计算成本；如有收益仍需区分跨层交互与额外参数的作用。
