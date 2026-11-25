# B1：三尺度交互与Mamba（三头、无随机增强）

新配置 `configs/polyp_pvt_slots_three_head_b1.yaml` 同时将B₃、B₄替换为独立B1实例，保留原A₁/A₂、原C、深层Cat＋1×1融合、三个预测头及原监督设置。原B代码与所有历史配置保留。

## 与确认图一致的计算

1. 1×1投影：32→48，均分三组，每组16通道。
2. 三组分别经过DWConv3×3、5×5、7×7得到L₃/L₅/L₇；无9×9分支。
3. 各尺度独立1×1投影得到q/k/v，维度均为16。同一来源的k/v由两个目标复用，不是所有尺度共享权重。
4. 每个目标i仅查询另外两个尺度j/k，逐空间位置计算q_i·k_j/sqrt(16)，在另外两个尺度间Softmax。没有空间N×N注意力，也不查询自身。半精度时logits和Softmax使用FP32。
5. m_i=a_ij·v_j+a_ik·v_k；Concat(L_i,m_i)为32通道，经独立1×1卷积32→16得到T_i。三个T分别计算，不是复制同一融合结果。
6. 每路T：Flatten＋transpose→LayerNorm(16)→独立Mamba-1→恢复二维网格→独立1×1卷积16→16。
7. 三路Concat为48通道，最终1×1卷积48→32。B₃输出32×22×22，B₄输出32×11×11（352输入）。

没有Channel Shuffle、组间递进相加或外层输入输出残差。跨尺度加权聚合属于信息聚合，不是残差旁路；Mamba内部标准状态递推、D项和SiLU输出门控保留。三路均为单向光栅扫描，非SS2D、非轴向扫描。Mamba参数d_state=16、d_conv=4、expand=2，16通道输入对应内部32通道。

## 运行

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_b1.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_b1.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_b1.yaml --checkpoint runs/polyp_pvt_slots_three_head_b1/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_b1_raw --sizes 352 --normalization raw
```

独立训练输出 `runs/polyp_pvt_slots_three_head_b1`。需要从同一PVTv2-B2预训练初始化重新训练，不能直接加载旧Full完整权重。新检查点校验b_variant；原检查点缺省original，保持旧配置兼容。若完整测试清单使用独立目录/后缀，评估时与Full采用同一清单，不能混入去重版本。

训练沿用1450图、batch16、99轮、AdamW lr=1e-4/weight_decay=1e-4、梯度裁剪0.5、seed42、基础352和256/352/448三尺度更新。`augmentation: none`，无随机裁剪/旋转/翻转/颜色扰动。辅助权重[0.5,0.5]，推理仅主头。A/C都不是之前失败的新变体。

## 验证与局限

CPU测试检查跨尺度权重仅覆盖另外两路、自身特征Concat保留、不混合空间位置、22/11网格梯度、无外层残差、三头与配置差异、历史路径兼容。尚未在此环境完成GPU训练，先在4090运行探针确认显存和耗时。

按当前默认Mamba参数，单个原B为8,992参数，单个B1为19,552参数，两个替换位置合计增加21,120参数。内部宽度由原B的32增至48，且增加Q/K/V等参数；删除9×9不意味着更省计算。需要将精度和效率一起评估。跨尺度注意力本身是已有机制，不能仅据结构改动宣称强原创性。后续可用独立三组、相同宽度无交互作为控制，区分交互收益与容量收益。
