# 修改方向：在原Polyp-PVT拓扑内进行接口替换

连续整体替换没有超过基线，本分支先保留已验证的CFM聚合、44网格双头、双头监督、
两头logit相加的预测协议，只改变明确的模块接口。新旧版本相互独立。
尚无本分支完整训练成绩，不保证超过基线。

## 实际修改

```text
F1 ─→ 原CIM 或 A(64通道) ─→ 原1×1+降采样 ─→ 细节引导 ─┐
                                                       ↓
F2 ─→ 原1×1 ────────────────────┐                  原SAM 或 C ─→ 原头2
F3 ─→ 原1×1 ─→ 可选B(32通道) ───┼→ 原CFM → 语义特征 ──┘
F4 ─→ 原1×1 ─→ 可选B(32通道) ───┘          └→ 原头1

两个44×44预测头 → 上采样 → logit相加 → Sigmoid
```

A开启：替换F1的原CIM，直接处理64通道，不再同时加在F2上。
B开启：仅增强投影后的F3/F4，再送入原CFM；**并没有替换原CFM**。
C开启：替换原SAM，浅层引导作为X、CFM语义作为Y，在44网格做选择融合。
全部A/B开启时仍去外层残差，组间递进与Mamba内部计算保留。

原CFM、原预测头及所有通道投影在各组保留。没有88网格最终头，没有新D2/D3辅助头。
损失恢复为两个原区域头structure loss直接相加；推理恢复两头logit相加。
A/B/C内部算子沿用当前实现，此次核心修改是插入接口和整体拓扑，而非声称提出了新模块。

## 配置与顺序

| 配置 | 实际改变 |
|---|---|
| polyp_pvt_slots_control.yaml | 全部关；原Polyp-PVT前向图 |
| polyp_pvt_slots_a_only.yaml | 仅CIM→A，其他原模块保留 |
| polyp_pvt_slots_b_only.yaml | 仅在CFM前加入B |
| polyp_pvt_slots_c_only.yaml | 仅SAM→C |
| polyp_pvt_slots_full.yaml | 上述三个变化同时开启 |

同权重下，control已在256/352/448输入上与固定官方实现逐元素完全一致。
因此不必再专门训练control来验证其图结构；已有原版结果仍是主基线。
先跑一个单项，例如a_only，再决定是否继续其他组；暂不建议直接再次训练full。

## 运行

依赖、预训练权重、1450训练清单仍沿用现有设置。训练保持99轮、batch16、AdamW1e-4、
weight_decay1e-4、FP32、三倍率、种子42、固定末轮。所有编码器/模块参数参与训练。
初始化来自相同ImageNet骨干；没有偷偷从基线息肉权重热启动新分支。

```bash
git pull
medsam probe-abc-pvt --config configs/polyp_pvt_slots_a_only.yaml --batch-size 16 --image-size 448
medsam train-polyp-pvt --config configs/polyp_pvt_slots_a_only.yaml
```

a_only不含Mamba，不需要mamba-ssm；b_only/full含Mamba，后端行为与原ABC说明一致。
输出`runs/polyp_pvt_slots_a_only/last.pt`，不覆盖原模型。

```bash
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_a_only.yaml --checkpoint runs/polyp_pvt_slots_a_only/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_a_only_raw --sizes 352 --normalization raw
```

若基线用per-image，此处也用per-image并另设输出目录。测试清单和评估口径保持相同。
新架构名`polyp_pvt_slots_b2`；不同配置的开关和训练记录在预测时会核对。
control状态字典键与原模型相同，但项目检查点架构标签不同，不直接混用checkpoint文件。

## 证据边界

对照仍需区分结构和容量变化。A从64通道开始、C替换锚点图聚合，均不是参数量严格匹配。
本分支主要排除整体拓扑、头分辨率、辅助损失和推理组合同时变化的混杂。
即使某个单项提升，也需要重复种子及机制对照后才能确定有效性。

## 检查记录

全套195项测试通过。真实官方PVT结构的control在256/352/448输入与原版同权重输出
逐元素一致；a_only和full在真实PVT结构（随机初始化）、352输入、batch2下前向与反向通过，
无缺失或非有限梯度。以上不代表已完成训练或验证4090 batch16显存。

## 最新变体：去掉原CFM

`configs/polyp_pvt_slots_no_cfm.yaml`开启A/B/C，并设置`slots.cfm: false`。
模型中不注册原CFM，不执行其级联乘法与多层卷积。
F2/F3/F4的投影特征分别为32×44×44、32×22×22、32×11×11，
较粗两层直接双线性对齐到44网格，三层拼接96通道，再用单个1×1卷积压到32通道。
这是汇合三层特征的最小连接操作，不是改名后的原CFM。

浅层仍为F1→A→32通道→44网格引导，C融合该引导与深层汇合特征。
两个44网格预测头、双头结构损失与logit求和推理保留，A/B外层残差仍关闭。
C是自定义选择融合，不保留原SAM；编码器各层投影与原预测头保留。
源码中的`out_CFM`沿用原预测头参数名，不代表CFM模块仍在。

```bash
medsam probe-abc-pvt --config configs/polyp_pvt_slots_no_cfm.yaml --batch-size 16 --image-size 448
medsam train-polyp-pvt --config configs/polyp_pvt_slots_no_cfm.yaml
```

保存到`runs/polyp_pvt_slots_no_cfm/last.pt`，其他配置继续保留有CFM的原行为。
预测需使用相同no_cfm配置：

```bash
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_no_cfm.yaml --checkpoint runs/polyp_pvt_slots_no_cfm/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_no_cfm_raw --sizes 352 --normalization raw
```

概率归一化继续与基线一致；若用per-image请同步更改参数及输出目录。
本配置仅相对`polyp_pvt_slots_full.yaml`去掉原CFM，其余训练和模块开关一致。
尚无完整训练结果，不能把结构删除直接认定为性能提升。

无CFM变体检查：39项相关测试通过；真实PVT结构（随机初始化）、352输入、batch2
前向/反向通过，全部参数梯度有限；确认不注册CFM模块。尚未验证4090 batch16峰值及训练收益。
