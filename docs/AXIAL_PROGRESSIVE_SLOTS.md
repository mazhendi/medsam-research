# 三头版的独立优化分支：轴向双向扫描 + 逐级融合

旧实验配置 `configs/polyp_pvt_slots_three_head.yaml` 完全保留，缺省行为仍为
单向 raster B 和对齐 Cat+1×1 融合。已跑出的结果仍对应旧配置。
新版不覆盖模型权重或实验输出；训练/预测输出目录已存在时入口会报错退出。
更新 Python 源码属于正常代码升级；旧算法由默认开关保留，不是复制整套仓库。

## 仅两项改变

1. `slots.b_scan: axial`：B内保留四组DW3/5/7/9、LN、Mamba-1和PW，保留
   `v_i=x_i+y_(i-1)` 递进。每组使用同一套SSM参数处理逐行正反向、逐列正反向；
   还原空间顺序后四向平均。不同组参数独立。每行/列是独立零初始状态序列，
   一维卷积也不跨行/列。不是标准SS2D，不声称新提出Mamba。
2. `slots.fusion: progressive`：替换三层一次Cat。
   `T3=J3(P4,P3)`，`U=J2(T3,P2)`。J内部将粗特征双线性调整到细特征尺寸，
   两层拼接64通道→1×1压到32→DWConv3×3→GroupNorm(8组)→GELU。
   两个J参数独立；不额外重复上采样。

A、C、投影层、三头监督接口、A/B外层残差关闭设置均保持。
辅助头仍接B3/B4原生22/11网格，不接T3或U；仅训练时执行。
最终输出仍由C后预测头产生。辅助权重仍为0.5/0.5，便于比较这两项结构改动。
训练协议仍为1450张、352基准输入、三尺度256/352/448、batch16、99轮、原AdamW配置。

## 独立配置及目录

| 配置后缀 | B | 融合 | 输出目录后缀 |
|---|---|---|---|
| 原three_head | raster | aligned | three_head |
| three_head_axial | axial | aligned | three_head_axial |
| three_head_progressive | raster | progressive | three_head_progressive |
| three_head_axial_progressive | axial | progressive | three_head_axial_progressive |

所有完整配置名均为 `configs/polyp_pvt_slots_<后缀>.yaml`。
新旧结构切换由检查点内配置核对；即使轴向B与原B的参数键形状相同，也禁止用错配置预测。
旧检查点未记录两个开关时按raster/aligned解释，支持原配置继续评估。
新版从同一ImageNet骨干初始化重新训练，不自动载入旧息肉权重。

## 运行合并版

```bash
git pull
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_axial_progressive.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_axial_progressive.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_axial_progressive.yaml --checkpoint runs/polyp_pvt_slots_three_head_axial_progressive/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_axial_progressive_raw --sizes 352 --normalization raw
```

若原基线评估为per-image，统一更换normalization和输出目录，不能混用指标口径。
先用probe验证本机后端、耗时与显存，再开始99轮；没有mamba-ssm也可走torch实现。
四方向不等于已验证运行成本，应在4090实测最大训练尺寸448、batch16的峰值。
建议先分别训练axial和progressive单项，确认作用后再比较合并版。
本分支未完成真实训练，不保证跨集收益。

## 检查记录

61项相关测试通过，1项可选集成测试跳过。另用真实PVTv2-B2随机初始化结构，
352输入、batch2，验证新版三头前向/反向及全部参数有限梯度；eval仅输出final。
直接从修改前Git提交加载旧类，与修改后缺省类使用同权重比较：参数键一致，
严格加载成功，输出逐元素完全一致。旧三头配置文件未修改。
尚无新版完整训练成绩，也未验证4090 batch16显存和吞吐。
