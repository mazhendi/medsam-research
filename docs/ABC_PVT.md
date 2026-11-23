# 最新A/B/C模型：实现与运行

状态：候选结构已实现并完成结构检查，尚无完整数据训练成绩。
原Polyp-PVT、SCSI/BCSC、SDS/ADF/RTR各版本保留为独立配置。
此处A/B/C对应最近确认的STPNet-inspired A、HAMB-inspired B、SSFM-inspired C，
不是旧SDS/ADF/RTR重命名。

## 整体连接

PVTv2-B2输出F1/F2/F3/F4，352输入下网格88/44/22/11，通道64/128/320/512。
各自独立1×1投影为32通道。F1/F2接A，F3/F4接B。

```text
F1 → 1×1 → A → E1 ─────────────────────────────→ C1 → D1 → 最终头
F2 → 1×1 → A → E2 ─────────────────→ C2 → D2 ↑
F3 → 1×1 → B → E3 ─────────→ C3 → D3 ↑
F4 → 1×1 → B → E4 → 初始化 → D4 ↑
```

每个C内部先把更深层解码特征上采样到当前层网格，两个输入单独投影后融合。
C3实际读取E3/D4，C2读取E2/D3，C1读取E1/D2；不是三个C共享D4。
每级A/B/C结构相同但参数独立。全部编码器及解码参数训练。
CIM、CFM、原相似性聚合模块均不进入新网络。

## A：空间—上下文增强

保留STPNet-inspired两支路及外部残差结构。输入1×1投影后：

- 空间分支：DW3×3→1×1→Sigmoid，再与投影特征相乘。
- 上下文分支：DW3/5/7三分支相加→DW3×3，再与投影特征相乘。
- 两分支拼接→1×1→加回输入。

上下文分支没有额外Sigmoid；没有偷偷加入旧版SDS的五分支空间Softmax。
这里用稠密DW核替换原型的大膨胀卷积，不声称与原作者完全同构或已改善效果。

## B：局部卷积前置的分组Mamba

1×1投影后分四组，组内依次DW3/5/7/9→LayerNorm→Mamba-1→1×1混合。
第二组及以后，在局部卷积之前加上前一组的完整输出。
四组拼接→1×1融合→加回模块输入。

这里四组是**通道分组**，不是四级编码尺度；B仅放在F3/F4。
每组将二维特征按行展平为序列，采用**单向Mamba-1**。
并非SS2D、双向扫描或作者官方HAMB复现；论文与图中泛称Mamba时应以此明确规格为准。

Mamba包含真实选择性状态空间：输入投影、因果深度Conv1d、输入相关Δ/B/C、
稳定负A、选择性状态递推、D跳连、SiLU门控和输出投影。
参考Mamba-1数学定义：Gu & Dao, arXiv:2312.00752；本项目自行实现，没有复制第三方算法文件。
默认每组state=16、causal_conv=4、expand=2。

### 后端兼容性

`decoder.backend: auto`：CUDA且已有兼容mamba-ssm扩展时使用fused selective scan；
否则使用纯PyTorch关联前缀扫描，计算同一递推，不以普通卷积冒充SSM。
`torch`可强制便携实现；`fused`强制CUDA实现，缺少环境会明确报错。

便携实现不要求编译CUDA扩展，Windows也可运行；但它不是线性时间fused kernel，
采用O(log L)个并行前缀阶段，训练耗时和显存可能更高。
不要以“无需安装”误解为“和官方CUDA kernel速度相同”。
本环境没有验证fused路径；CUDA数值一致性与速度需在本机核验。

## C：已确认的通道—空间联合选择

X细层与Y深层分别对齐投影，拼接后单个3×3+BN+ReLU生成共同特征F。

- 全局分支：GAP→1×1→ReLU→1×1，产生gx/gy，各C×1×1。
- 局部分支：1×1→ReLU→1×1，产生sx/sy，各1×H×W。
- 广播相加得到Lx/Ly；在两路维度Softmax得到Wx/Wy，各C×H×W。
- 输出1×1(Wx×X′+Wy×Y′)。Wx+Wy=1。

被加权的是原投影X′/Y′，不是共同特征F。不再有额外多尺度、圆环、轴向注意力。
源码保留图中对称双流及一次共同选择。

## 训练与预测

安装依赖、vendor官方PVT代码及pvt_v2_b2.pth权重准备沿用docs/POLYP_PVT.md。
默认无需额外安装mamba-ssm就可运行便携路径。
同一ImageNet骨干初始化，从头训练新解码器，不加载旧版本的完整分割检查点。

先运行**最大训练尺寸的合成数据探针**：

```bash
git pull
medsam probe-abc-pvt --config configs/abc_pvt_full.yaml --steps 1 --batch-size 16 --image-size 448
```

探针包含前向、loss、反向、梯度检查、AdamW更新及CUDA峰值allocated显存，
不读医学数据、不写检查点；用于检查开销，不用于评价分割质量。
输出scan_backends记录实际选择的scan实现。若过慢/OOM，请保留输出和报错，
不要直接改batch后当作同条件实验；先确认是否用了便携后端。
合成数据探针不覆盖真实数据加载、长训练稳定性和全部运行开销。

```bash
medsam train-polyp-pvt --config configs/abc_pvt_full.yaml
```

352、1450张、batch16、99轮、FP32；三尺度实际256/352/448。
保持原AdamW lr1e-4、weight_decay1e-4、梯度逐元素裁剪0.5及无随机增强。
同一轮每尺度各一次更新，共26730步。末轮权重：
`runs/abc_pvt_352_full/last.pt`，不覆盖原实验。架构名`abc_pvt_b2`。
日志额外记录实际扫描后端；没有自动降低batch/切换精度/缩短训练。

损失：L(final)+0.5L(aux2)+0.5L(aux3)，L为原structure loss。
D2/D3有辅助头；**推理只用D1最终头**，不与辅助logit相加。
这与原版两头相加协议不同；完整模型间收益不可直接归给一个模块。

```bash
medsam sweep-polyp-pvt --config configs/abc_pvt_full.yaml --checkpoint runs/abc_pvt_352_full/last.pt --manifest-dir data/polyp_test --output outputs/abc_pvt_full_raw --sizes 352 --normalization raw
```

若基线使用per-image，本次也使用per-image并指定新输出目录；测试清单、阈值口径一致。
不以五测试集选择训练轮数或超参数。

## 对照配置

`abc_pvt_full.yaml`完整；`control`全关；`a_only`/`b_only`/`c_only`各单开；
`ab`/`ac`/`bc`为两两组合。各自独立输出目录。
A/B关闭时保留投影特征；C关闭时用常规3×3卷积融合。
对照共享相同初始化来源、最终头推理和辅助监督，但并非严格参数量匹配。
先确认full可正常运行，再根据结果选择消融；不必同时启动八组。

## 已完成检查

- 全套181项测试通过，原版与旧版流程回归通过。
- 便携选择性扫描在多个长度上，与逐步状态递推的输出及梯度一致（双精度检查）。
- A各分支、B通道组递进、C权重归一化/原特征选择，以及八种模块组合通过测试。
- 最新真实PVTv2-B2结构（随机初始化）在352、batch2下前向/反向通过，
  无缺失或非有限梯度；256/448前向通过。总参数24,993,609，实际使用8个torch扫描器。
- 这些是CPU结构检查，未完成医学数据训练、CUDA fused核对或4090 batch16峰值验证。

## A、B去外层残差的独立版本

使用`configs/abc_pvt_no_residual.yaml`，明确设置：

```yaml
decoder:
  # 其余字段见完整配置
  a_residual: false
  b_residual: false
```

A和B分别从`output = input + transform(input)`改为`output = transform(input)`。
仅去掉各模块最外层的输入直连相加：A的两条分支与乘法、B的组间递进相加、
Mamba选择性扫描内部D*u项全部保留；C、投影、解码器、监督及所有训练超参保持不变。
没有删除PVT编码器内部的Transformer残差。

```bash
medsam probe-abc-pvt --config configs/abc_pvt_no_residual.yaml --batch-size 16 --image-size 448
medsam train-polyp-pvt --config configs/abc_pvt_no_residual.yaml
```

新输出：`runs/abc_pvt_352_no_residual/last.pt`。原`abc_pvt_full.yaml`保留旧残差行为，
以便重现已完成的0.8484实验，不覆盖其目录或权重。

预测也必须使用no_residual配置，例如：

```bash
medsam sweep-polyp-pvt --config configs/abc_pvt_no_residual.yaml --checkpoint runs/abc_pvt_352_no_residual/last.pt --manifest-dir data/polyp_test --output outputs/abc_no_residual_raw --sizes 352 --normalization raw
```

若原实验用per-image，此处同步改成per-image。两种残差模式的参数键与尺寸相同，
因此预测会额外核对训练配置中的残差开关，防止旧权重在新计算图上静默评估。
旧检查点未记录开关时按旧行为true解释。去残差是否有效仍需同协议实验验证。
