# 当前基线验证范围

保留数据清单/二值标签/FOV/重复与分组审计、训练保存和权重加载、原图评估、无标注推理、视网膜导入及验证集诊断测试。普通测试使用小型网络检查流程，不代表SAM分割精度。

可选集成测试使用锁定的真实SAM 2.1 Hiera-Small架构、随机权重和64×64输入，核对batch一致性、任务token与解码器梯度及编码器冻结。不据此推断1024输入的4090精度、速度或显存。

本次清理不修改 model.py 或检查点加载格式；engine.py / data.py / config.py 新增 bf16 AMP 与训练增强选项（`training.augmentation`），检查点格式保持不变。息肉基线（1305 张训练、无增强与 light 增强两组）已在本机 4090 上完成训练与五组外部测试评估，权重与逐图结果保存在本地产物目录，不提交 Git。

运行命令：

```bash
ruff check .
pytest -q
```

未安装官方SAM2时会跳过相应集成测试。历史研究分支的验证记录可从Git历史查看，不能当作当前基线的实验结论。

2026-11-19 增加增强与优化选项后，本地 29 项测试全部通过（含新增 `tests/test_augmentation.py` 与当前环境可执行的真实 SAM2 集成检查），Ruff 和 git diff --check 通过。CLI 保留基线与数据/诊断命令；息肉配置（`polyp.yaml`、`polyp_full.yaml`、`polyp_full_aug.yaml`）路径解析通过。
