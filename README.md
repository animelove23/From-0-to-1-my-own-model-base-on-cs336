# From-0-to-1-my-own-model-base-on-cs336
基于 cs336 手写大模型架构以及训练基座，后续会根据学习进度加入训推加速等 infra 优化、模型架构改进、实验对比和多模态架构。

## 4×4 Scaling：训练 token 与最终验证损失

![四种模型在不同训练 token 预算下的最终验证损失](scalling_exp/tokens_vs_validation_loss.png)

四种模型的验证损失都随训练 token 增加而下降；在每档 token 预算下，64M 模型的损失最低。64M 模型从 16M 到 128M token 的损失由 2.542 降至 1.576，继续增加 token 仍有收益，但每次翻倍带来的改善逐渐缩小。

## 4×4 Scaling：最终验证损失热力图

![四种模型与四档训练 token 的最终验证损失热力图](scalling_exp/validation_loss_heatmap.png)

16 组实验均已完成。热力图右下角的 64M 模型、128M token 组合达到最低最终验证损失 **1.576**；在相同 token 预算下，模型规模越大，损失越低。这些数值来自单个随机种子的最终一步评估，不是训练期间的最佳检查点损失。[查看完整结果表](scalling_exp/results_summary.csv)。

## 脚本用途

`cs336_basics/` 包含模型、分词器与优化器。`cs336_basics/public/` 中是可以公开的入口脚本；个人实验的原文件和具体参数不上传。

- [训练脚本](cs336_basics/public/training.py)：读取 token 数据，构建模型，执行训练、验证和保存模型检查点。运行 `python -m cs336_basics.public.training --config CONFIG.json`。训练集路径、模型结构和训练参数都由本地 JSON 配置提供；脚本中的 `REQUIRED_KEYS` 列出了必填项。
- [对话脚本](cs336_basics/public/chat.py)：读取模型检查点、词表和 BPE 合并规则，从提示词生成回复。运行 `python -m cs336_basics.public.chat --config CONFIG.json`。配置需提供 `checkpoint_path`、`vocab_path`、`merges_path` 和模型结构参数；输入 `quit` 退出。
- [数据集转 token 脚本](cs336_basics/public/tokenize_dataset.py)：把 UTF-8 文本逐行转换为 `uint16` token 的 `.bin` 文件，供训练读取。运行 `python -m cs336_basics.public.tokenize_dataset --input TEXT --output TOKENS.bin --vocab VOCAB.pkl --merges MERGES.pkl`。从已有分词断点继续时，增加 `--resume-checkpoint CHECKPOINT.json`，脚本会核对现有输出后追加写入。

分词断点记录文本转换进度；训练检查点保存模型与优化器状态，它们是两类不同的文件。请将个人配置、数据和权重保存在仓库之外。