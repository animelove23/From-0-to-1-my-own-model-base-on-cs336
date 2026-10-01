# 4 × 4 scaling experiment：分析快照

这个目录只保存实验产出与分析说明，不包含训练数据、模型权重或训练入口。原始本地实验文件仍位于独立的 scalling_exp 目录，未被修改。

本次实验独立训练 8M、16M、32M、64M 四种模型，每种模型对应 16M、32M、64M、128M 四档训练 token。batch size 为 8，context length 为 256，每步处理 2,048 个 token。名义规模及精确超参数见各组 config.json，数据划分与源码哈希见 experiment_manifest.json。

汇总表 results_summary.csv 记录每组最终一步的 Validation Loss。tokens_vs_validation_loss.png/.pdf 显示不同模型随训练 token 增加的损失曲线；validation_loss_heatmap.png/.pdf 显示 4 × 4 的最终损失矩阵。Pending 表示该组合尚无最终结果，不应当被视为零损失。当前快照完成 12/16 组；四组 64M 模型尚未完成，因此图中没有它们的最终损失曲线。

各组目录保留 config.json、console.log、training_log.csv，以及已完成组的 result.json。日志记录训练过程，result.json 记录最终与最佳验证损失。结果来自单个随机种子，不包含置信区间。仓库不保存任何 checkpoints、.pt 权重或 train.bin 数据集。
