# 4 × 4 scaling experiment

在 `assignment1-basics` 目录运行。现有模型和训练源文件均不修改；新增入口复用 Transformer、AdamW、交叉熵、采样、梯度裁剪和学习率调度。

```bash
# 查看计划，不训练
python scalling_exp/run_scaling.py

# 准备全部目录、配置和待运行图表，不训练
python scalling_exp/run_scaling.py --prepare

# 顺序运行全部 16 组；已完成的组自动跳过，中断的组自动续跑
python scalling_exp/run_scaling.py --run

# 只运行一个组合
python scalling_exp/run_scaling.py --run --models 8M --budgets 16

# 只重新生成两张图
python scalling_exp/plot_scaling.py
```

需要 Python 3.12/3.13 与 requirements.txt 中的依赖。

## 实验设计

每组独立初始化，不从其他 token 预算的模型接着训练。batch size 固定为 8、context length 固定为 256，每一步实际处理 2,048 个训练 token。token 预算指累计采样量，数据按原 `data_loading` 实现有放回采样，不等于不重复的 token 数量。

| Token 预算 | MAX_STEPS | 实际处理 tokens |
|---|---:|---:|
| 16M | 7,813 | 16,001,024 |
| 32M | 15,625 | 32,000,000 |
| 64M | 31,250 | 64,000,000 |
| 128M | 62,500 | 128,000,000 |

| 名义规模 | D_MODEL | NUM_LAYERS | NUM_HEADS | D_FF |
|---|---:|---:|---:|---:|
| 8M | 256 | 4 | 4 | 768 |
| 16M | 384 | 4 | 6 | 1152 |
| 32M | 512 | 6 | 8 | 1536 |
| 64M | 640 | 10 | 10 | 1920 |

其他默认设置取自本地 `training_together_simple.py`：包括学习率、warmup 比例、优化器和日志周期。每组 cosine decay 的终点等于该组 MAX_STEPS，warmup 按同一比例计算。精确参数、源码哈希和数据快照记录在 `experiment_manifest.json` 与各组 `config.json`。

模型直接采用你原组件中的初始化，包括 FFN 的 `nn.init.xavier_normal_`；实验入口不再次初始化或覆盖权重，原文件保持不变。

默认取准备实验时 `train.bin` 前 95% 作训练、后 5% 作验证，16 组共用这个固定划分。数据随后追加的 token 不会自动加入此实验。也可第一次准备时指定 `--data PATH --val-data PATH`；后续启动须使用同一参数。程序校验固定数据前缀的哈希，避免续跑时换数据。

验证使用固定的随机采样种子，同一批验证片段用于所有模型、所有评估。验证不消耗训练采样器的随机状态。每组最后一步必定评估。两张汇总图使用 **最终一步 Validation Loss**，不是训练过程中最小的 loss。只展示已完成组的数据；Pending 不会填成 0 或伪造数值。结果为单个随机种子，不包含置信区间。

## 输出

```text
scalling_exp/
  experiment_manifest.json
  results_summary.csv
  tokens_vs_validation_loss.png / .pdf
  validation_loss_heatmap.png / .pdf
  8m_model/
    8m_16mtoken/
      config.json
      console.log
      training_log.csv
      result.json
      checkpoints/
        latest.pt   # 定期覆盖，供中断续跑；完成后更名为 final.pt
        best.pt     # 最低验证 loss 的模型权重、配置和步数
        final.pt    # 最终模型、优化器、步数及 RNG 状态，可用于恢复
    ...
  16m_model/ ...
  32m_model/ ...
  64m_model/ ...
```

按 Ctrl+C 会在当前优化步骤完成后保存 latest.pt 并停止整个实验队列。再次运行相同命令会续跑。默认每 500 步覆盖保存一次 latest.pt；意外退出最多需要重做这一保存间隔内的步骤。源码、配置、数据与已有实验不一致时会拒绝混用，想更换设置请指定新的 `--output-dir`。

此仓库保存实验快照；结果表仅包含已完成的组合，尚未完成的组合显示 Pending。
每种模型规模、每个 token 预算都有独立的权重目录。例如 `8m_model/8m_16mtoken/checkpoints/` 和 `8m_model/8m_32mtoken/checkpoints/` 完全分开；其他模型同样分开，共 16 个独立 checkpoints 文件夹。

GitHub 仓库只包含 64M 模型的 .pt 权重（通过 Git LFS 存储）；其他模型保留配置、日志和结果，不上传权重。原始 training_together_simple.py 未上传，仓库中的实验入口直接读取已保存的实验默认配置。克隆后运行训练需要安装 Git LFS 并准备自己的训练数据；使用新数据时请指定 --data 与新的 --output-dir。

当前训练集位于仓库根目录的 data/train.bin，也由 Git LFS 管理。已有实验清单只使用该文件前 332,008,126 字节，追加在后面的数据不会改变这批实验。服务器克隆后先运行 git lfs pull，再从仓库根目录执行上述命令；脚本会验证冻结前缀，允许数据文件位于新的绝对路径。

在新服务器上克隆或更新仓库后，先安装 Git LFS 并运行 git lfs pull。然后从仓库根目录运行：

```bash
uv venv --python 3.13
uv pip install -r scalling_exp/requirements.txt
source .venv/bin/activate
python scalling_exp/run_scaling.py --run --models 64M
```

已有 64M/16M 检查点会自动续跑；后续三个 token 预算按顺序执行。运行前可用 python -c "import torch; print(torch.cuda.is_available())" 检查 CUDA。
