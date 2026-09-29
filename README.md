# From-0-to-1-my-own-model-base-on-cs336
基于cs336手写大模型架构以及训练基座，后续会根据学习进度加入训推加速等infra优化，模型架构改进，实验对比，以及多模态架构等更多内容

## CPU 单元测试

在 Linux / WSL 中运行：

```bash
uv venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

测试使用临时生成的小数据，不需要下载 TinyStories、GPT-2 词表或旧的快照文件。
目前只覆盖已接通测试适配器的组件；梯度裁剪、批次采样、检查点和 tokenizer
等尚未完成的接口不包含在这组测试中。
