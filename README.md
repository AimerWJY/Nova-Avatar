# Nova-Avatar

面向单张 RTX 4090 24GB 的流式半身人物生成实验。目标为服务预热后首播不超过 2 秒，保留口型、表情、头肩及半身手势。

当前阶段：建立小规模数据集与质量审核流程，尚未启动训练或教师生成。

云服务器统一入口：

```bash
bash scripts/cloud.sh init
bash scripts/cloud.sh install-deps
bash scripts/cloud.sh check
bash scripts/cloud.sh model-check
bash scripts/cloud.sh model-repair
```

首次及更新执行顺序、各命令副作用见 [docs/cloud-setup.md](docs/cloud-setup.md)。目前入口不启动推理或训练。

数据准备说明见 [docs/data-preparation.md](docs/data-preparation.md)。
