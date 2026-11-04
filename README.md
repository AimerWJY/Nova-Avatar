# Nova-Avatar

面向单张 RTX 4090 24GB 的流式半身人物生成实验。目标为服务预热后首播不超过 2 秒，保留口型、表情、头肩及半身手势。

当前阶段：数据准备与DiffSynth低显存教师推理基线，尚未启动蒸馏训练。

云服务器统一入口：

```bash
bash scripts/cloud.sh init
bash scripts/cloud.sh install-deps
bash scripts/cloud.sh check
bash scripts/cloud.sh model-check
```

模型默认直接使用 `/model/HuggingFace/Wan-AI/Wan2.2-S2V-14B`。共享权重完整时无需model-repair；旧的/cloud模型副本不会被自动选中。/cloud仍保留环境、缓存和输入输出。

首次及更新执行顺序、各命令副作用见 [docs/cloud-setup.md](docs/cloud-setup.md)。上述准备命令不启动推理或训练。

环境及模型检查通过后，运行第一条离线教师视频：

```bash
bash scripts/cloud.sh install-diffsynth
bash scripts/cloud.sh infer --dry-run
bash scripts/cloud.sh infer
```

默认输入为数据盘inputs/person_01.png和inputs/normal.wav，输出为480×640、16FPS、最多5秒、40步。低显存配置不是实时部署配置。安装与输出说明见 [docs/inference.md](docs/inference.md)。

数据准备说明见 [docs/data-preparation.md](docs/data-preparation.md)。
