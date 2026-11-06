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

目标手机为小米17（12GB版本），设备采集及后续验证说明见 [docs/mobile-validation.md](docs/mobile-validation.md)。
在连接手机的Mac上执行 `python3 scripts/mobile_profile.py`，生成设备与开发工具基线；该命令不运行推理。
执行 `python3 scripts/mobile_benchmark.py` 构建并部署小型原生驱动/算子探测程序；结果不代表完整视频模型性能。
NNAPI在本机仅提供CPU；使用 `python3 scripts/mobile_benchmark.py --backend vulkan` 测试独立Adreno GPU计算路径。
成熟运行时测试入口为 `python3 scripts/mobile_runtime.py` 和 `python3 scripts/mobile_benchmark.py --backend ncnn`；
下载固定版本官方包并校验SHA256后，测试FP32/FP16的FFN、Attention及二维解码代理网络，详见手机验证文档。
使用 `python3 scripts/mobile_benchmark.py --backend candidate` 验证带图像缓存、音频条件和时序状态的
256×256候选计算图，默认连续160帧；这是随机权重结构验证，尚不能生成有用的人物视频。
对应的PyTorch模型、梯度冒烟测试和跨端权重校验见 [docs/student-export.md](docs/student-export.md)。
已审核教师视频的时间对齐、因果音频特征、小样本拟合及对比预览见 [docs/student-overfit.md](docs/student-overfit.md)。
保留原图细节的v1候选与同片段对照结果见 [docs/student-reference-v1.md](docs/student-reference-v1.md)；v1尚未接入手机导出。
针对闭嘴不动的局部嘴部解码与张口/闭口对照见 [docs/student-mouth-v2.md](docs/student-mouth-v2.md)；仍限同片段验证。
v2未训练音频预览和原始ncnn导出诊断见 [docs/student-v2-mobile.md](docs/student-v2-mobile.md)。
当前固定填充区、逐头Gemm的手机导出变体已通过数值验收，见 [docs/student-v2-runtime-fix.md](docs/student-v2-runtime-fix.md)；仍有GridSample CPU路径，首次输出超过2秒。
