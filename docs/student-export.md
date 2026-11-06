# 可训练候选模型与手机参数一致性

student/model.py是recurrent_conditioned_rgb_v0的PyTorch实现，结构对应
mobile/native/candidate_probe.cpp。当前只有246112个参数，容量是否足够生成自然人物尚未验证。
它是面向蒸馏的结构候选，不是从Wan的14B权重中剪出来的模型。

本阶段验证反向传播、优化器更新、checkpoint保存/恢复、参数映射，以及PyTorch和手机输出一致性。
本节的合成数据验证没有加载14B教师，也不评估人物视频质量。
随后新增的已审核教师视频拟合、音频特征和带音轨预览见[student-overfit.md](student-overfit.md)。

## 导出与梯度冒烟验证

训练/导出环境需要已有PyTorch及NumPy。云服务器复用当前GPU环境，不重新安装或替换平台PyTorch：

```bash
bash scripts/cloud.sh student-export --smoke-train \
  --output /cloud/nova-avatar/exports/student-parity-001
```

Mac本地可使用项目缓存中的独立环境：

```bash
# 本机验证使用PyTorch 2.9.1，独立环境不替换系统或云端PyTorch。
python3 -m venv --system-site-packages cache/student-venv
cache/student-venv/bin/python -m pip install --index-url https://pypi.org/simple torch==2.9.1
cache/student-venv/bin/python scripts/student_export.py --smoke-train \
  --output outputs/student-parity-001
```

--smoke-train用合成图片和音频特征展开2帧，反向传播一次并执行一次Adam更新。
检查所有参数有有限梯度、至少有参数被更新，并导出更新后的权重。
该操作只验证训练代码链路，不是知识蒸馏，不证明损失收敛或人物质量。
不加该参数则导出固定随机种子初始化的模型。
输出目录必须尚不存在，避免覆盖已有导出结果。

导出目录包含：

| 文件 | 用途 |
| --- | --- |
| checkpoint.pt | schema、结构标识、state_dict和训练来源；恢复时使用weights_only=True |
| candidate.bundle | FP32参数、合成输入、缓存身份特征及前8帧PyTorch参考输出 |
| manifest.json | 结构、权重槽形状、SHA256、来源、梯度检查和参考输出范围 |

导出已有checkpoint：

```bash
python scripts/student_export.py --checkpoint /路径/checkpoint.pt \
  --output /新的导出目录
```

默认包含160帧所需的输入特征，前8帧含完整参考RGB和状态。
--frames可设8到160；仍固定校验前8帧。图像为[1,3,256,256]、范围[0,1]，
音频输入为[1,8,40]合成浮点特征，初始时序状态为零向量[1,128]。
音频输入尚未定义真实Mel提取和归一化契约，不能直接把WAV字节传给模型。

## 在Mac连接手机验证

如果在云端导出，先把整个导出目录复制到Mac；手机测试本身在Mac执行。
只需manifest.json和candidate.bundle参与手机验证，checkpoint留作训练与重导出。

```bash
python3 scripts/mobile_benchmark.py --backend candidate \
  --bundle-manifest outputs/student-parity-001/manifest.json \
  --serial c9fb5e39
```

导出只有8帧时，手机命令同时加--iterations 8。
脚本先校验固定包名、字节数和SHA256，再推送到本轮/data/local/tmp专用目录。
原生加载器严格检查magic、版本、节点数、参数槽数量、每槽元素数量、有限值、帧数及文件结尾，
并限制整个bundle不超过32MiB，不接受任意网络结构或代码。

bundle是本项目候选图的固定二进制协议，**不是通用ncnn的.param/.bin模型格式**，
也没有验证ONNX、pnnx导出或Android APK加载。修改图结构时必须同步修改两端及协议版本。

使用同一套权重与输入，分别比较身份空间特征、身份全局条件、前8帧RGB及跨帧状态：

- 手机FP32 CPU相对PyTorch：abs_error <= 1e-5 + 1e-4*abs(reference)。
- 手机FP16存储GPU相对PyTorch：abs_error <= 1e-4 + 2%*abs(reference)，关闭FP16算术。
- --candidate-precision fp32使用全FP32 GPU，并采用与CPU参考相同的更严格阈值。
- 仍保留原有全部帧有限值、手机CPU/GPU对照及状态重置重放检查。

必须全部通过，报告才允许pytorch_equivalence_validated=true。
源checkpoint、权重和输入都是导出端给定的，手机不会再为测试另生成一组随机权重。
CPU参考在GPU性能测量后执行，不计入单帧延迟；来源为PyTorch的参考数据与手机CPU实现独立。
实际GPU内部精度不保证逐条为FP32；FP16存储与关闭算术选项仅描述观测与配置。

## 2026-11-05实际验证结果

Mac的PyTorch 2.9.1 CPU环境完成两帧反向传播及一次Adam更新，所有参数梯度有限，
18个参数张量发生更新。checkpoint恢复后重新导出的bundle逐字节一致。
使用该环境执行`python -m unittest discover -s tests`，50项测试全部通过，无跳过项。
系统Python未安装PyTorch时会跳过3项训练模型测试，不应将跳过视为已验证。

本机导出目录outputs/student-parity-20261105-01，bundle约8.22MiB。
来源标识purpose=synthetic_gradient_check，teacher_trained=false。
实机报告benchmark_20261105T053343Z_77fdb4ac，status=candidate_graph_validated，
pytorch_equivalence_validated=true：

| 指标 | 结果 |
| --- | --- |
| 对照元素数 | 每后端1705088，包含身份空间特征、全局条件及前8帧RGB/状态 |
| 手机FP32 CPU相对PyTorch最大绝对误差 | 1.19e-7 |
| 手机GPU相对PyTorch最大绝对误差 | 2.62e-4 |
| 手机GPU相对PyTorch最大归一化误差 | 0.7524，小于通过门槛1 |
| 连续输出 | 160帧RGB/状态全部有限，状态重置重放误差0 |
| 单帧中位 / P95 | 1.81 / 1.95ms |
| GPU测量阶段进程峰值RSS | 132.0MiB，包含导出参考张量的内存开销 |

GPU最大归一化误差出现在身份空间特征，本次在预先设定的阈值内；未因结果修改阈值。
仍需针对更多输入及真实训练权重检验精度。上述延迟只适用于约24.6万参数的候选图，
不是Wan 14B或最终蒸馏模型的性能，也不证明人物质量或应用首播目标达成。

## 验证范围与下一阶段

前8帧通过不等于长序列质量通过；随机输入、随机/一次合成梯度更新的权重无法评估口型、
表情、身份保持或半身手势。单帧性能仍只适用于该小型候选图。
模型文件读取计入使用bundle的新进程首帧计时，但进程启动、动态库加载、音频预处理、
图片解码、播放和未清空的驱动缓存都需要另行测量，不能将其直接作为用户首播耗时。

下一阶段接入真实参考图、明确时间对齐的音频特征与教师帧，先做少量样本拟合。
仅在跨端一致和小样本质量可接受后，才扩大教师数据量及训练资源。
