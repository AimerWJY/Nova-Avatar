# v2新音频预览与真实图导出

本轮没有增加训练步骤或改动v2权重，继续使用mouth_refinement_v2的864376个参数。
新增独立音频推理、分离编码器/逐帧图导出、TorchScript回放校验及ncnn实机加载工具。
这不是14B模型量化，也不是手机应用已经完成。

## 未参与训练的normal素材

```bash
cache/student-venv/bin/python scripts/student_infer.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --training-manifest data/emotion-overfit-001/manifest.json \
  --audio-source '/Users/wangjingyu/Downloads/video (1).mp4' \
  --reference '/Users/wangjingyu/Downloads/reference (1).png' \
  --output outputs/v2-normal-audio-001
```

该视频对应此前normal.wav测试，训练使用的则是用户确认的emotion版本。
脚本只读取它的音轨，不使用该视频的人物帧作监督或输入。
解码后的音频前缀与训练音轨不同，原图像素与训练参考图一致，生成78帧、4.875秒。
已查看6个时刻，嘴部有变化；未提供合格的匹配教师口型或独立同步评估，
所以generalization_verified、lip_sync_verified保持false。不同文件名或嘴部在动不能证明同步正确。

推理入口支持带音轨的视频、WAV或MP3，初版最多截取前10秒。
保留实际音频PTS，沿用causal_logmel40_v1；拒绝与训练音轨重复的前缀及无时间覆盖的输入。
适用于当前参考图和固定嘴部位置，不证明新人物泛化。

输出：

```text
outputs/v2-normal-audio-001/student.mp4
outputs/v2-normal-audio-001/audio_vs_silence.mp4  # 左正常条件，右静音条件，均配原音轨便于对照
outputs/v2-normal-audio-001/frames.jpg
outputs/v2-normal-audio-001/report.json
```

静音对照输入为契约规定的全-1特征，不会因为文件未播放声音就误称静音模型条件。

## 导出真实v2图

本地验证使用PyTorch 2.9.1、pnnx 20261020和ncnn 20261020。
转换入口依据[pnnx官方使用说明](https://github.com/pnnx/pnnx)，固定版本，避免替换平台PyTorch：

```bash
cache/student-venv/bin/python -m pip install --index-url https://pypi.org/simple \
  --no-deps pnnx==20261020
cache/student-venv/bin/python scripts/student_mobile_export.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --output outputs/v2-mobile-export-001 --convert-ncnn
```

输出目录必须不存在。student/mobile.py划分两个batch=1的固定形状图：

| 图 | 输入 | 输出 |
| --- | --- | --- |
| encoder | [1,3,256,256]原图 | [1,32,64,64]空间特征及[1,128]身份条件 |
| frame | 原图、缓存特征、身份条件、[1,8,40]音频、[1,128]状态 | [1,3,256,256]RGB及下一状态 |

保存并重新加载encoder.pt、frame.pt，然后逐帧与原PyTorch模型比较。
本次78帧、15476608个元素，最大绝对差0，TorchScript序列化校验通过。
还用不同音频及非零状态做trace检查，避免将某个状态常量化。

pnnx生成encoder.ncnn.param/bin和frame.ncnn.param/bin，图中没有遗留pnnx/aten等未转换类型。
manifest记录权重和图SHA256、版本及转换结果。
转换文件生成成功不表示运行时加载或手机数值校验通过；这两件事单独记录。
原student_export.py仍只支持v0专用bundle，不能拿它导出v2。

## 手机全序列数值验证

```bash
cache/student-venv/bin/python scripts/student_mobile_fixture.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --model-manifest outputs/v2-mobile-export-001/manifest.json \
  --output outputs/v2-mobile-fixture-001

python3 scripts/mobile_model.py \
  --model-manifest outputs/v2-mobile-export-001/manifest.json \
  --fixture-manifest outputs/v2-mobile-fixture-001/manifest.json \
  --serial c9fb5e39
```

全CPU对照再加--device cpu；仅构建不操作手机可加--build-only。
云端可用cloud.sh的student-mobile-export、student-mobile-fixture和student-infer入口，
手机测试仍在连接ADB的Mac执行。不同PyTorch或转换版本需重新检验，不直接沿用本机结论。

fixture含实际训练片段的预计算音频、参考图、身份缓存及完整78帧PyTorch RGB/状态。
它约59.8MiB，只是测试对照文件，不是运行模型所需的数据包。
主机验证模型、fixture和checkpoint来源及SHA256；原生端限制大小128MiB，
检查版本、帧数、每个输入有限值、尾部字节，再加载真正的ncnn Net图。

本次采用FP32、两CPU线程、预热3次，每一帧RGB和状态均与PyTorch比较，
阈值保持abs_error <= 1e-5 + 1e-4*abs(reference)。同时检查显式状态重置后首帧一致。

## 实机结果：加载可运行，严格验收未通过

小米17、Adreno 840的原始转换图报告：

- Vulkan请求：model_20261105T073741Z_10d8d400。
- 全CPU对照：model_20261105T074009Z_be0e4a89。

两种方式都成功加载并计算78帧，约1567万个元素检查中输出全部有限，状态重置一致。
Vulkan请求下最大绝对误差1.26958e-5，最大归一化误差1.14451；
CPU对照最大归一化误差1.14414，主要出现在第5帧RGB。
两者均超过门槛1，状态如实标记numerical_validation_failed，未放宽阈值。

两个Attention MatMul和GridSample没有声明Vulkan支持，会通过ncnn进行CPU回退。
这不是纯GPU执行，也没有使用NPU。层能力标记不是实际GPU指令跟踪，
即使其他层声明支持，也不会将gpu_only_execution_verified自动改为true。

分离转换图中的乘加融合、改用参考顺序采样的诊断均未消除超标，尚未定位根因。
这些实验性修改没有纳入默认路径，原始和对照报告保留供复查。

Vulkan混合路径本次预热后中位18.27ms、P95 20.01ms，首次原生输出约13.8秒；
CPU路径中位45.77ms、P95 50.62ms，首次原生输出约367ms。
另一次有缓存的混合对照中位约9.79ms，但不是清除缓存或温度受控的加速比试验。
上述速度只作诊断记录，不能用来掩盖数值门槛失败。
进程RSS包含约60MiB参考fixture，不等于应用或模型单独内存。

手机目前还没有音频前处理、图片文件解码、播放器、APK或10分钟持续温度测试。
因此首播2秒、完整端到端实时性、独立音频口型同步和视频质量均未验收通过。
下一阶段优先对照转换前后中间张量定位数值差异，并处理CPU回退及独立音频质量验证。

后续定位及已通过的新导出变体见[student-v2-runtime-fix.md](student-v2-runtime-fix.md)。
新变体明确固定原有零填充区并改写Attention，不能把本节原始raw_v2的失败记录直接改成通过。
