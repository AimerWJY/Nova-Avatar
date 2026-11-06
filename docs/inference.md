# 第一条S2V教师视频

前置条件：cloud.sh check的errors为空、model-check的四个分片均通过、人物图和音频已在inputs目录。GPU必须可用。此阶段是离线质量/资源基线，不是首播2秒或流式Student实现。

## 安装

在服务器本仓库目录运行：

```bash
bash scripts/cloud.sh install-diffsynth
```

从GitHub官方codeload端点安装configs/diffsynth-revision.txt固定提交的ZIP源码包，不经过git clone/fetch，避免Git的GnuTLS握手中断。先安装requirements/diffsynth.txt，复用并约束现有PyTorch及核心包，再用--no-deps安装该提交，防止上游依赖解析替换平台GPU包。新增ModelScope、pandas、PEFT等依赖；不安装量化、训练、audio/all extras，也不要求额外FlashAttention。PEFT为兼容当前transformers固定为0.15.2。安装结束检查S2V接口能否导入和依赖状态，依赖冲突会报错。

源码下载设置重试5次和连接读取超时120秒；仍需要能访问codeload.github.com，不保证能绕过网络故障。无需关闭TLS证书校验或升级pip。若旧安装命令在git fetch阶段失败，拉取更新后重新运行install-diffsynth即可，已满足的依赖会复用。推理接受原有固定提交的Git安装和新的固定提交官方ZIP安装；报告记录来源及pip保存的archive哈希，不接受main分支ZIP或未知来源。

此命令需要网络来下载代码和Python包；推理只读本地模型，不下载任何示例、模型或数据集。首次安装后保留reports里的环境快照。若实际环境有额外不兼容，保存错误后针对性处理，不随意升级PyTorch。

## 执行

```bash
# 验证模型文件、图片解码和音频时长；不加载大模型，不执行GPU推理。
bash scripts/cloud.sh infer --dry-run

# 首次正式离线推理。
bash scripts/cloud.sh infer
```

默认使用NOVA_MODEL_DIR（默认/model/HuggingFace/Wan-AI/Wan2.2-S2V-14B），以及NOVA_ROOT/inputs/person_01.png、NOVA_ROOT/inputs/normal.wav。音频读取前会转为单声道16kHz PCM WAV，截取开头最多5秒，时长向下对齐到0.25秒。输出帧数是4的倍数，pipeline输入帧数为4n+1，按上游示例去掉首个参考帧。

图像按原始比例缩放后补边至480×640，不拉伸或裁掉手部。准备后的reference.png用于检查构图。若是JPG，显式传入实际路径，不要仅改扩展名：

```bash
bash scripts/cloud.sh infer \
  --image /cloud/nova-avatar/inputs/person_02.png \
  --audio /cloud/nova-avatar/inputs/pause.wav \
  --seconds 5 --seed 42
```

默认40步、CFG 4.5、shift 3、BF16、VAE分块解码。S2V这条实现按原生16FPS推理，不能只改导出帧率来当作25FPS模型。没有姿态控制视频，手势由模型生成，尚不提供精确手势控制。

## 显存与速度

默认--offload disk，权重按上游低显存示例在磁盘、CPU及GPU之间调度；vram-limit默认20GiB，并根据当前空闲显存扣除3GiB余量后取较小值。该限制控制权重调度，不是严格的进程显存上限；激活和CUDA缓存仍会额外占用。不能保证24GB上任何分辨率都不OOM。

共享盘若是网络文件系统，反复读权重可能很慢；.pth格式的T5/VAE仍需要CPU内存。CPU模式会让更多权重常驻内存，64GB机器可能有内存压力，不建议第一轮切换。默认先验证能否生成，不承诺速度。

若OOM，保留报告后用更小设置再测试（不要同时开多个推理任务）：

```bash
bash scripts/cloud.sh infer --width 384 --height 512 --seconds 2 --vram-limit 18
```

宽高须为16的倍数，单片时长限制1～5秒。--steps可调整，但未蒸馏模型直接减步数可能损伤质量；第一轮先保留40步。长音频目前仅取首段，尚未实现滑窗或流式输出。

## 输出与测量

每次创建独立的outputs/s2v_<UTC时间>_<随机ID>/目录，包含：

- reference.png：保持比例处理后的人物图。
- audio.wav：实际使用的短音频。
- video_silent.mp4：无音轨编码中间文件。
- video.mp4：最终带音频的视频；音画合成失败则本次运行失败。
- report.json：参数、固定DiffSynth提交、GPU信息、加载/生成/编码耗时和峰值显存；失败也保存报告。

终端输出同步保存到reports/infer-*.log。命令非零退出时不要只凭存在MP4判断成功；以报告status=success为准。

generation_fps包括本次条件编码、去噪和VAE解码，不包含模型加载及视频编码；total_after_preflight包括其余整条执行链，但不包含预检查。显存统计为PyTorch分配/保留量，不是nvidia-smi的整卡峰值。本入口不测首个可播放片段时间，不证明持续实时速度。

上游依据：

- https://github.com/modelscope/DiffSynth-Studio/blob/1e8ae1d59c1baaae6c4c85fef717a44682b0fbd5/examples/wanvideo/model_inference_low_vram/Wan2.2-S2V-14B.py
- https://github.com/modelscope/DiffSynth-Studio/blob/1e8ae1d59c1baaae6c4c85fef717a44682b0fbd5/diffsynth/core/loader/config.py

## Q2_K 短片测试

同一 Wan2.2-S2V-14B 的公开 GGUF 量化权重，使用固定版本的
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1052-f5df5f8)。
这项测试不需要 Python 虚拟环境。CUDA 编译需要 `git`、`cmake`、C++ 编译器及 CUDA Toolkit 的 `nvcc`。
只有 CUDA 版 PyTorch 不代表镜像包含 `nvcc`。推理需要 `ffmpeg`、`ffprobe` 和 `nvidia-smi`。

```bash
bash scripts/cloud.sh quant-s2v plan
bash scripts/cloud.sh quant-s2v build
bash scripts/cloud.sh quant-s2v assets
bash scripts/cloud.sh quant-s2v run
```

先确认能构建运行器，再下载约 16.4 GB 权重。默认放在
`/cloud/nova-avatar/models/wan-s2v-q2`。下载中断可重新执行 `assets`，
已有或手工上传的同名文件会按大小和 SHA256 校验。可用 `--assets-dir` 指向已下载目录。
如果已有兼容的 CUDA 版 `sd-cli`，可用 `run --binary /path/to/sd-cli`。

默认以 `person_01.png` 和 `emotion.wav` 生成 480×640、约 2 秒的短片。
按上游 S2V 示例用 Euler、20 步、CFG 6、16 FPS，不叠加少步 LoRA。
`outputs/s2v_q2_*/` 包含 `video.mp4`、`report.json`、`sd-cli.log` 及显存采样。
显存为设备级采样，包含同一 GPU 上其他进程的占用。
这轮用于判断短片画质和运行可行性。与 DiffSynth 的原始教师视频对照时，
需考虑后端、采样器和参数差异。手机装载和 10 秒跨段生成另行验证。
