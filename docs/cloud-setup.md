# 云服务器统一执行

本阶段将已讨论的环境和模型准备命令纳入 Git 管理。代码可以在本地提交并推送，服务器只需拉取和调用统一入口。模型、音视频、缓存及报告不进入 Git。

## 目录约定

| 路径 | 内容 |
| --- | --- |
| /cloud/nova-avatar/repo | 建议的本仓库克隆位置 |
| /cloud/nova-avatar/.venv | Python 环境；复用云平台 PyTorch |
| /cloud/nova-avatar/models | 完整模型视图及补下载的权重 |
| /cloud/nova-avatar/inputs | person.png、speech.wav 等测试输入 |
| /cloud/nova-avatar/outputs | 后续推理结果 |
| /cloud/nova-avatar/reports | 环境检查、依赖快照 |
| /cloud/nova-avatar/cache | Hugging Face、pip 缓存 |
| /model/HuggingFace/Wan-AI/Wan2.2-S2V-14B | 平台共享模型，只读使用 |
| /cloud/nova-avatar/Wan2.2 | 此前下载的官方代码，与本仓库分开 |

若本仓库已经克隆到别处，直接在那个目录运行即可，不必重新克隆。Git 中仅需提交源码、requirements、测试和文档。首次克隆请使用你自己的仓库地址：

```bash
git clone <你的仓库地址> /cloud/nova-avatar/repo
cd /cloud/nova-avatar/repo
```

## 初次执行

系统前置条件：Linux、Python 3.10+、可运行 CUDA 的平台 PyTorch、git、ffmpeg、ffprobe。服务器已有这些工具时，不重复安装系统包。缺失时在 Ubuntu/Debian 上由管理员执行 `apt-get update` 和 `apt-get install -y --no-install-recommends ffmpeg git`。入口不会自动执行 apt 或 sudo。

在本仓库目录逐条执行；遇到错误先处理错误，不继续后续步骤：

```bash
bash scripts/cloud.sh init
bash scripts/cloud.sh install-deps
bash scripts/cloud.sh check
bash scripts/cloud.sh model-check
```

当前已知共享模型缺少第一个 DiT 分片，`model-check` 此时以状态码 1 退出是预期结果。确认报告是缺分片后再执行：

```bash
bash scripts/cloud.sh model-repair
bash scripts/cloud.sh model-check
```

`model-repair` 需要访问 Hugging Face。已知缺失文件约 9.97GB，下载消耗数据盘空间及流量。中断后可以重跑，下载恢复行为由 huggingface_hub 管理。它将：

1. 检查共享索引、现有分片张量头及必需组件。
2. 固定本次远端 revision，比较远端与本地索引；不同则停止。
3. 只下载缺失的 DiT 分片到 models/Wan2.2-S2V-14B。
4. 将共享目录的其他现有条目链接到该目录，不复制大文件，不修改共享目录。
5. 再次检查分片，记录下载来源和 revision。

完整性检查包含张量名、头信息和payload边界，不是完整内容校验和，也不证明共享文件来自相同revision。遇到损坏或本地冲突时停止，不删除或覆盖已有权重。链接视图依赖 /model 继续挂载，不能脱离共享存储直接搬走使用。

## 后续更新

```bash
cd /cloud/nova-avatar/repo
git pull --ff-only
bash scripts/cloud.sh check
```

requirements 更新后再运行 install-deps；不必每次运行 model-repair。命令失败时将终端输出和 reports 中的环境报告带回分析。不要在报告中添加登录凭证。

## decord 0.6.0 平台标签告警

上游 Linux wheel 的文件名为 py3-none-manylinux2010_x86_64，但内部 WHEEL 标签误写为 cp36-cp36m-manylinux2010_x86_64，参见 https://github.com/dmlc/decord/issues/356 。在 Python 3.12 中，可能出现能导入、但 pip check 报 `decord 0.6.0 is not supported on this platform` 的情况。重复安装同一个包不能修复上游标签。

check/install-deps 使用 scripts/cloud_dependencies.py 保留原始 pip check 输出和返回码。仅当报错恰好只有这一条、版本为0.6.0、内部标签完全匹配、系统为Linux x86_64时，额外用FFmpeg生成5帧小视频，在限时子进程中验证decord的CPU解码、首中尾帧形状和像素内容。测试通过将已知元数据问题记为warning，不修改WHEEL或关闭检查；测试失败或出现其他依赖问题仍返回失败。

这项小视频测试不保证所有编码格式和长视频都兼容，后续还需要使用真实输入测试。直接执行 `python -m pip check` 仍会报告原始标签错误，这是预期行为；以项目check报告中的errors、warnings及decord_decode_test共同判断。服务器拉取更新后运行check即可，无需因本次更新重装整套环境。

## 命令及配置

| 命令 | 行为 |
| --- | --- |
| init | 创建数据目录；已有环境复用，不重新创建；新环境使用 --system-site-packages |
| install-deps | 安装项目基础依赖，生成GPU包约束和pip freeze快照 |
| check | 导入基础依赖、查询CUDA/工具/剩余磁盘、运行pip check，写JSON报告 |
| model-check | 优先检查本地模型视图，否则检查共享目录；不下载模型 |
| model-repair | 下载缺分片并建立本地链接视图 |

install-deps 将现有 torch、torchvision、torchaudio（若存在）版本写入 pip constraints，依赖冲突时报错而不主动替换GPU包。venv隔离新增包，但可读取平台包；不会安装 torchaudio、FlashAttention、ComfyUI 或低显存推理框架。这只是基础环境，不是可直接运行S2V的完整依赖锁文件。非核心依赖为版本范围，成功安装的准确版本以服务器生成的 freeze 文件为准。

可以用环境变量覆盖路径，例如：

```bash
NOVA_ROOT=/another/data/nova-avatar bash scripts/cloud.sh init
NOVA_ROOT=/another/data/nova-avatar bash scripts/cloud.sh check
```

所有命令应使用相同 NOVA_ROOT。可选 NOVA_VENV、NOVA_SHARED_MODEL、HF_HOME、PIP_CACHE_DIR。已有HF_HOME/PIP_CACHE_DIR会被尊重；否则缓存默认位于数据盘。

## 当前限制

此入口只完成准备阶段，不启动官方BF16推理，不转换Kijai FP8格式，也不启动蒸馏。FP8权重必须匹配加载器；24GB显存和首播2秒指标仍需实测。不要直接将官方14B模型整体移到4090上。

本地逻辑测试：

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/cloud.sh
```
