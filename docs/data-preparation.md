# 第一阶段：数据入库与审核

先准备 10～20 条有训练授权、音画同步的真实半身讲话视频，确认流程后扩至 100～300 条。起步片段建议 5～15 秒；另外保留一些 30～60 秒连续视频，用于后续流式切片。片中尽量只出现一个人，脸、肩膀、手部清晰，背景简单。真实源文件不要变速、插帧或重新配音。

## 环境

需要 Python 3.10+、FFmpeg 和 ffprobe，并可从 PATH 调用。无需 GPU 或 Python 第三方依赖。云服务器按其包管理器安装 FFmpeg；先用 `ffmpeg -version`、`ffprobe -version` 确认。

本仓库忽略 data/，原视频和派生数据不要提交到 Git。云服务器推荐将数据放在持久数据盘，例如 /mnt/data/nova-avatar；请先确认实际挂载位置。

## 入库

把一位人物的一批原视频放在单独目录，每个视频必须自带对应音轨。多个人物分别执行命令，使用不同 person-id；同一人物始终使用相同 ID。此工具不处理多人物视频或单独音频配对。

```bash
python3 scripts/prepare_dataset.py --help

# 仅读取和检查原视频，不生成训练资产。
python3 scripts/prepare_dataset.py scan \
  --input /mnt/data/incoming/person_001 \
  --output /mnt/data/nova-avatar \
  --person-id person_001 --source-type real \
  --rights-note '填写真实来源及训练授权依据'

# 确认清单后：复制原文件，提取音频、参考图，生成待审核清单。
python3 scripts/prepare_dataset.py prepare \
  --input /mnt/data/incoming/person_001 \
  --output /mnt/data/nova-avatar \
  --person-id person_001 --source-type real \
  --rights-note '填写真实来源及训练授权依据'
```

工具不改变视频分辨率、帧率或时长，原视频复制到 originals/。音频转为单声道 16kHz PCM WAV（供起步 Wav2Vec2 实验；源文件仍完整保留），中间帧作为临时参考图。参考图可人工换成质量更好的一帧。没有音轨的样本拒绝入库。相同内容、人物及来源的重复文件只保留一条。

每次执行会在 manifests/ 创建独立且不会覆盖旧文件的 scan/prepare JSONL 和 JSON 报告。prepare 清单只包含本次成功处理的样本；已有产物不会覆盖，重复入库会验证后复用。报告包含失败原因、原始帧率、VFR 时间戳检查和时长告警。未扫描到视频时命令报错；部分样本失败时仍保存报告并以非零状态退出。

```text
nova-avatar/
  originals/<id>.<ext>
  audios/<id>.wav
  references/<id>.png
  manifests/prepare_<unique>.jsonl
  manifests/report_<unique>.json
```

## 人工审核和分类

清单是项目中间格式，不是 Wan 官方训练格式。每行是一条 JSON 记录。所有样本初始状态为 `pending`、split 为 `unassigned`，不得直接当作已验收训练集。

逐条观看原视频及派生音频，填写以下字段：

- `review.status`：`approved` 或 `rejected`；`review.notes` 记录问题。
- `review.av_sync`、`review.face_visible`、`review.hands_quality`：`pass`、`fail` 或 `not_applicable`。手部不出现不必拒绝，可标记 hands_out_of_frame。
- `caption`：真实描述可见表情、动作，不臆测情绪或对白含义。
- `tags`：多标签，不复制文件到多个类别目录。

起步标签词表：

| 维度 | 候选值 |
| --- | --- |
| speech | normal, fast, slow, pause, silence |
| expression | neutral, smile, serious, puzzled, surprised |
| head | still, nod, shake, turn_left, turn_right, tilt |
| upper_body | still, shrug, lean_forward, turn |
| gesture | none, single_hand, both_hands, point, wave, hands_out_of_frame |

左右按人物自身方向记录。可以多选；缺少标签表示尚未标注，不代表 neutral/none。合成样本另需补全 `generation` 中的教师模型版本、提示词、seed、步数、采样器、CFG；生成后也必须审核同步、手指及身份稳定性。

审核通过后再按人物和原始录像分组划分 train/val/test。若以后将长视频切片，所有子片必须继承 `source_id`，并记录切片起止时间；同一原录像和相关合成变体不能跨集合。任意人物任务应按人物隔离；固定人物任务应至少按原录像/录制场次隔离。测试集保留独立真实样本及长片段。

## 自动检查的边界

音视频时长接近不能证明唇音同步；ffprobe 不能判断脸部、手势或画面是否自然。VFR 检查通过逐帧时间戳判断，检查失败时会明确告警。VFR 样本保留原件，后续确定模型训练帧率后再统一处理。

本阶段不统一裁剪至 480P/25FPS、不生成关键点、不下载模型、不导出教师轨迹。先确认样本质量与最终训练加载器，再生成模型相关缓存，避免重复处理。
