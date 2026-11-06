# 教师视频小样本拟合

此流程把已有教师输出作为监督信号，验证候选网络能否拟合一段音画配对视频。
它不是Wan权重剪枝、扩散步蒸馏或已经完成的通用数字人训练。
当前只支持单个已审核的16FPS片段，用于发现训练、时序和容量问题，不直接扩大数据量。

## 准备数据

云端复用既有PyTorch环境，并安装PyAV媒体依赖；该命令约束平台PyTorch版本，避免替换：

```bash
bash scripts/cloud.sh install-student-deps
bash scripts/cloud.sh student-prepare \
  --video /实际路径/video.mp4 --reference /实际路径/reference.png \
  --output /cloud/nova-avatar/data/emotion-overfit-001 \
  --person-id person_01 --approved --review-note '记录已核对的素材版本与审核结论'
```

本地独立环境可运行对应Python入口：

```bash
cache/student-venv/bin/python -m pip install --index-url https://pypi.org/simple av==16.0.1
cache/student-venv/bin/python scripts/student_prepare.py \
  --video '/Users/wangjingyu/Downloads/video (2).mp4' \
  --reference '/Users/wangjingyu/Downloads/reference (1).png' \
  --output data/emotion-overfit-001 --person-id person_01 \
  --approved --review-note '用户确认emotion版本及其配对参考图'
```

实际本地配对已经获得用户确认。不要把`--approved`用于尚未核对的素材。
该新流程独立于prepare_dataset.py的原始素材入库，不会修改原视频、图片或原清单。
输出目录必须不存在；原件和派生数据均不提交Git。

准备阶段检查并保留以下信息：

- 每帧真实PTS，要求连续且均匀16FPS，帧数8到160，不自动变速或猜测时间戳。
- 提取视频内嵌的对应音轨，依PTS转为16kHz单声道；不另外找同名WAV替换。
- 拒绝前端或中间音频缺口；小于2%的尾部音频缺失会使未完整覆盖的尾帧被移除。
- 图像和视频都保持比例缩放至256方形并居中补黑边，不进行额外人脸裁剪。
- 来源文件SHA256、缓存SHA256、用户审核说明、裁剪记录和音频特征契约。

本次素材原有80帧，解码音频覆盖78848样本，保留78个完整帧区间，即4.875秒。
原始音频覆盖率0.9856及移除2个尾帧都写入manifest，保留段没有缺失音频补零训练。
本段对应的生成report未提供，generation_metadata_status如实记录为not_supplied_for_this_clip；
不会把normal.wav那次report冒充成emotion版本元数据。此数据仍限于诊断用途。

```text
data/emotion-overfit-001/
  manifest.json        # 审核、来源、特征契约、时间对齐、缓存校验值
  sample.npz           # reference、frames、audio_features、audio、times；禁用pickle加载
  aligned_audio.wav    # 与保留帧区间对应的PCM16音轨
  reference.png        # 训练实际使用的256方形参考图
```

## 固定音频特征契约

student/audio.py定义causal_logmel40_v1，使用NumPy实现，不依赖torchaudio或librosa版本默认值。
16kHz单声道，400样本对称Hanning窗，512点FFT；40个HTK Mel三角滤波器覆盖50到7600Hz，
每个滤波器权重归一化为和1，功率除以512。固定归一化为：

```text
clip((log10(max(mel_power,1e-8))+4)/4,-2,2)
```

每个视频帧使用截至该帧PTS之前的25ms音频，起始不足补零，不读取未来样本。
每帧携带当前及之前共8帧特征，历史不足重复首个静音特征；不计算整段均值或方差。
这是初始实验前端，稀疏时间采样与上下文长度是否足够支持自然口型仍需验证。
手机目前只接受预计算特征，尚无Android麦克风/WAV前处理；不能把本轮训练当成实时音频链路已经上线。

## 训练与预览

```bash
bash scripts/cloud.sh student-train \
  --dataset /cloud/nova-avatar/data/emotion-overfit-001/manifest.json \
  --output /cloud/nova-avatar/outputs/emotion-overfit-001 \
  --steps 200 --sequence-length 8 --device auto
```

本地使用`cache/student-venv/bin/python scripts/student_train.py`，其余参数相同。
auto优先CUDA，否则CPU，不自动选择MPS。训练目标为整帧L1＋3倍手工口部区域L1＋
0.5倍帧间差分L1，使用Adam、全局梯度裁剪1、8帧随机连续片段和至多8帧无梯度状态预热。
这不包含专用唇音同步网络、语音识别、关键点监督或人物感知损失。

默认手工口部区域为256图中的[90,92,166,133]，本次已查看mouth_roi_review.png确认覆盖嘴部。
这只是固定矩形，不能代替人脸追踪；换人物或构图时通过--mouth-box X0 Y0 X1 Y1调整归一化坐标。
拟合后从零状态按整段顺序生成，避免只看随机训练片段的局部损失。

输出包括checkpoint.pt、report.json、带原音轨的student.mp4、
左教师/右学生的comparison.mp4、三时刻comparison.png及口部区域检查图。
--init-checkpoint可载入相同结构的参数进行后续拟合，Adam状态不会恢复，属于权重热启动。

评估只使用同一训练片段，同时记录：

- 完整序列的整帧/口部误差及帧间变化；训练日志中的片段损失不能直接与整段指标混比。
- 将音频循环错位半段后的误差及输出变化，以及全静音后的输出变化。
- 同一训练视频逐像素均值图的静态参考指标。这是使用训练目标计算的oracle基线，不是可泛化推理方法。

错位音频改变了输出，只证明模型对输入敏感，不能证明响应对应正确音素。
损失下降或静态人像变清晰，也不能证明学会了嘴型与手势。
报告始终明确evaluation_split=same_training_clip_reconstruction，
lip_sync_verified、generalization_verified、video_quality_validated均为false。

## 导出已拟合的参数

```bash
bash scripts/cloud.sh student-export \
  --checkpoint /cloud/nova-avatar/outputs/emotion-overfit-001/checkpoint.pt \
  --output /cloud/nova-avatar/exports/emotion-overfit-001
```

导出时保留teacher_trained来源字段。手机的trained字段现在反映该来源标签，
而不是画质已通过；video_quality_validated始终单独记录。
标准导出仍使用合成输入检查跨端数值，并不替代真实视频质量审核。
手机验证与误差阈值见[student-export.md](student-export.md)。

## 本次实际结果（2026-11-05）

本地CPU先进行200步、lr=0.001，输出outputs/emotion-overfit-001；
再从其checkpoint热启动1000步、lr=0.0003，输出outputs/emotion-overfit-002。
两轮均为8帧训练片段，第二轮重建Adam状态，并非恢复优化器的断点续训。
第一轮初始参数来自前一阶段的合成梯度验证checkpoint，人物数据监督共1200步。

| 同一训练视频的整段指标 | 训练前 | 200步后 | 再1000步后 | 静态训练均值图 |
| --- | --- | --- | --- | --- |
| 整帧MAE | 0.33301 | 0.07710 | 0.04223 | 0.02772 |
| 手工口部区域MAE | 0.25646 | 0.08380 | 0.06358 | 0.07285 |
| 帧间差分MAE | 0.01064 | 0.01094 | 0.01294 | 0.01008 |

第二轮用正确音频的口部误差0.06358，音频错位半段后0.07902。
这说明在本段训练数据上存在与正确音频对应的学习迹象，但不足以证明音素级同步或泛化。
整帧误差仍高于静态均值图，差分误差也没有改善；人工查看预览发现脸部模糊、手部动作丢失，
**当前候选的画质及动作目标未通过**。不能只凭重建误差下降宣称蒸馏成功。

本机已生成并核对：

```text
outputs/emotion-overfit-002/comparison.mp4       # 左教师，右学生，配保留段原音轨
outputs/emotion-overfit-002/student.mp4
outputs/emotion-overfit-002/comparison.png
outputs/emotion-overfit-002/mouth_roi_review.png
outputs/emotion-overfit-002/checkpoint.pt
outputs/emotion-overfit-002/report.json
outputs/emotion-overfit-002-export/manifest.json
```

已对第二轮参数做手机合成输入校验，没有把随机初始化时的精度结论套用到训练后权重：

| 精度 | 跨端最大归一化误差 | 结果 |
| --- | --- | --- |
| FP16存储、关闭FP16算术 | 2.35396 | 超过门槛1，失败，主要出现在时序状态 |
| 全FP32 GPU | 0.01153 | 通过，未放宽门槛 |

失败报告benchmark_20261105T062528Z_c914b8ff，成功对照报告benchmark_20261105T063452Z_5c4de1de，
均保留在本地reports/mobile。FP32 GPU相对PyTorch最大绝对误差1.43e-6，
比较1705088个身份特征/RGB/状态元素；连续160帧均有限、重置状态一致。
FP32单帧中位2.61ms，P95 2.72ms；这些是该小型计算图与合成输入的性能，
不含手机音频前端和播放，也不能外推更大模型的速度或最终首播时间。

复现训练后权重的数值验证：

```bash
python3 scripts/mobile_benchmark.py --backend candidate \
  --bundle-manifest outputs/emotion-overfit-002-export/manifest.json \
  --candidate-precision fp32 --serial c9fb5e39
```

下一阶段应先改进保留参考图细节的结构、局部口型与时序监督，并检验音频上下文，
重复本段诊断后再决定是否增加模型容量、教师样本数量或训练资源。

已完成的原图细节保留v1实验见[student-reference-v1.md](student-reference-v1.md)。
该结构与本节v0不同，不复用v0的手机性能或导出结论。
