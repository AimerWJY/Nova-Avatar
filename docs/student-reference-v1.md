# 保留参考图细节的v1候选

v0在单片段拟合后仍然模糊，当前实验改用reference_warp_residual_v1，
实现位于student/reference_model.py。它有513669个参数，属于新的学生结构，
不是Wan权重剪枝结果，也不是已达到手机质量目标的模型。

## 结构变化

- 缓存原始参考图及其64×64空间特征。初始化时位移、补充像素输出均为零，逐像素复制参考图。
- 由音频历史、身份条件及上一帧状态产生下一状态，用tanh将状态限制在[-1,1]。
- 将128维状态投影为32×8×8空间条件，放大后与图像特征拼接；替代v0的全图相同通道偏移。
- 解码两通道位移和三通道补充像素。位移经tanh限制在±24像素，双线性采样原图，
  再加有界补充像素并限制到[0,1]。

训练使用与之前相同的整帧L1、3倍手工口部L1、0.5倍帧间差分L1。
本轮没有增加感知模型、预训练语音模型、关键点或专用唇音同步损失。
因此细节保留主要来自原图通路，不能把脸部变清晰解释成已经学会正确嘴型。
状态有界是结构保障，但尚未证明解决了手机FP16误差问题。

## 复现实验

继续使用已审核的data/emotion-overfit-001：78帧、16FPS、4.875秒，
同一参考图、同一音轨和causal_logmel40_v1音频契约。
本次没有更换或扩大数据集。

```bash
cache/student-venv/bin/python scripts/student_train.py \
  --architecture reference_warp_residual_v1 \
  --dataset data/emotion-overfit-001/manifest.json \
  --output outputs/emotion-reference-v1-001 \
  --steps 600 --sequence-length 8 --lr 0.0003 --device cpu

cache/student-venv/bin/python scripts/student_compare.py \
  --dataset data/emotion-overfit-001/manifest.json \
  --v0-checkpoint outputs/emotion-overfit-002/checkpoint.pt \
  --v1-checkpoint outputs/emotion-reference-v1-001/checkpoint.pt \
  --output outputs/emotion-v0-v1-comparison-001
```

云端复用既有环境，分别使用`bash scripts/cloud.sh student-train`与`student-compare`，
其余参数相同并将路径改为实际数据盘位置。输出目录必须不存在。

本次v1从头初始化后训练600步；对照v0来自之前200+1000步。
两者参数量、初始化路径和训练预算不同，这不是严格同预算或单因素消融实验。
它仅回答当前两份候选在同一训练片段上的实际表现。

## 2026-11-05结果

比较脚本重新加载两份checkpoint，在相同原始缓存及时间轴上重新推理，
统一计算指标；不从压缩后的视频截图反推误差。
要求两份权重的教师视频SHA256及数据清单SHA256都与当前数据匹配。

| 同一训练片段指标，越低越好 | 复制原图 | 训练均值图oracle | v0 | v1 |
| --- | --- | --- | --- | --- |
| 整帧MAE | 0.02987 | 0.02772 | 0.04223 | 0.01773 |
| 手工口部MAE | 0.08502 | 0.07285 | 0.06358 | 0.04282 |
| 口部边缘差分MAE | 0.04600 | 0.03455 | 0.03395 | 0.02779 |
| 口部帧间差分MAE | 0.03282 | 0.03282 | 0.03222 | 0.03041 |
| 整帧帧间差分MAE | 0.01008 | 0.01008 | 0.01294 | 0.01064 |

v1口部MAE较v0下降约33%，正确音频下为0.04282，音频循环错位半段后为0.08168。
这支持该训练片段上的音频条件拟合有所改善，不证明跨音频泛化或音素级同步。
整帧帧间差分误差仍高于静态基线，不能宣称时序目标达成。

已人工查看三时刻对比：v1明显保留了面部细节，但嘴部动作仍偏弱，
参考图中未出现的手部在预测里仍是模糊影子。
当前lip_sync_verified、video_quality_validated和independent_validation均为false。

产物：

```text
outputs/emotion-reference-v1-001/checkpoint.pt
outputs/emotion-reference-v1-001/report.json
outputs/emotion-v0-v1-comparison-001/teacher_v0_v1.mp4  # 左教师，中v0，右v1；原音轨
outputs/emotion-v0-v1-comparison-001/teacher_v0_v1.jpg  # 三时刻对照
outputs/emotion-v0-v1-comparison-001/report.json
```

三栏视频已经完整解码检查：768×256、78帧，音轨4.875秒。

## 当前边界与后续

v1的图像采样、缓存与参数结构尚未接入Android导出和数值验证。
现有student_export.py明确只接受v0，拒绝将v1权重写入v0格式。
phone_v1_validated=false；v0的2.61ms结果不适用于v1。
新模型尚不能宣称手机端实时运行、首播2秒或FP16可用。

下一阶段需要加强口部动态约束与音频上下文，并解决未见手部的生成质量；
同时验证新结构的手机算子、参数映射和速度。仍应先做小规模验证，再扩大训练资源。
