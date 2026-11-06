# v2数值定位、固定填充区和Attention GPU改写

本轮未重新训练或修改checkpoint中的学习参数，也未放宽验收阈值。
原始raw_v2导出仍保留失败结论；新变体明确采用fixed_letterbox_mask_v1输出策略。
这是固定构图的输出约束修复，不宣称原始模型突然获得逐位一致性。

## 中间张量诊断

新增手机参数--diagnostic-frame，在该帧关闭light mode并导出选定中间张量。
诊断运行不用于性能验收。大型传输单独允许120秒超时，其他ADB查询仍默认20秒。
可用--frames 8制作较小的参考包，保持首帧到第5帧的真实状态递推。

```bash
cache/student-venv/bin/python scripts/student_mobile_fixture.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --model-manifest outputs/v2-mobile-export-001/manifest.json \
  --output outputs/v2-mobile-fixture-diagnostic-001 --frames 8

python3 scripts/mobile_model.py \
  --model-manifest outputs/v2-mobile-export-001/manifest.json \
  --fixture-manifest outputs/v2-mobile-fixture-diagnostic-001/manifest.json \
  --device cpu --frames 8 --warmup 0 --diagnostic-frame 5 --serial c9fb5e39
```

本机诊断报告model_20261105T081758Z_97e23f15。
用对应报告生成Python对照（也可用cloud.sh student-mobile-diagnose）：

```bash
cache/student-venv/bin/python scripts/student_mobile_diagnose.py \
  --phone-report reports/mobile/model_20261105T081758Z_97e23f15/report.json \
  --model-manifest outputs/v2-mobile-export-001/manifest.json \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --output outputs/v2-diagnosis-001/report.json
```

诊断同时比较完整原始递推和“将手机端实际输入交给PyTorch”的结果，避免将上游输入差异误认成当前层错误。
第5帧主要发现：

| 阶段 | 最大绝对差 |
| --- | --- |
| 输入图片/音频 | 0 |
| 输入空间特征 | 5.22e-8 |
| 输入时序状态 | 4.17e-7 |
| 位移输出卷积 | 7.15e-7 |
| 归一化采样网格 | 1.19e-7 |
| 形变采样后的图像 | 1.17e-5 |
| 两端使用完全相同网格重新采样 | 1.79e-7 |

因此观测到的是上游数值微差经形变采样放大，不能归咎于GridSample本身明显算错。
最严重归一化超标点为RGB第0通道、y=163、x=224，位于右侧黑色填充区。

## 固定原有黑色填充区

本参考原图480×640，256方形输入的真实内容区为[32,0,224,256]。
原模型形变会把内容颜色带入本应固定黑色的填充条，原始输出在填充区最高达到约0.4103。
数据准备时参考图和所有教师帧的这些区域都严格为零，因此新导出策略将它们固定为零。

--preserve-letterbox是显式可选策略：验证原参考文件SHA256，用原图几何推导有效区域，
并验证参考图和全部训练帧的区域外像素为零；否则拒绝导出，不能用它隐藏真实内容误差。
该策略只修改填充条，在PyTorch参考实现中有效区域的每个输出像素与原模型完全一致。
手机端仍按原有浮点误差阈值校验，不宣称不同硬件之间逐位一致。
新TorchScript和手机参考fixture均使用该输出契约；manifest记录策略、坐标和固定来源。
它不是偷偷将原始失败模型的参考答案替换成手机输出。

固定填充区版本报告model_20261105T082336Z_82e02d57，完整78帧按原阈值通过，
最大归一化误差0.608337（门槛1），但仍有两个MatMul及GridSample的CPU回退。
不同原图或预处理几何需要重新验证，不能把该固定策略套用到任意用户图片。

## Attention逐头Gemm

原导出两个MatMul的输入分别为[4,8,32]×[4,32,8]和[4,8,8]×[4,8,32]。
scripts/ncnn_v2_rewrite.py将每个运算拆为4个通道切片、二维Gemm、重新成形及通道拼接。
二维Gemm在本机运行库声明支持Vulkan；不改变矩阵运算、权重、头数或音频上下文。
只支持当前v2的固定batch=1、8 tokens、4 heads、head dimension=32图签名，非通用转换器。
该改写依据[ncnn Gemm参数定义](https://raw.githubusercontent.com/Tencent/ncnn/master/src/layer/gemm.cpp)，
实际以本机固定20261020运行库及完整序列数值结果验收。

--gpu-attention显式启用改写。测试独立解释生成图，和NumPy批量矩阵乘法比较，
同时拒绝未知MatMul模式；实际手机进一步验证全部78帧。

## 复现已通过的新变体

```bash
cache/student-venv/bin/python scripts/student_mobile_export.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --output outputs/v2-mobile-export-gemm-001 \
  --convert-ncnn --preserve-letterbox --gpu-attention

cache/student-venv/bin/python scripts/student_mobile_fixture.py \
  --checkpoint outputs/emotion-mouth-v2-001/checkpoint.pt \
  --dataset data/emotion-overfit-001/manifest.json \
  --model-manifest outputs/v2-mobile-export-gemm-001/manifest.json \
  --output outputs/v2-mobile-fixture-gemm-001

python3 scripts/mobile_model.py \
  --model-manifest outputs/v2-mobile-export-gemm-001/manifest.json \
  --fixture-manifest outputs/v2-mobile-fixture-gemm-001/manifest.json \
  --serial c9fb5e39
```

本机已存在这些结果，复跑时请使用新目录，避免覆盖。

## 2026-11-05实机结果

报告model_20261105T082946Z_4b4f853c，status=validated_with_cpu_fallback，
phone_model_numerics_validated=true：

| 指标 | 结果 |
| --- | --- |
| 序列 | 78帧，全部有限，约1567万个元素检查 |
| 最大绝对 / 归一化误差 | 1.27e-5 / 0.608337 |
| 验收阈值 | abs_error <= 1e-5 + 1e-4*abs(reference)，未变 |
| 状态重置误差 | 0 |
| 不支持Vulkan的计算层 | 从3个减为1个，仅GridSample |
| 单帧中位 / P95 / 最大 | 17.42 / 18.36 / 18.55ms |
| 模型设置及权重上传 | 18.52秒 |
| 原生进程首次RGB | 19.17秒 |
| 进程峰值RSS | 约173.4MiB，包含约60MiB测试参考数据 |

最大绝对差不是通过与否的唯一标准；相对项取决于对应像素亮度，填充区修正后的最大归一化误差低于门槛。
不能把未修正的原始raw_v2也标为通过。
本轮缓存/温度未受控，不据此宣称相对于其他运行记录有确定加速比。

单帧计算满足当前16FPS的62.5ms预算，但仍非纯GPU，gpu_only_execution_verified=false。
音频前处理、图片文件解码、APK、播放器及长时间温度/降频测试尚未接入。
**首次输出仍明显超过2秒，端到端目标没有达成**。
独立音频口型同步、跨人物泛化和完整视频质量也未验收通过。

下一阶段优先优化加载/预热，并接入手机音频和播放链路，分开测应用初始化与请求到首帧延迟。
