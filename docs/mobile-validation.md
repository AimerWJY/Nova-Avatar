# 小米17手机端验证

当前实机用户回传：小米17，ro.product.model=25113PN0EC，ro.soc.model=SM8850，Android 16。
MemTotal=11285460kB（约10.76GiB系统可见内存），对应12GB标称物理内存版本；
MemAvailable=5245004kB（约5.00GiB），为当时系统快照，不是单个应用可安全使用的内存上限。
较小的MemFree不是内存耗尽，Linux可回收缓存体现在MemAvailable中。扩展内存不作为模型预算。

## 设备基线采集

在连接手机且已获USB调试授权的Mac上运行，不在云服务器上运行：

```bash
python3 scripts/mobile_profile.py
```

多台设备时加--serial。输出位于本仓库reports/mobile/profile_*.json（被Git忽略）。
只读取指定系统属性和/proc/meminfo；不安装应用、不上传素材、不运行模型或改变手机配置。
同时记录本机SDK、NDK、CMake、Java是否就绪，方便后续构建实机测试程序。

## 原生算子与驱动探测

在连接手机的Mac上执行：

```bash
# 只交叉编译，不操作手机。
python3 scripts/mobile_benchmark.py --build-only

# 枚举NNAPI后端；仅对GPU/accelerator类型测试，不自动回退CPU。
python3 scripts/mobile_benchmark.py

# 显式测试参考CPU，结果单独记录。
python3 scripts/mobile_benchmark.py --include-cpu --iterations 3 --warmup 1
```

需要已安装的NDK；可以用--ndk传入目录，多台设备用--serial选择。
编译输出位于build/mobile（Git忽略），报告与设备stdout/stderr位于reports/mobile/benchmark_*。
工具将测试二进制推送到/data/local/tmp/nova-avatar-bench/<本次编号>/，不安装APK、不读取用户图片。
设备上的测试程序保留在该专用临时目录供复查，不会自动删除其他文件。
单轮设备进程限时180秒，默认3次预热、20次测量；CPU示例仅3次测量，用于冒烟验证而非稳定性能结论。

Android 15起NNAPI已弃用，因此这里只用它探测驱动，不作为最终部署后端选择。
若只有nnapi-reference或硬件算子不支持，报告status=no_valid_hardware_results；
命令退出0表示诊断正常完成，不表示手机加速验证通过。后续需接入QNN或其他GPU运行时。

测试内容严格限定为：

- tokens=128、width=512、hidden=2048的两层全连接＋ReLU，约8MiB随机权重。
- 64×64×32特征图上的3×3 Conv2D＋ReLU，约36KiB随机权重。

这些不是Wan DiT或VAE，没有Attention、GELU、时序卷积、音频条件、蒸馏权重、量化或完整视频生成。
输入/权重为固定随机种子生成的FP32值，默认允许驱动降至FP16；加--fp32禁用该选项。
分别检查整个小图能否由单个设备支持，使用createForDevices指定单设备，防止NNAPI运行时CPU回退；
驱动内部调度仍是黑箱，不据此宣称所有计算都已在NPU上执行。输出只抽查部分点与手写CPU参考值的误差。
延迟包括输入/输出绑定及同步compute，排除编译、创建Execution和参考值校验；编译时间另记。
RSS为ADB shell进程，部分驱动内存不包含，不能用作APK或完整Student的峰值内存。

### 2026-11-04实机结果

小米17、Android 16、系统OS3.0.319.0.WPCCNXM，NNAPI只枚举到类型2的nnapi-reference，
没有GPU/accelerator驱动。本轮没有获得有效GPU/NPU延迟，不代表手机硬件没有GPU/NPU。

显式CPU冒烟测试（预热1次、测量3次）成功，两类算子抽样数值校验通过：

| 负载 | CPU中位延迟 | 抽样最大绝对误差 |
| --- | --- | --- |
| 两层FC＋ReLU | 9.02ms | 2.10e-9 |
| Conv2D＋ReLU | 1.68ms | 2.79e-9 |

进程峰值RSS约34.1MiB。此数据只证明测试与报告链路可用；不用于估算Student参数预算、
视频帧率或首播延迟，也未做10分钟温度/降频测试。原始报告留在本地reports/mobile中，不提交Git。

## 接下来需要验证的内容

### Vulkan GPU路径（2026-11-04已实机验证）

NNAPI未暴露加速设备后，使用不依赖NNAPI的Vulkan计算队列：

```bash
python3 scripts/mobile_benchmark.py --backend vulkan --build-only
python3 scripts/mobile_benchmark.py --backend vulkan
```

代码只依赖本机NDK和其中的glslc/spirv-val，不额外下载运行时或模型。编译CPU端ARM64程序和
SPIR-V着色器并进行SPIR-V结构校验，随后通过ADB推送到本次专用临时目录。
Vulkan程序只选择integrated/discrete GPU，不接受CPU实现，不通过NNAPI，也不声称使用NPU。
构建产物位于build/mobile/vulkan。--include-cpu不能和--backend vulkan一起使用。

着色器是自定义16×16分块FP32矩阵乘法，执行两层投影，中间用ReLU或tanh近似GELU。
不包含偏置、Attention、归一化、位置编码、音频条件或完整扩散步。GPU缓冲区使用host-visible内存，
必要时flush/invalidate，并设置两次投影之间的读写屏障及GPU写至CPU读屏障；通过fence等待完成。
记录墙钟提交/等待时间和GPU时间戳（如果队列支持）。没有FP16、INT8、INT4或专用矩阵指令优化。

实机识别到Adreno (TM) 840，预热3次、测量20次：

| 负载 | 墙钟中位耗时 | GPU时间戳中位耗时 | 抽样最大绝对误差 |
| --- | --- | --- | --- |
| 128 tokens / width 512 / hidden 2048，ReLU | 9.04ms | 8.42ms | 0 |
| 同尺寸，近似GELU | 12.09ms | 8.47ms | 9.31e-10 |
| 17 tokens / width 33 / hidden 65，ReLU，边界尺寸检查 | 0.275ms | 0.021ms | 2.91e-11 |

两项主要负载的权重约8MiB，显式GPU buffer约9.5MiB，整轮进程峰值RSS约43.1MiB（不含完整驱动内存）。
pipeline创建约96ms，数据初始化、分配、pipeline创建和CPU数值校验不计入上述单轮延迟。
每轮包含GPU队列提交和fence等待；墙钟与GPU时间的差值受CPU调度及驱动开销影响。
输出全量检查有限值，参考值仅抽查首尾token的部分通道；数值通过阈值为abs_error <= 1e-5 + 1%*abs(reference)。

此结果确认GPU路径可用及所测运算正确，不代表加速目标达成。之前NNAPI CPU冒烟测试只有3次样本，
与本轮次数和实现不同，不能直接作为严格加速比；当前数据没有证明FP32自定义实现有明显优势。
下一阶段需要成熟的FP16/量化推理运行时、Attention和VAE真实子网络测试，之后才确定Student预算。

### ncnn Vulkan FP16与Attention验证入口

在连接手机的Mac上执行：

```bash
python3 scripts/mobile_runtime.py
# 网络慢时，从官方发布页下载同版本ZIP，然后导入：
python3 scripts/mobile_runtime.py --archive /绝对路径/ncnn-20261020-android-vulkan-shared.zip
python3 scripts/mobile_benchmark.py --backend ncnn --build-only
python3 scripts/mobile_benchmark.py --backend ncnn
```

运行库固定为Tencent/ncnn的20261020官方Android Vulkan shared包，URL和官方SHA256固定在
configs/ncnn-runtime.json；下载不完整或校验不通过时不会解压/运行。仅提取arm64-v8a及许可文件，
缓存在cache/mobile，不提交Git。不通过第三方镜像获取可执行文件。

测试包括固定随机权重的两层投影＋近似GELU、128 tokens/256维/4头的Self-Attention，
以及64×64×32特征图上的Conv2D＋Swish＋Conv2D。最后一项是二维解码代理，**不是Wan VAE**，
不包含时序卷积、上采样或训练权重；Attention不含RoPE或mask，也不是完整DiT块。
Convolution3D仅探测Vulkan工厂是否存在，不代表实际图可导出或运行。

显式创建Vulkan层并调用VkMat接口，缺少GPU实现时报告失败；FP32 CPU计算仅用于全量输出数值校验，
不计入GPU性能。分别运行FP32、FP16请求，以及FP16存储但关闭FP16算术的对照请求，检查GPU输入实际标量位宽及FP16能力。
FP16 arithmetic选项不等于已逐条审计GPU机器指令，单个算子内部仍可能采用混合精度。

延迟包括命令记录、输入上传、GPU运算、输出下载与同步；pipeline创建和权重上传另记。
这与前一轮自定义Vulkan内核的常驻输入测法不同，不直接给出两者加速比。
数值门槛FP32为abs_error <= 1e-6 + 0.1%*abs(reference)，FP16为1e-4 + 2%*abs(reference)，
校验全部输出元素。只有误差和位宽同时通过才进入hardware_results。

#### 2026-11-05首轮实机结果

官方运行库SHA256校验、NDK编译及实机运行完成，设备为Adreno (TM) 840。
首轮报告benchmark_20261105T030147Z_e43da133，预热3次、测量20次：

| 负载 | FP32中位耗时 | FP16请求中位耗时 | 校验结果 |
| --- | --- | --- | --- |
| FFN＋近似GELU | 4.50ms | 3.21ms | 两种精度通过 |
| Self-Attention | 1.83ms | 无有效成绩 | FP32通过，FP16输出含非有限值 |
| 二维解码代理 | 1.90ms | 1.81ms | 两种精度通过 |

状态为hardware_microbench_partial，不能报告为全部通过。FP16 Attention异常的根因尚未定位，
不据此认定为模型、驱动或GPU硬件故障。增加关闭FP16算术、保留FP16存储的对照测试，
失败结果仍单独保留，不并入有效耗时。失败信息区分GPU输出和FP32 CPU参考，并记录首个非有限输出坐标。

首轮各负载的pipeline创建与权重上传约1.0–10.2秒，不能忽略初始化并声称满足首播2秒。
峰值进程RSS约163.9MiB，不是完整模型内存。Convolution3D Vulkan工厂不可用，
因此不能直接假定原始Wan时序VAE可在这条后端运行。
即使这些小图通过，也不能据此承诺手机生成帧率、首播2秒或完整14B模型可部署。

#### 同日精度对照结果

报告benchmark_20261105T030431Z_6b3ddbf3，预热3次、测量20次。三项负载各测三种设置，
共9项中8项通过；开启FP16算术的Attention仍失败，报告整体仍为hardware_microbench_partial。

| 负载 | FP32中位耗时 | FP16存储、开启FP16算术 | FP16存储、关闭FP16算术 |
| --- | --- | --- | --- |
| FFN＋近似GELU | 4.55ms | 3.20ms，通过 | 3.19ms，通过 |
| Self-Attention | 1.84ms | GPU输出非有限值，失败 | 1.85ms，通过 |
| 二维解码代理 | 1.89ms | 1.76ms，通过 | 1.85ms，通过 |

关闭FP16算术后的Attention检查32768个输出元素，最大绝对误差1.61e-7，
GPU输入存储实际为16位。仅改变算术选项即可使该负载通过，提示异常与所选精度路径有关，
但还不足以区分运行时、shader、驱动或测试调用的根因，也不承诺所有内部指令都使用FP32。

候选完整网络的起始设置采用FP16存储、关闭FP16算术，先验证正确性；
后续再逐层验证是否启用FP16算术。完整图需包含层间转换及同步，不能把表中孤立负载耗时相加当作视频性能。
本轮峰值进程RSS约154.6MiB，pipeline与权重上传仍单独记录，不用于完整模型预算。

### 连接候选网络验证（2026-11-05已实机通过）

```bash
python3 scripts/mobile_benchmark.py --backend candidate --build-only
python3 scripts/mobile_benchmark.py --backend candidate --serial c9fb5e39
# 较短的冒烟验证，至少8帧：
python3 scripts/mobile_benchmark.py --backend candidate --iterations 8
```

实现位于mobile/native/candidate_probe.cpp；默认160帧，最多160帧，预热3次，
设备进程最多运行300秒。图中所有层都使用Vulkan接口，缺少GPU实现即失败。
共享运行库和算子构建工具与ncnn探测相同，共享头文件SHA256也写入构建记录。

结构候选recurrent_conditioned_rgb_v0有246112个参数，使用固定随机权重和合成输入：

1. 256×256 RGB输入经过两层stride=2的卷积及Swish，得到64×64×32特征图；
   全局平均池化与投影得到128维身份条件，两项结果都缓存于GPU。
2. 每帧输入8个历史块、每块40维的音频特征形状数据，经投影、Swish、4头Self-Attention、
   Flatten及投影得到128维音频条件。数据由随机数构造，**没有提取真实WAV的Mel特征**。
3. 音频条件与上一帧128维状态、身份条件相加，经投影及Swish产生下一帧状态；
   投影为32通道后与身份空间特征广播相加。
4. 卷积、Swish及两次双线性上采样，再经RGB卷积和Sigmoid输出256×256×3。

GPU状态跨帧保留，参考图只编码一次。每轮上传音频特征、下载RGB和状态并同步，
测量包含这些传输、命令记录和执行；不含图片文件解码、WAV预处理、视频编码或音画播放。
输入和输出GPU存储均实际观测为16位，关闭FP16算术选项。

每帧RGB和状态都检查有限值，前8帧与同一结构的独立FP32 CPU运行结果逐元素比较，
阈值为abs_error <= 1e-4 + 2%*abs(reference)。CPU参考在GPU计时后运行。
序列完成后显式重置状态，重放首帧，检查RGB和状态与首次结果一致。
这不是训练质量评估，也不是独立数学实现对运行时内部算子的交叉验证。

本机报告benchmark_20261105T031247Z_7804a397，status=candidate_graph_validated：

| 指标 | 结果 |
| --- | --- |
| 连续计算 | 160帧，全部RGB和状态有限 |
| 单帧中位 / P95 / 最大耗时 | 2.29 / 2.42 / 3.11ms |
| 整段循环墙钟 | 471.65ms，包含循环中的有限值检查和RSS采样 |
| 预热后首帧计算 | 1.73ms |
| 原生程序进入后首次RGB输出 | 1545.58ms |
| pipeline创建及权重上传 | 1479.37ms |
| 一次性参考图编码 | 17.10ms |
| GPU阶段进程峰值RSS | 117.3MiB |
| FP32 CPU参考检查 | 前8帧，共1573888个RGB和状态元素 |
| 最大绝对 / 归一化误差 | 2.38e-7 / 0.00094 |
| 状态重置重放归一化误差 | 0 |

新进程首次输出计时从native main内部开始，不含Android进程启动、动态库加载或模型文件读取。
驱动和shader缓存未清空，所以1545.58ms不能当成应用首次安装后的冷启动或用户首播时间。
RSS从约111.3MiB升至117.3MiB后，本轮采样保持稳定；不包含完整GPU驱动内存，也不构成长期无泄漏证明。

160帧以最快速度背靠背计算，**没有播放成10秒视频，也没有持续运行10秒或10分钟**。
报告的synthetic_graph_*预算字段只比较该合成计算图与16FPS/2秒门槛；
can_infer_video_generation_speed和video_quality_validated仍为false。
约24.6万参数的这个图不是已确定的Student规模，不是Wan剪枝产物，也没有蒸馏权重。
它不包含扩散过程或原Wan时序VAE，当前输出不具备有用的人物画质。

对应PyTorch模型及已通过的导出验证流程现位于[student-export.md](student-export.md)。
使用--bundle-manifest指定导出结果后，手机加载给定参数和输入，额外对比PyTorch参考张量；
不指定时仍保留原始随机权重结构探测行为。两者的来源与结果应分别记录。
下一阶段接入真实图片与音频特征，再用小规模教师配对样本检查口型、身份保持与跨帧稳定性。
模型结构和容量是否足够，必须由这些质量测试决定。

### 后续模型验证

1. 候选C++连接图、PyTorch梯度及导出参数一致性已通过，接入真实数据检验小样本拟合。
2. 比较代表性DiT计算、音频编码和VAE解码的延迟、内存和数值误差。
3. 检查实际算子分配和CPU回退，不能仅凭创建NPU session就认定全模型在NPU运行。
4. 加入帧输出及音频播放，分开记录人物首次初始化、预热请求首播和持续生成吞吐。
5. 连续运行至少10分钟，检查温度、降频、内存增长和队列延迟是否持续累积。

先以低分辨率胸像及16FPS作为验证起点，再讨论更高画质和半身手势。Student参数规模、
进程内存预算、量化精度、分块大小都待实测决定。现有14B教师不直接部署到手机。
本仓库已有设备信息采集、NNAPI/CPU探测、Vulkan GPU算子验证及随机权重候选连接图，
已增加单段教师输出拟合及训练后参数的手机合成输入校验，见[student-overfit.md](student-overfit.md)。
尚无可验收的Android人物生成App、QNN配置、跨身份质量验证或扩散轨迹/步数蒸馏实现。

参考：

- https://developer.android.com/tools/adb
- https://developer.android.com/ndk/guides/neuralnetworks
- https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- https://developer.android.com/ndk/guides/graphics/getting-started
- https://docs.vulkan.org/guide/latest/synchronization.html
