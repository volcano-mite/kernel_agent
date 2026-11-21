# Add 与 RMSNorm 融合实验

## 1. 从公式推导

设输入 $X$ 和残差 $R$ 的形状都是 $B\times H$，权重 $w$ 有 $H$ 个元素，每行共享它。这里 $H=4096$，$\varepsilon=10^{-5}$。用 $Q$ 表示舍入到 BF16，$b$ 表示行号，$j$ 表示列号。

先做 Add。输入从 BF16 转成 FP32 相加，再舍入一次，得到拆分版中间张量的数学定义。

$$
s_{b,j}=Q\bigl(\mathrm{FP32}(X_{b,j})+\mathrm{FP32}(R_{b,j})\bigr).
$$

然后按行平方、取平均，再加上 $\varepsilon$ 后开方。平方和在 GPU 上用 FP32 累加，$\varepsilon$ 让全零行的分母仍为正。

$$
m_b=\frac{1}{H}\sum_{k=0}^{H-1}s_{b,k}^2,
\qquad
\mathrm{RMS}_b=\sqrt{m_b+\varepsilon}.
$$

一行的所有输出共用同一个分母。代码用 `rsqrtf` 求倒数，之后逐元素乘以倒数和列权重，最后舍入到 BF16。

$$
a_b=(m_b+\varepsilon)^{-1/2},
\qquad
Y_{b,j}=Q(s_{b,j}\,a_b\,w_j).
$$

把 Add 的定义代入 $m_b$ 和 $Y_{b,j}$ 后，可以看出 $s_{b,j}$ 有两个用途。它先参与平方和，再用于归一化输出。例如，若两个长度为 2 的输入行分别为 `[1, 2]` 和 `[2, 2]`，加法结果是 `[3, 4]`。均方为 $(9+16)/2=12.5$，若权重为 `[1, 1]`，最终 BF16 舍入前的输出约为 `[0.849, 1.131]`。融合时保留 Add 后的 BF16 舍入，因为舍入过的值同时决定平方和与输出分子。

## 2. 哪些数据要写入显存再读回来

```text
拆分版，两个 kernel 在同一 CUDA stream 上依次执行

全局内存 X、R ──读──▶ Add ──写──▶ 全局内存 sum
                                       │
                                       └──读──▶ RMSNorm ──写──▶ 全局内存 Y
全局内存 weight ────────────────────────读──────────▲

融合版，一次 kernel launch

全局内存 X、R ──读──▶ 加法并舍入 ──▶ 每线程 values[]
                                           │
                                           ├─▶ 局部平方和 ──▶ block shared memory
                                           │                         │
全局内存 weight ──读─────────────────────────┼─────────────────────────┤
                                           └─▶ 归一化并乘权重 ──写──▶ 全局内存 Y
```

| 操作 | 拆分版源码 | 融合版源码 |
| --- | --- | --- |
| 读输入、残差 | [第 27 行](bench.cu#L27) | [第 41 行](bench.cu#L41)、[第 43 行](bench.cu#L43) |
| 将 Add 结果写入全局内存 | [第 27 行](bench.cu#L27)的 `sum[i] = ...` | 第 45 行将结果留在每线程的 `values[]` |
| 从全局内存读 Add 结果 | [第 67 行](bench.cu#L67)把 `sum` 作为 `x` 传入，因此第 41 行的 `x[i]` 读的是 `sum[i]` | 第 46、61 行在同一 kernel 内复用线程私有值 |
| 跨线程交换局部平方和 | [第 48–54 行](bench.cu#L48)访问 block shared memory | 同左 |
| 读权重、写最终输出 | [第 60–61 行](bench.cu#L60) | 同左 |

`sum` 是第 106 行通过 `cudaMalloc` 分配的设备缓冲区。两个 kernel 之间用这个全局内存地址传递结果。第 48 行写的 `partial` 是同一个 block 内共有的片上 shared memory。第 45 行写的 `values[]` 是每线程私有临时数据，此次编译展开后没有报告寄存器 spill。

中间张量有 $BH$ 个 BF16 元素。一次写入加一次读取，每个元素共 $2+2=4$ 字节，因此融合省去 $4BH$ 字节的逻辑访问。$B=1$ 时是 16 KiB，$B=14509$ 时约为 226.7 MiB。仅计输入、残差、中间结果和输出，拆分版每元素 10 B，融合版 6 B。两者还要读 weight。全局内存读写可能经过 L2，测得的 DRAM 流量取决于缓存和硬件。小 batch 可以重点观察省掉一次启动的影响，大 batch 可以重点观察中间张量带来的流量和寄存器用量。若后续算子还需要 $s$，可以让融合 kernel 顺手写出它，仍可省去归一化阶段对它的全局读取和一次启动。

## 3. 逐行读 bench.cu

行号对应 [bench.cu](bench.cu)。空行和只负责关闭作用域的括号随相邻代码理解。

### 头文件、配置与错误检查

| 源码行 | 说明 |
| --- | --- |
| [2](bench.cu#L2) | cuda_bf16.h 提供 BF16 类型与转换函数。 |
| [3](bench.cu#L3) | cuda_runtime.h 提供设备、内存、事件和错误检查 API。 |
| [5](bench.cu#L5) | algorithm 提供排序。 |
| [6](bench.cu#L6) | array 提供固定大小数组。 |
| [7](bench.cu#L7) | cmath 提供 CPU 开方和绝对值。 |
| [8](bench.cu#L8) | cstdio 提供日志和 CSV 输出。 |
| [9](bench.cu#L9) | cstdlib 提供错误时退出进程的函数。 |
| [10](bench.cu#L10) | vector 提供随 batch 调整大小的 CPU 缓冲区。 |
| [12](bench.cu#L12) | 为 CUDA BF16 类型起简名。 |
| [13](bench.cu#L13) | 固定每行 4096 列。 |
| [14](bench.cu#L14) | 固定每个 block 256 个线程。 |
| [15](bench.cu#L15) | 每线程 4096 / 256 = 16 个元素。 |
| [17](bench.cu#L17) | 定义 CUDA 状态检查函数。 |
| [18](bench.cu#L18) | 判断 API 返回值是否成功。 |
| [19](bench.cu#L19) | 将 CUDA 错误描述写到 stderr。 |
| [20](bench.cu#L20) | 退出码 1 表示 CUDA 调用失败。 |

### 独立 Add

| 源码行 | 说明 |
| --- | --- |
| [24](bench.cu#L24) | 定义逐元素 Add kernel，sum 指向全局内存。 |
| [25](bench.cu#L25) | 把 block 和线程索引展平成元素下标。 |
| [26](bench.cu#L26) | 跳过超出元素总数的线程。 |
| [27](bench.cu#L27) | 从全局内存读 x 和 residual，转 FP32 相加，舍入成 BF16 并写入全局内存 sum。 |

### RMSNorm 公共实现

| 源码行 | 说明 |
| --- | --- |
| [31](bench.cu#L31) | 用编译期布尔参数区分是否在同一个 kernel 内做 Add。 |
| [32–33](bench.cu#L32) | 设备函数接收输入、残差、权重、输出及调用方提供的 shared memory。 |
| [34](bench.cu#L34) | lane 是整个 block 内的线程编号，范围 0–255。 |
| [35](bench.cu#L35) | row 是当前行首元素的偏移量，由 blockIdx.x 决定。 |
| [36](bench.cu#L36) | 每线程保留 16 个 FP32 值供输出阶段复用。 |
| [37](bench.cu#L37) | 初始化该线程的局部平方和。 |
| [38](bench.cu#L38) | 请求编译器展开定长循环。 |
| [39](bench.cu#L39) | 遍历线程负责的 16 个元素。 |
| [40](bench.cu#L40) | 连续线程访问连续列。线程 t 负责 t、t+256 等列。 |
| [41](bench.cu#L41) | 读全局内存并转 FP32。拆分版此处读的是 sum，融合版读的是原始 x。 |
| [42](bench.cu#L42) | 在编译期选择是否保留 Add 分支。 |
| [43](bench.cu#L43) | 从全局内存读 residual，相加后先舍入 BF16，再转回 FP32，以便后续计算与拆分版对齐。 |
| [45](bench.cu#L45) | 将当前值暂存在每线程的 values 数组。 |
| [46](bench.cu#L46) | 累加当前值的平方。 |
| [48](bench.cu#L48) | 把当前线程的局部平方和写进 shared memory 的独立槽位。 |
| [49](bench.cu#L49) | 等待全部线程完成写入，再开始读其他线程的部分和。 |
| [50](bench.cu#L50) | 将归约跨度依次缩到 128、64，直到 1。 |
| [51](bench.cu#L51) | 只有编号小于本轮跨度的线程做加法。 |
| [52](bench.cu#L52) | 读取相隔 stride 的两个部分和，把相加结果留在前半区。 |
| [54](bench.cu#L54) | 全 block 同步，下一轮可以读取本轮结果。同步在 if 外面，因此所有线程都会经过。 |
| [56](bench.cu#L56) | partial[0] 是整行平方和。除以 hidden，加入 epsilon，再计算平方根的倒数。 |
| [57](bench.cu#L57) | 请求编译器展开输出循环。 |
| [58](bench.cu#L58) | 再遍历当前线程负责的 16 列。 |
| [59](bench.cu#L59) | 算出当前列号，用来索引共享权重。 |
| [60–61](bench.cu#L60) | 用 values[j] 乘以倒数和从全局内存读来的 weight[col]，舍入 BF16 并写进全局输出。 |

### 两个 RMSNorm 入口

| 源码行 | 说明 |
| --- | --- |
| [65](bench.cu#L65) | 声明拆分流程的第二个 kernel，接收已算好的 sum。 |
| [66](bench.cu#L66) | 为每个 block 分配 256 个 FP32 部分和槽位，共 1024 B。 |
| [67](bench.cu#L67) | 传 sum 作为 normalize 的输入。false specialization 会移除 residual 访问，可传 nullptr。 |
| [70–71](bench.cu#L70) | 声明融合 kernel，直接接收原始 x 和 residual。 |
| [72](bench.cu#L72) | 分配同样大小的 shared memory。 |
| [73](bench.cu#L73) | true specialization 在读取输入后立刻做 Add。 |

### 实验配置与数据准备

| 源码行 | 说明 |
| --- | --- |
| [76](bench.cu#L76) | 进入 CPU 端 benchmark。 |
| [83](bench.cu#L83) | 初始化设备信息结构。 |
| [84](bench.cu#L84) | 查询可见设备 0。设置 CUDA_VISIBLE_DEVICES 后，0 指其可见设备列表的第一张卡。 |
| [85](bench.cu#L85) | 把 GPU 名写到 stderr，让 stdout 保持 CSV 格式。 |
| [86](bench.cu#L86) | 每次计时连续提交 100 次调用。 |
| [87](bench.cu#L87) | 每个版本重复 7 轮。 |
| [88](bench.cu#L88) | 输出 CSV 列名，延迟单位为微秒。 |
| [89](bench.cu#L89) | 遍历六种 batch，先测小矩阵，再测大矩阵。 |
| [91](bench.cu#L91) | 计算 x、residual 和输出的元素数。 |
| [92](bench.cu#L92) | 为 CPU 输入和权重分配数组。 |
| [93](bench.cu#L93) | 逐元素初始化两个输入。 |
| [94](bench.cu#L94) | 用取模生成确定性 x，并转 BF16。 |
| [95](bench.cu#L95) | 用另一组周期和值域生成 residual，并转 BF16。 |
| [97](bench.cu#L97) | 遍历 4096 个权重。 |
| [98](bench.cu#L98) | 给每列生成确定性 BF16 权重。 |
| [101](bench.cu#L101) | 声明设备指针，分别指向两个输入、权重、中间结果和两个输出。 |
| [102](bench.cu#L102) | 计算每个完整张量所需字节数。 |
| [103](bench.cu#L103) | 在 GPU 上分配 x。 |
| [104](bench.cu#L104) | 在 GPU 上分配 residual。 |
| [105](bench.cu#L105) | 在 GPU 上分配一个长度为 hidden 的权重向量。 |
| [106](bench.cu#L106) | 仅供拆分路径使用的 sum 缓冲区。 |
| [107](bench.cu#L107) | 分配拆分路径的输出。 |
| [108](bench.cu#L108) | 分配融合路径的输出。 |
| [109](bench.cu#L109) | 将 x 从 CPU 拷贝到 GPU。 |
| [110](bench.cu#L110) | 将 residual 从 CPU 拷贝到 GPU。 |
| [111](bench.cu#L111) | 将权重从 CPU 拷贝到 GPU。 |

### 调用、预热与正确性校验

| 源码行 | 说明 |
| --- | --- |
| [113](bench.cu#L113) | 定义一次拆分调用，按引用捕获设备指针和形状。 |
| [114](bench.cu#L114) | 先启动 Add，向上取整计算 block 数。这个 kernel 写入 sum。 |
| [115](bench.cu#L115) | 同一默认 stream 再启动 RMSNorm，每行一个 block。该 kernel 读取 sum。 |
| [117](bench.cu#L117) | 定义一次融合调用。 |
| [118](bench.cu#L118) | 每行一个 block，只提交一个 kernel。 |
| [120](bench.cu#L120) | 执行 20 轮预热。 |
| [121](bench.cu#L121) | 提交一次拆分调用。 |
| [122](bench.cu#L122) | 提交一次融合调用。 |
| [124](bench.cu#L124) | 检查 CUDA launch 错误。 |
| [125](bench.cu#L125) | 分配两份 CPU 端输出用于验证。 |
| [126](bench.cu#L126) | 把拆分结果拷回 CPU。 |
| [127](bench.cu#L127) | 把融合结果拷回 CPU。 |
| [128](bench.cu#L128) | CPU 参考按行计算。 |
| [129](bench.cu#L129) | 初始化长度为 hidden 的临时加法结果。 |
| [130](bench.cu#L130) | 用 double 累计 CPU 参考的均方。 |
| [131](bench.cu#L131) | 遍历当前行的列。 |
| [132](bench.cu#L132) | 将行列索引展开为数组偏移量。 |
| [133–134](bench.cu#L133) | 模拟 Add 的 BF16 舍入，再转 FP32。 |
| [135](bench.cu#L135) | 将当前平方除以 hidden 并累加。 |
| [137](bench.cu#L137) | 再遍历每列，验证归一化后的结果。 |
| [138](bench.cu#L138) | 重新计算线性索引。 |
| [139–140](bench.cu#L139) | 用本行均方、epsilon 和列权重计算参考值，保存为 float。 |
| [141](bench.cu#L141) | 读取融合 BF16 输出并转 float。 |
| [142–143](bench.cu#L142) | 检查拆分与融合的数值相等，并要求融合值对 CPU 参考的绝对误差不超过 1e-5 + 0.004 × &#124;expected&#124;。 |
| [144](bench.cu#L144) | 报告首个失败的 batch、行、列。 |
| [145](bench.cu#L145) | 以退出码 2 中止，此 batch 不进入计时。 |

### 事件计时和资源回收

| 源码行 | 说明 |
| --- | --- |
| [158](bench.cu#L158) | 声明测量区间的 CUDA events。 |
| [159](bench.cu#L159) | 创建开始事件。 |
| [160](bench.cu#L160) | 创建结束事件。 |
| [161](bench.cu#L161) | 定义可以接受任一运行路径的计时函数。 |
| [162](bench.cu#L162) | 在默认 stream 上记录开始事件。 |
| [163](bench.cu#L163) | 遍历本轮的 100 次调用。 |
| [164](bench.cu#L164) | 提交当前路径的一次调用。 |
| [166](bench.cu#L166) | 检查 launch 错误。 |
| [167](bench.cu#L167) | 在这些调用之后记录结束事件。 |
| [168](bench.cu#L168) | 等待结束事件完成，也就等到了该测量区间的设备任务执行完毕。 |
| [169](bench.cu#L169) | 声明毫秒结果变量。 |
| [170](bench.cu#L170) | 读取两个事件之间的时间差。 |
| [171](bench.cu#L171) | 转成微秒并除以调用次数，得到每次调用的平均延迟。 |
| [173](bench.cu#L173) | 分别保存两条路径的 7 个测量值。 |
| [174](bench.cu#L174) | 逐轮进行配对测量。 |
| [175](bench.cu#L175) | 偶数轮先测拆分路径。 |
| [176](bench.cu#L176) | 保存拆分测量值。 |
| [177](bench.cu#L177) | 保存融合测量值。 |
| [178–180](bench.cu#L178) | 奇数轮交换顺序，减少总是让某一版本先跑造成的偏差。 |
| [183](bench.cu#L183) | 排序拆分延迟，以便取中位数和范围。 |
| [184](bench.cu#L184) | 对融合延迟做同样的排序。 |
| [185](bench.cu#L185) | 取拆分延迟的中位数。 |
| [186](bench.cu#L186) | 取融合延迟的中位数。 |
| [187–189](bench.cu#L187) | 写出 batch、两组中位数、加速比、两组最小和最大延迟。加速比为拆分中位数除以融合中位数。 |
| [190](bench.cu#L190) | 刷新 stdout，使该 batch 的结果立即写入输出文件。 |
| [191](bench.cu#L191) | 销毁开始事件。 |
| [192](bench.cu#L192) | 销毁结束事件。 |
| [193](bench.cu#L193) | 遍历本 batch 分配的 GPU 缓冲区。 |
| [194](bench.cu#L194) | 释放每个设备缓冲区。 |

## 4. 已跑结果

用 CUDA 13.0 的 `nvcc` 为 `sm_120` 编译成功。ptxas 报告如下。

| Kernel | 每线程寄存器 | 每 block shared memory | Spill load / store |
| --- | ---: | ---: | ---: |
| Add | 14 | 0 B | 0 / 0 B |
| 独立 RMSNorm | 30 | 1024 B | 0 / 0 B |
| 融合 Add + RMSNorm | 40 | 1024 B | 0 / 0 B |

融合较独立 RMSNorm 多用 10 个寄存器。更高的寄存器用量可能影响同时驻留的 block 数。

一次历史运行的 CUDA event 中位延迟如下。拆分版的小 batch 测量中出现过明显离群值。复跑命令见下一节。

| Batch | 拆分版 (µs) | 融合版 (µs) | 加速比 |
| ---: | ---: | ---: | ---: |
| 1 | 6.204 | 4.141 | 1.498× |
| 64 | 6.210 | 6.188 | 1.004× |
| 14509 | 4982.767 | 3135.038 | 1.589× |

## 5. 在 RTX 5090 宿主机复跑

从仓库根目录执行。将 `GPU-UUID` 替换成 `nvidia-smi -L` 列出的空闲 RTX 5090 的 UUID。

```bash
source scripts/gpu_env.sh
nvcc -std=c++17 -O3 -arch=sm_120 -Xptxas=-v -ccbin "${CXX:-g++}" \
  experiments/add_rmsnorm/bench.cu -o /tmp/kernel-agent-add-rmsnorm
CUDA_VISIBLE_DEVICES=GPU-UUID /tmp/kernel-agent-add-rmsnorm \
  > experiments/add_rmsnorm/results.csv
```

程序在每个 batch 上预热，再逐元素比较两个 GPU 输出，并用 CPU 公式检查误差。通过后才计时，交替运行两种实现，各测 7 轮，每轮 100 次。CSV 提供 CUDA event 测得的每次调用中位延迟、范围和拆分延迟除以融合延迟得到的加速比。分配、数据拷贝和正确性检查都在计时区间外。

只做 batch=1 正确性检查时，运行 `CUDA_VISIBLE_DEVICES=GPU-UUID /tmp/kernel-agent-add-rmsnorm --check-only`。此模式每条路径只执行一次，校验后退出。


## 6. 采集小 batch 与大 batch 的 profile

从仓库根目录运行以下命令，将 `GPU-UUID` 替换成空闲 RTX 5090 的 UUID。脚本先在 profiler 外输出六个 batch 的真实延迟，再分别采集 batch 1 和 14509 下的三个 kernel。每次采集只运行指定 batch，不进入计时循环。每次运行写入一个新目录。

```bash
bash experiments/add_rmsnorm/profile.sh GPU-UUID
```

输出目录中，`timings.csv` 是 CUDA event 测量结果，`batch-1.ncu-rep` 和 `batch-14509.ncu-rep` 是原始报告，对应的 `.txt` 是终端可读版本。比较大 batch 的 L2 和 DRAM 读写时，把拆分路径的 Add 和 RMSNorm 的值加起来，再与融合 kernel 比较。独立 kernel 的 replay 和缓存设置会影响实际计数，报告用于分析资源和访存，整体加速比取自 CSV。
