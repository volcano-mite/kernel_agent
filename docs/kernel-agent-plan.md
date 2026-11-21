# GPU Kernel Agent 开发文档

Ryan Hou、Sheng Wei · EE P 500 D · Autumn 2026

## 目标与范围

用 Claude Code 驱动一个 CUDA C++ 开发循环。Agent 读取 FlashInfer-Bench 的算子定义、PyTorch reference 和公开工作负载，从空的候选目录开始，迭代编译、验算和测速。每次运行固定使用一张 NVIDIA GPU，并保存尝试记录与最佳候选。正式实验统一使用 FlashInfer-Bench，先完成 RMSNorm，再接入 GEMM 和 GQA 分页解码注意力。MoE 是扩展目标。

项目交付一次可回放的 Agent 演示、一张比较 B、D、F 的结果表、一份含失败分类的短报告，以及可复跑的脚本和运行记录。

## 仓库起点

| 已有入口 | 用途 |
| --- | --- |
| [README](../README.md) | 安装环境、下载固定版本的 FlashInfer-Trace 数据、配置单卡运行 |
| [RMSNorm smoke 脚本](../benchmarks/rmsnorm/smoke_rmsnorm.py) | 把固定的 CUDA 样例编译为 PyTorch 扩展，评测 `rmsnorm_h4096` 的 14 个官方工作负载，可对一个工作负载采集诊断 |
| [Smoke 候选](../benchmarks/rmsnorm/rmsnorm.cu) | 验证评测器的成功路径，也可用于早期性能排查 |
| [Add + RMSNorm 微基准](../benchmarks/add_rmsnorm/README.md) | 练习采集和解读 Nsight Compute profile，观察融合前后的访存与延迟 |

首个联调任务是普通 RMSNorm（`rmsnorm_h4096`）。独立微基准和 FlashInfer 头文件里的 `FusedAddRMSNormKernel` 对应 Add + RMSNorm，包含残差相加。普通 RMSNorm 的库基线应调用与其算子契约匹配的 FlashInfer 实现。正式 Agent 运行时，smoke 候选放在 Agent 可读范围之外。

已有环境可在有可用 GPU 的宿主机上执行以下命令。按根目录 README 先完成依赖和数据安装。

```bash
source benchmarks/gpu_env.sh
nvidia-smi -L
CUDA_VISIBLE_DEVICES=GPU-UUID python benchmarks/rmsnorm/smoke_rmsnorm.py --diagnostics
```

## 两人分工

| 负责人 | 负责实现 | 交接产物 |
| --- | --- | --- |
| **Ryan · Agent 与诊断** | 管理每次运行的 git worktree 和预算。接入 Claude Code，保存会话和候选历史，记录工具反馈，支持 B、D、F 三档工具配置。为 D、F 接入 memcheck、racecheck 和 Nsight Compute。 | 接收任务 ID、版本和预算的一键运行入口，带候选源码和诊断日志的运行目录 |
| **Sheng · 评测与实验** | 从 smoke 脚本抽出通用评测入口。固定任务定义、工作负载、参考实现和测速配置。管理隐藏工作负载、三组对照实验的运行顺序和成绩汇总。 | 接收任务 ID 和任意候选目录的一键评测入口，逐工作负载 JSON、正式基线和结果表 |

两人先用同一个 RMSNorm 候选完成交接。Ryan 提交候选目录并读取返回结果，Sheng 用同一入口验证手写和 Agent 生成的候选。演示由 Ryan 从日志中选出可回放的修复和优化案例，Sheng 复跑并核对结果。报告由 Ryan 写 Agent 设计和案例，Sheng 写实验方法和结果，两人一起整理失败类型。

## 评测接口

下面是首轮联调拟采用的接口。第一阶段实现后，把实际命令补到根目录 README。

| 方向 | 约定 |
| --- | --- |
| Agent → 评测器 | 传入任务 ID、候选目录和结果路径。任务配置由评测器持有，指定 FlashInfer-Bench definition、入口符号、编译绑定、工作负载和评测预算。RMSNorm 复用当前样例的 `rmsnorm.cu::run`。 |
| 评测器 → Agent | 返回候选哈希、评测状态、逐工作负载状态与延迟，以及必要的错误摘要。状态至少区分通过、编译失败、运行失败、数值错误和超时。B、D、F 使用同一份基础评测结果。 |
| 评测器 → 实验记录 | 保存原始 FlashInfer-Bench traces、候选与基线延迟、数据版本、GPU UUID、测时配置及工具版本。最终评测从已归档的候选源码重建，并覆盖公开与隐藏工作负载。 |

每次尝试归档源码快照、父候选哈希、修改理由和评测结果。失败或超时后清理相关进程，使下一次尝试可以继续。Agent 根据公开评测选出候选并在预算内提交。Sheng 用冻结的候选跑最终评测。

## 对照实验

| 版本 | Agent 可用资源 |
| --- | --- |
| **B** | `nvcc`、FlashInfer-Bench 的通过状态和延迟反馈，以及编译或运行失败的错误摘要 |
| **D** | B 的资源，加上 Compute Sanitizer 和 Nsight Compute |
| **F** | D 的资源，加上固定版本的 FlashInfer CUDA 示例 |

三组共用同一模型、推理设置、基础提示词和任务预算。版本差异由提示词的工具清单和运行环境共同实现。三个版本都可读 PyTorch reference。FlashInfer 的 CUDA 示例仅提供给 F。开发机的 `.venv` 已包含 FlashInfer CUDA 头文件，因此 B、D 的 Agent 进程需要与该环境隔离，由评测器在自己的环境中调用依赖。正式运行前验证对应版本的工具和文件可见性。候选目录对 Agent 可写，公开任务文件和评测入口对 Agent 只读，隐藏工作负载保存在评测器一侧。

每个任务运行 B、D、F 各两次，记录随机种子和运行顺序。正式测速在同一张空闲 GPU 上串行执行，固定编译参数、预热、迭代次数和重复次数。Profiler 输出用来解释瓶颈，延迟使用 FlashInfer-Bench 的测量结果。工作负载、预算、测速参数和汇总规则在正式运行前固定。

- **正确性与速度**。按任务容差逐工作负载对比 PyTorch reference，并检查 NaN/Inf。建议用各工作负载的 `reference 延迟 / 候选延迟` 等权几何平均衡量总体加速。预算内提交的候选通过公开及隐藏工作负载，且总体加速大于 1×，记为成功。另列相对 FlashInfer 库 kernel 的加速和提交中的正确性失败比例。
- **成本**。记录 token 用量、端到端耗时、实际占用的 GPU 时间、首次得到正确候选的时间和每个成功 kernel 的成本。
- **安全性**。记录修改评测或 reference 文件的尝试，以及执行注释中绕过评测指令的情况。保留相应日志供报告引用。

每组只有两次重复。结果表同时列出成功次数与总次数，保留逐次结果供检查。

## 实施顺序与验收

| 阶段 | Ryan | Sheng | 完成条件 |
| --- | --- | --- | --- |
| **1. 跑通 RMSNorm** | 从空候选目录启动一次 Agent，生成 CUDA 候选并把结果接回日志 | 泛化 smoke 脚本，让它接受候选目录，验证已有样例、故意写错的候选和超时后的恢复 | 已有样例通过 14 个官方 workload。Agent 候选可经同一入口评测，日志可定位每次提交 |
| **2. 跑通三种版本** | 接上诊断工具和 F 的示例，验证工具与文件可见性 | 冻结 RMSNorm 的公开和隐藏划分，测 PyTorch 与 FlashInfer 库基线，验证结果汇总 | B、D、F 各完成一次试跑，保留结果和中间日志 |
| **3. 执行正式实验** | 用同一运行入口接入 GEMM 与 GQA 分页解码注意力，记录运行预算与失败 | 挑选实际 benchmark 定义、验证库基线，按随机顺序运行并汇总结果 | 每个纳入的任务产生 6 份独立记录，最佳候选可复跑 |
| **4. 第 10 周演示和报告** | 回放从空目录到正确实现，以及后续修复或提速 | 独立复验演示候选，整理结果表和失败分类 | 交付脚本、运行记录、结果表和报告 |
