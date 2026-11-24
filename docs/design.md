# Kernel Agent 评测反馈系统设计

首版在单张 NVIDIA GPU 上评测 FlashInfer-Bench 的 `rmsnorm_h4096`。每次 run 启动时选定 Claude Code 或 Codex。Agent 修改 CUDA C++ 候选，宿主机 Kernel Harness 负责编译、验算、测速、归档与独立验收。两种 Agent 共用同一套 MCP 工具和评测协议。

## 系统框架图

![Kernel Agent 宿主机评测架构。无 GPU 的 Agent 容器经 MCP 提交候选源码。宿主机按 run 排队，调度 GPU Worker 或 CPU Mock，返回公开评测 JSON。会话结束后，宿主机从冻结快照执行私有留出验收。](figure/kernel-agent-framework-v9.png)

| 组件 | 职责 |
| --- | --- |
| Run 配置与启动 | 创建工作区并写入公开任务规格，选定 Agent、模型、评测后端、实验变体和预算，记录运行轨迹。 |
| Coding Agent | 在无 GPU 的工作区读取任务，修改 `rmsnorm.cu`，经 MCP 提交源码并参考评测反馈继续迭代。 |
| Kernel Harness | 在宿主机提供 MCP 工具，按 run 校验权限与配额，调度 GPU 或 Mock 后端，管理快照、排名和最终验收。 |
| GPU Worker | 每次评测启动独立进程，从标准输入读取评测请求，编译并运行候选，从标准输出返回评测 JSON。 |

启动脚本用 `claude -p` 或 `codex exec` 建立非交互会话。Agent 容器挂载本次工作区，F 变体额外挂载只读示例。Agent 可访问模型服务，并通过带鉴权的 Streamable HTTP 调用工具。评测代码、基线实现、私有用例与完整记录由宿主机管理。 [Codex 非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode) · [Claude Code 非交互模式](https://code.claude.com/docs/en/headless)

## 宿主机 Worker 模式

`harness serve` 接收容器发来的候选。GPU 后端支持 Podman CDI 和 `--gpu-worker host`。宿主机模式由 Bubblewrap 启动 `python -m harness.gpu_worker`，每个子进程处理一个请求。初始选择写入 run 配置，`serve`、宿主机 `evaluate` 和 `accept` 都按该配置启动 Worker。 [评测入口](../harness/__main__.py) · [GPU Worker 启动器](../harness/worker.py)

| Run 选项 | 执行方式 |
| --- | --- |
| `--backend mock` | CPU Mock，复用 MCP 和评测 JSON，用于无卡联调。 |
| `--backend gpu --gpu-worker host` | 宿主机子进程直接访问指定 UUID 的 GPU。 |
| `--backend gpu --gpu-worker podman` | 默认走现有 CDI 路径，由 `--gpu-image` 指定镜像。旧 run 配置按 `podman` 读取。 |

Agent 使用现有 MCP 工具作为客户端，提交完整 `rmsnorm.cu` 后等待同一次调用返回反馈。首版沿用 HTTP，宿主机服务仅监听回环地址，Podman Runner 已将回环 URL 映射到 `host.containers.internal`，并向客户端注入本 run 的 Bearer 令牌。GPU 执行方式由 Harness 选择。 [MCP 服务](../harness/mcp_server.py) · [Agent Runner](../runners/agent.py)

容器到宿主机的 MCP 网络不可用时，可另接共享目录传输。每个 run 额外挂载自己的 `inbox/` 和 `outbox/`，容器写入请求、只读结果。写入方在同一目录先写临时文件，再原子重命名。宿主机拒绝链接和非普通文件，读取限长内容并校验 run、令牌和请求后交给同一调度入口。临时文件、积压和权限由传输层管理。

候选编译后会作为本机 C++ 扩展加载。宿主机启动器通过独立的用户、PID 和网络命名空间、只读运行时挂载及独立临时文件系统限制每个任务。Worker 在命名空间内使用 UID/GID 65534，映射到宿主机的普通权限 Harness 账户。启动器按白名单设置环境变量，当前快照与当前用例通过标准输入传入。Harness 在隔离环境外管理工作区、私有用例清单与 `results/`，并持有按 GPU UUID 建立的设备锁。启动器只开放所选 GPU 和必需的驱动设备，再以 `CUDA_VISIBLE_DEVICES` 将该卡映射为 `cuda:0`。 [宿主机启动器](../harness/host_worker.py)

## 提交候选与返回反馈

基础 MCP 工具为 `evaluate_kernel(run_id, source)`，计划增加可选的 `request_id` 用于重试。提交携带本 run 的令牌和单份 UTF-8 CUDA 源码，上限为 128 KiB。Harness 先归档按内容哈希命名的不可变快照，再将该快照送入 Worker，由 Worker 写到自己的临时构建目录。目录、编译命令和用例均由宿主机确定。

GPU 和 CPU Mock 返回同一格式的公开 JSON，包含 `run_id`、`request_id`、`candidate_id`、`backend`、`status` 和逐用例结果或失败摘要。`queue_wait_s` 记录从入队到调度的时间。GPU 反馈可带上当前排名第一的 `best_candidate_id`，通过正确性验证并完成实测的候选才获得分数。客户端用 `candidate_id` 关联宿主机保留的源码快照。`public_feedback` 按字段白名单生成结果，并对诊断与报错摘要做限长及路径、凭据脱敏。完整日志保存在宿主机 `results/`。 [公开反馈出口](../harness/feedback.py)

### 队列与重试

- `serve` 为每个 run 维护有界 FIFO，由一个消费者按顺序调用 `Evaluator`。新请求在鉴权后检查提交频率、总量与队列容量，并在 run 锁内原子入队和预留一次评测额度。公开评测、诊断和离线验收都经同一宿主机账户持设备锁，同一张卡在任一时刻只执行一个任务。
- 客户端为一次提交生成并保留 `request_id`。服务用 `RunStore` 按 `(run_id, request_id)` 原子保存入队顺序、状态和最终公开结果。相同 ID、相同源码的调用等待已有请求或取回原结果，相同 ID 绑定不同源码时拒绝。旧客户端省略 ID 时由服务生成新 ID。同一候选可用新 ID 主动复测。
- MCP 等待超时或连接断开后，可用原 ID 重新等候。调度时先记录 `running` 和任务标识，再启动 Worker。服务以同一 run 重启时，核实并回收旧任务，将未确认完成的 `running` 请求记为 `infrastructure_error`，再继续处理 `queued`。需要重测失败任务时由客户端创建新 ID。
- `--worker-timeout` 从 Worker 启动起算。Worker 启动失败、失联或达到外层时限且未返回有效评测结果时，记录 `infrastructure_error` 并释放评测额度，提交频率和总量限制仍适用。明确归因于候选的执行超时记为 `timeout`。超时先强制回收本次任务及其子进程，再做设备健康检查，检查通过后恢复调度。

## 四阶段 GPU 评测

固定 FlashInfer-Bench 任务定义。输入和输出为 BF16，参考实现按 FP32 计算，`hidden_size=4096`，`eps=1e-5`，候选入口为 `rmsnorm.cu::run`。公开用例覆盖不同 batch 大小。

1. **静态检查。** 检查源码格式、入口和依赖规则，拒绝调用现成算子实现的候选。
2. **隔离编译。** 在受限子进程运行 `nvcc`，保存相关编译日志及 `ptxas` 报告的寄存器、spill 和静态 shared memory 数据。编译产物的符号检查在这一步完成。
3. **正确性验证。** 在公开用例上检查输出形状、dtype 和数值容差。返回失败用例与误差摘要。
4. **性能测量。** 统一候选与基线的计时范围、输出分配和缓存策略。预热后复用预分配缓冲区，在计时区间外冲刷 L2，再用 CUDA Event 多次采样。保存逐用例延迟中位数和波动，按相对 PyTorch eager 加速比的几何平均值排名。

任一阶段失败即返回对应状态，例如 `rejected`、`compile_error`、`runtime_error`、`wrong_answer` 或 `timeout`。

## B/D/F 实验变体

| 变体 | Agent 可用资源 |
| --- | --- |
| B | 基础 `evaluate_kernel` 工具，返回编译、正确性与测速结果。 |
| D | B 加 `sanitize_kernel` 和 `profile_kernel`。Harness 返回 Compute Sanitizer 的检查状态与错误摘要，以及 Nsight Compute 的硬件计数器摘要。诊断从归档候选发起，正式延迟由基础评测独立测量。 |
| F | D 加只读挂载的跨算子 CUDA 示例。人工筛选示例，排除目标算子的实现。 |

Harness 按 run 执行工具权限。一次 B/D/F 比较固定同一个 Agent、同一个模型、任务、起点和预算，记录模型用量、GPU 评测与诊断耗时。

## 独立验收

会话结束后，宿主机关闭提交入口，先将已接收请求处理到终态，再按公开排名冻结已通过 GPU 评测的入围快照。独立验收由宿主机触发，启动新的 GPU Worker，在预先固定的私有留出用例上验证并复测性能。留出结果只进入验收记录。交付确认正确且加速超过测量波动的候选，否则回退到 PyTorch eager。离线报告逐用例延迟，并加入同算子语义与 `eps` 的 `flashinfer.norm.rmsnorm` 基线。 [FlashInfer Norm API](https://docs.flashinfer.ai/api/norm.html)

## 拟定代码结构

```text
kernel-agent/
├── tasks/
│   └── rmsnorm.py           # 算子接口、PyTorch 参考实现和用例
├── harness/
│   ├── __main__.py           # 初始化 run、服务启动与 Worker 选择
│   ├── evaluator.py         # 规则检查、配额、Worker 调度与验收
│   ├── worker.py            # Worker 调度、设备锁和故障恢复
│   ├── host_worker.py       # 宿主机隔离启动、CUDA 环境和设备挂载
│   ├── gpu_worker.py        # 隔离编译、正确性检查与测速
│   ├── mock_worker.py       # CPU 模拟后端
│   ├── diagnostics.py       # sanitizer 与 ncu 摘要
│   ├── results.py           # 候选快照、请求状态与评测历史
│   ├── feedback.py          # 精简诊断与性能反馈
│   └── mcp_server.py        # MCP 鉴权、接单与反馈
├── runners/                 # Agent 启动、MCP 客户端配置与日志
├── examples/                # F 的跨算子 CUDA 示例
└── results/                 # 每个 run 的源码快照、评测结果和最终交付
```

## 启动与验证

从有 GPU 的 Linux 宿主机启动流程。安装 Bubblewrap，准备 CUDA 版 PyTorch、`nvcc`、C++ 编译器和 Ninja，独立验收还需要 FlashInfer。D/F 变体需要 Compute Sanitizer、Nsight Compute 和相应的计数器权限。用 `nvidia-smi -L` 查询 `GPU_UUID`，将 `GPU_WORKER_PYTHON` 指向 Worker 的 Python 可执行文件，并准备 Agent 镜像、模型和 API 凭据。 [Agent 启动说明](../README.md#run-an-agent-on-a-gpu)

```bash
python -m harness run --agent codex --model "$MODEL" --agent-image kernel-agent:local \
  --backend gpu --gpu-worker host --gpu-uuid "$GPU_UUID" --gpu-worker-python "$GPU_WORKER_PYTHON"
```

联调时可拆开运行。宿主机用 `init` 创建 run，然后用 `serve` 启动 MCP。`RUN_ID` 和 `KERNEL_AGENT_MCP_TOKEN` 分别取 `init` 返回的 `run_id` 和 `token`，令牌通过环境变量提供给 Runner。

```bash
python -m harness init --agent codex --model "$MODEL" --backend gpu \
  --gpu-worker host --gpu-uuid "$GPU_UUID" --gpu-worker-python "$GPU_WORKER_PYTHON"
python -m harness serve "$RUN_ID" --port 8000
```

在宿主机另一终端启动 Codex 容器。Runner 挂载本 run 的工作区，容器内 Codex 通过 MCP 调用 `evaluate_kernel`。会话结束后让 `serve` 停止接单并完成收尾，再由宿主机执行 `python -m harness accept "$RUN_ID"`。

```bash
python -m runners --backend codex --model "$MODEL" --image kernel-agent:local \
  --run-id "$RUN_ID" --mcp-url http://127.0.0.1:8000/mcp --timeout 900 \
  --workspace "results/$RUN_ID/workspace" --log-dir "results/$RUN_ID/agent"
```

连接失败时先检查容器能否访问 `host.containers.internal:8000/mcp`，收到 `401` 时核对 run 令牌。Worker 初始化失败时，在宿主机检查 `nvidia-smi -L` 的 UUID、Worker 账户的设备权限和 CUDA 环境，详细原因查宿主机日志。

## 两人并行开发

一人负责宿主机 Worker 启动、设备锁与故障恢复。另一人使用 CPU Mock 验证 MCP 鉴权、跨 run 拒绝、队列容量、同 ID 重试、服务重启和反馈脱敏。上真卡后，用 starter 验证数值失败，用 golden 验证正确性、测速和留出验收，用调用现成算子的样例验证规则拒绝，再用受控的 Worker 超时验证进程回收和下一次评测。最后用 Codex 跑通「提交 → 失败反馈 → 修改 → GPU 复测」。

## 后续扩展

- 外层进化搜索，维护 Top-K 候选与多条优化分支。
- Agent 后端运行时热插拔。首版通过 run 配置在启动时选择 Claude Code 或 Codex。
- 根据硬件计数器生成 `actionable_hint` 专家提示。
- 通过 `query_idiom` 按瓶颈动态检索跨算子范式。
- 在同一框架中支持 Triton 和 CUDA 候选混跑。
