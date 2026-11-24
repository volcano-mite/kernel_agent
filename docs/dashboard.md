# Kernel Agent 数据看板开发交接

看板供宿主机操作者观察一次 Kernel Agent 实验。首版用一页展示 Agent 活动、公开评测反馈、候选性能和最终验收。看板开发用合成回放和 CPU Mock 联调，GPU 评测和 B/D/F 对照由另一条开发线负责。

**本次交付是一个按 run 打开的只读网页。** 操作者中途打开页面，应能读到 Codex 之前说过什么、执行过哪些命令，并在会话仍在运行时看到下一条消息或工具返回。操作者随后可以选中一次评测，查看失败原因或各 batch 延迟，最后确认宿主机交付了哪份候选源码。

当前仓库已有实时日志、终端监看脚本和合成回放数据。浏览器页面、下文的快照接口和 SSE 事件流属于待开发部分。可以用同一个 run 的终端输出核对网页。在仓库根目录执行以下命令，将 `YOUR_RUN_ID` 换为宿主机打印的 ID。

```bash
RUN_DIR="results/YOUR_RUN_ID"
sh scripts/watch_run.sh --once "$RUN_DIR"  # 最近活动、公开评测、Agent 汇总和验收
sh scripts/watch_run.sh --agent "$RUN_DIR" # 从头显示交互，运行中的会话继续跟随
```

## 从无卡环境开工

仓库附带一份 [RTX 5090 实测记录](../results/recorded/e8c125d47fb84e75b7b61d602d0d79c3)，克隆后即可从仓库根目录运行：

```bash
RUN_DIR=results/recorded/e8c125d47fb84e75b7b61d602d0d79c3
sh scripts/watch_run.sh --once "$RUN_DIR"
sh scripts/watch_run.sh --agent "$RUN_DIR"
```

这份只读副本包含 Codex 交互、4 次公开评测、3 份候选源码和独立验收明细。留出加速比为 6.98×，保守加速比为 6.86×。导出时保留原始测量值，移除了认证字段、用例种子、宿主机路径及工具请求中重复的完整源码。完整运行目录由宿主机保留。

用 [合成回放数据](../tests/fixtures/dashboard/replay.json) 搭第一版界面。它覆盖启动、等待评测、正确性失败、修复、同一候选复测和通过验收。快照是回放起点，按 `seq` 应用后续事件就能得到每个时间点的状态。样例中的 ID、误差和测量值均为合成数据。

按 [README](../README.md#run-the-cpu-workflow) 准备宿主机 Python 环境。在仓库根目录创建一个实际的 CPU Mock run。

```bash
RUN_ID=$(python -m harness init --backend mock | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
printf '%s\n' "results/$RUN_ID"
```

另开一个终端，运行 `sh scripts/watch_run.sh results/YOUR_RUN_ID`，把 `YOUR_RUN_ID` 换成刚打印的 ID。原终端依次提交起始样例和正确样例。

```bash
python -m harness evaluate "$RUN_ID" "results/$RUN_ID/workspace/rmsnorm.cu"
python -m harness evaluate "$RUN_ID" tasks/rmsnorm_golden.cu
```

终端依次新增 `wrong_answer` 和 `passed` 两条记录。CPU Mock 的用途是验证状态更新和反馈展示。耗时曲线和 FlashInfer 基线可以用合成回放样例先实现。

## 页面和现有数据来源

顶部显示 run、Agent、模型、B/D/F 变体、Worker 模式和评测预算。左侧时间线按事件顺序显示运行进度。选中某次评测后，右侧保留它的反馈和逐 batch 数据，新事件继续进入时间线。留出验收完成后可查看最终决定和三组延迟。

| 需要显示的信息 | 当前宿主机数据来源 |
| --- | --- |
| Run 配置 | `run.json` 中的任务、Agent、模型、后端、变体、GPU UUID 和预算 |
| Agent 状态、耗时和用量 | `agent/result.json` 在会话退出后写入。Runner 在运行中将完整事件脱敏后追加到 `agent/agent.stdout.jsonl`，终端用 `watch_run.sh --agent` 跟随交互 |
| 评测状态和结果 | `evaluations/<evaluation_id>.json` 先写 `pending`，随后更新为终态，包含阶段、错误摘要和逐用例结果 |
| D/F 诊断 | `diagnostics/*.json` 中的 sanitizer 或 profiler 记录 |
| 候选源码 | `candidates/<candidate_id>.cu` 中按源码哈希存放的快照 |
| 最终决定和留出数据 | `acceptance/decision.json` 和实际复测的 `acceptance/<candidate_id>.json` |

评测记录用 `evaluation_id` 区分请求，用 `candidate_id` 关联源码。同一源码复测时保留每次的反馈。公开榜采用每份 GPU 候选首次通过的得分，得分是相对 PyTorch eager 的几何平均加速比。`pending` 表示等待最终反馈，公开评测记录暂时只提供这一种进行中状态。界面将缺失的延迟、用量或 Agent 状态显示为未知。

## 实时 Agent 交互

页面默认跟随最新活动。点选较早的一条命令或评测时，详情停在所选记录，新事件继续追加；点「跟随最新」回到末尾。命令输出支持展开和换行。工具执行期间显示工具名及等待状态，收到完成事件后再显示反馈。Agent 退出后保留整段交互，并补上耗时和已上报的 token 用量。

服务端可以增量读取 `agent/agent.stdout.jsonl`。文件尚未创建时显示「等待 Agent 活动」，读到暂时的文件末尾时继续等候。按完整 JSON 行推进读取位置；若最后一行尚未写完，保留到下次读取。Codex 命令的输出随完成事件一起给出。终端已有字段映射，网页可以用同一段输入核对文字、事件顺序和工具结果。[Runner 日志实现](../runners/agent.py) · [终端事件映射](../scripts/agent_activity.jq)

| Codex 原始事件 | 操作者应看到的内容 |
| --- | --- |
| `item.completed` + `agent_message` | 显示 `item.text`，保留消息顺序 |
| `item.started` / `item.completed` + `command_execution` | 开始时显示命令；完成后按同一 `item.id` 补上退出码和输出 |
| `item.started` / `item.completed` + `mcp_tool_call` | 开始时显示工具名；完成后显示公开反馈中的状态、阶段、分数或错误摘要。多次调用同一工具时按 `item.id` 配对 |
| `error` / `turn.failed` | 在发生位置显示错误摘要，保留前面的活动。会话最终状态读取 `agent/result.json` |

建议在已有 `agent_activity.payload.summary` 上增量扩展字段。每条活动保留可独立显示的 `summary`，再按需携带 `kind`、`phase`、`item_id` 和对应详情。下面是工具返回的数据示例。已有回放只有 `summary` 时也能显示。

```json
{
  "type": "agent_activity",
  "payload": {
    "summary": "evaluate_kernel 返回 wrong_answer，数值误差超过容差",
    "kind": "mcp_tool_call",
    "phase": "completed",
    "item_id": "item_7",
    "tool": "evaluate_kernel",
    "status": "wrong_answer",
    "stage": "correctness",
    "detail": "tolerance"
  }
}
```

`item_id` 用来配对 Agent 的开始和结束事件，`evaluation_id` 用来区分 Harness 的提交记录，`candidate_id` 指向源码快照。工具返回提供候选 ID 时可以链接到该候选的评测列表；同一源码复测时仍能分别查看两次结果。当前终端也支持提取 Claude 的消息、工具名和工具返回状态，可沿用相同显示字段。

## 拟定的接口和实时接入

首个 PR 可用回放数据完成页面，再用 CPU Mock 的 run 目录验证读取。建议将页面和只读服务放在独立的 `dashboard/` 模块。在接入实时事件的 PR 中复用现有 `RunStore` 和评测结果，浏览器使用宿主机的独立入口。

- `GET /api/runs/{run_id}` 返回当前快照，字段参照样例的 `snapshot`。快照记录已经应用到的 `as_of_seq`。
- `GET /api/runs/{run_id}/events?after=<seq>` 用 SSE 返回更晚的事件。每条事件包含 `run_id`、单调递增的 `seq`、UTC 时间 `at`、`type` 和 `payload`。SSE 的 `id` 使用 `seq`，重连时优先使用 `Last-Event-ID` 续读。
- 宿主机可按 run 追加 `events.jsonl`。用独立短锁分配序号并写入事件，快照和续读边界使用同一序号。浏览器按序号去重，评测按 `evaluation_id` 更新，重连后恢复已选中的记录。

中途打开时需要补齐活动历史。回放样例里的 `snapshot.agent_activity` 只保存最新摘要，建议网页快照另带截至 `as_of_seq` 的有序活动列表，再从该序号开始订阅。静态回放和实时订阅共用同一套事件处理逻辑。旧 run 可以从已归档的 JSONL 恢复 Agent 交互顺序；缺少原始时间戳时，界面只展示顺序。

| 事件 | 发布点和状态变化 |
| --- | --- |
| `run_started` | `harness` 启动会话，run 进入 `running` |
| `agent_activity` | Runner 取得一条可公开的结构化活动，更新 Agent 摘要 |
| `evaluation_started` | Evaluator 写入 `pending`，按 `evaluation_id` 新增一条进行中记录 |
| `evaluation_finished` | Evaluator 完成落盘并计算排名，更新该次记录和当前最佳候选 |
| `agent_finished` | Runner 退出，写入会话状态和已上报的用量 |
| `acceptance_started` | Harness 冻结入围候选，run 进入 `accepting` |
| `run_finished` | 收尾完成或异常退出，更新为 `finished` 或 `failed`。验收决定可随事件传入 |

接入时可从 [CLI 生命周期](../harness/__main__.py)、[评测记录](../harness/evaluator.py) 和 [Runner 日志](../runners/agent.py) 找发布点。浏览器的实时 Agent 活动可从已有的结构化日志提取字段白名单内的摘要。首版用已有的 `pending` 和终态即可完成评测更新。显示 GPU 排队或编译等细分阶段，需要对应的 Worker 或调度事件。

看板接口只返回字段白名单。令牌、环境变量、私有用例种子和原始 Agent 日志由宿主机保留。反馈摘要按纯文本渲染。留出延迟在宿主机验收完成后供操作者查看，MCP 公开评测继续使用现有协议。

Agent 的命令、输出和报错也走上述投影。命令输出使用有长度上限的纯文本预览，发生截断时给出提示。工具请求中的 `arguments.source` 可达整份 CUDA 源码，时间线仅展示工具名及反馈字段。对 GPU run，`agent_finished` 和最终验收分别更新：Agent 超时后 Harness 仍可能验收入围候选；`accepted` 或 `fallback` 以 `acceptance/decision.json` 为准。

## 合并前验收

- 用合成回放展示完整时间线、错误详情和最终验收。重复评测显示两条记录，并关联同一候选。
- 用会输出一条消息后等待放行的测试 Agent 验证浏览器。保持 Agent 运行且 `agent/result.json` 尚未生成时，已打开的页面应显示该消息；此时再打开第二个页面，也能看到同一历史消息。放行后两个页面各收到一次完成事件。宿主机已有同类[实时日志测试](../tests/test_live_agent.py)，可直接复用测试思路，无需 API key 或 GPU。
- 用一对命令开始／完成事件和两次交错的同名工具调用验证配对。退出码、命令输出与工具反馈应落到各自的详情；含凭据的测试字符串和完整工具请求源码应从 API 响应中排除。
- 用 CPU Mock 的真实目录核对等待、失败和通过状态。缺失的测速和用量显示为未知。
- 实时接入后，中途刷新或断开重连，确认已有反馈恢复且后续反馈能继续追加。浏览器操作只读取记录。
- 用一次 GPU run 核对公开排名、最终决定和逐 batch 延迟。性能数据与同一 run 的验收 JSON 一致。
