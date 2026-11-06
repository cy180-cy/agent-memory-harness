# FullHistory / BudgetWindow 单任务对比入口（Phase1-03）

> 本文保留对应阶段的设计、配置和结果记录；“本轮/新配置”指该历史阶段。当前配置与可复跑入口见 [三策略统一评测](memory_evaluation.md)。命令从仓库根目录运行，公共运行时已整理到 `validation/runtime_support/`，不依赖本机历史运行目录。

本入口准备“全历史参考组与受输入预算约束的滑窗组之间的工程取舍对比”。A、B 都是已有 `memory_agent`，分别选择 `full_history` 和 `budget_window`。FullHistory 没有输入裁剪目标，因此这不是“相同输入预算下的能力比较”。

本阶段只有离线准备、历史输入校准、确定性测试及 Mock 执行，没有新增真实 API 请求。旧 train34 的真实成功记录不计入新对比结果。

## 代码与复用边界

- `validation/run_memory_comparison.py`：默认准备，显式 `--mock` 或 `--tests`；真实入口先检查新的实验身份及收费授权。
- `validation/memory_comparison/spec.py`：实验身份、严格双组配置校验、不可重置的请求账本和一次启动锁。
- `validation/memory_comparison/calibration.py`：只用历史 Agent 可见前缀校准输入预算。
- `validation/memory_comparison/execution.py`：顺序构造、记录、调用上游 runner、资源停止分类与结果汇总。
- `tests/test_memory_comparison.py`：定向离线验证。

当前模型调用复用 `validation/runtime_support/` 中的 `qianwen_adapter`、`budget_bindings`、`budget_transport`、`output_budget`、`compatibility` 和 `offline_safety`。这些模块从原生效版本整理而来，没有重写客户端、Agent 循环、Retail 工具或评分器。旧时间戳目录原样保留在本机，但不再是正式入口的依赖。

构造链为：上游 `build_environment` → 注册的 `memory_agent` factory（显式策略及输入配置）→ 上游 `build_user` → 上游 `Orchestrator` → `runner.simulation.run_simulation`。使用上游低层构造函数，是因为现有高层 `build_agent` 没有 Memory 配置参数；没有为此改写共享 runner 或扩展全局配置。

每组重新加载 Task、构造 Retail 环境/DB、Agent、User 及各自状态。只在新 DB 中执行工具。普通 Agent、Memory 源码、UserSimulator、官方评分器、政策和原始数据均未修改。

## 输入校准与证据类型

本阶段原始来源为历史运行 `20261103T1022124442166+0800`，当时根据账本、轨迹、运行指纹与配置的内容及 SHA256 关联，未按修改时间猜测。正式仓库使用 `tests/fixtures/memory/` 中保留来源说明的最小脱敏样本；原始完整账本、授权和响应目录不进入 Git。

历史账本保存了 Agent 请求的消息、工具 schema、tool_choice 和模型等输入字段，属于 **captured**。原脚本按字段白名单保存参数，不能称为完整原始 HTTP 报文。

为与 Phase1-02 完全相同的计数边界对接，从 captured 消息恢复 typed history，再用 `MemoryAgent._model_input_view` 生成 **reconstructed** 视图。它与轨迹中相应 Agent 响应之前的前缀在相同 `to_litellm_messages` 边界严格比对；不包含当前响应、未来消息、隐藏情境、金标或目标 DB 状态。原始 HTTP 输入估算另外保存在 JSON，不能与生成边界的对象直接当作同一个结构比较。

`calibration.json` 每行同时标注 `input_source=captured`、`count_view_source=reconstructed`，保存历史请求序号、轨迹前缀区间、视图指纹、完整/固定/最低集合估算、可移除块、裁剪结果和工具配对诊断。完整分块位置及每个候选重计数也保留。

继续使用 `complete-json-bpe-surrogate` v1：本机 LiteLLM 1.82.6 已安装的 `cl100k_base`，tiktoken 0.14.0。一次编码完整 JSON 视图，包含 system/政策、工具 schema、角色、调用 ID、参数和返回。固定开销用空历史的完整视图计算，不将组件估算相加冒充完整请求计数。这仍不是 Qwen 原生分词或供应商计费数据。

### 固定预算选择

1. 独立余量固定为 128，是本地保守约定，未通过供应商原生计数校准。
2. 安全下界为所有已检查前缀的最大最低集合估算加余量。
3. 裁剪上界为可移除旧块的前缀中，完整估算加余量再减 1 的最大值。
4. 若存在交集，先将安全下界向上取整到 128 的倍数；若取整会越过裁剪上界，则使用精确下界。没有交集时不强压预算，报告不适合检验裁剪差异。

本次实际结果：11 个 Agent 生成位置，固定输入估算 4273；最大最低集合 6975，安全下界 7103，裁剪上界 8562。选择 **max_input_tokens=7168、safety_margin_tokens=128**。历史 Agent 位置 8、9、10、11 触发裁剪，各最低集合均适配。完整表在运行目录的 `calibration_table.md`。

该选择在未来真实运行前冻结，不看新得分调参，不依赖订单或特定回答。已知前缀能适配不保证新对话也能适配；旧事实仍可能随窗口移出。

## 冻结的两组条件

| 条件 | A | B |
|---|---|---|
| Agent 类型 | memory_agent | memory_agent |
| Memory | full_history v1，不裁剪 | budget_window v1 |
| 输入预算/余量 | 无裁剪目标 | 7168 / 128 |
| 任务 | Retail train34 × 1 | 相同 |
| Agent / User | openai/qwen3.7-plus / openai/qwen3.8-flash | 相同 |
| Base URL | https://maas.qianwenaiapi.com/compatible-mode/v1 | 相同 |
| total-output-v2 | Agent 1024 / User 512，max_completion_tokens | 相同 |
| temperature / stream | 0 / false | 相同 |
| 请求上限 | 每组两角色合计 20 | 相同；整个对比最多 40 |
| 步数 / 任务超时 | 30 / 300 秒 | 相同 |
| HTTP 超时 | connect/read/write/pool 各 30 秒 | 相同 |
| SDK / HTTP / 任务重试 | 0 / 0 / 0 | 相同 |
| 评测 | 官方 ALL、DB basis、strict replay | 相同 |
| Judge / Review | 禁止 | 禁止 |

上游给两个角色设置状态 seed=42；原适配器没有将该 seed 参数发送给供应商，本入口保持原规则并明确记录，不悄悄改变参数。相同 seed 不保证真实两组对话相同，也不强制重放旧用户回答。UserSimulator persona 为 None，指南内容另做指纹。

不发送 `max_tokens`，不更换思考模式、模型或输出参数，不做旧参数回退。total-output-v2 的供应商实际执行情况仍未真实验证；本地响应后暂停不能撤销已产生的 Token 和费用。

实验身份不仅是 HEAD。`comparison_plan.json` 同时冻结：当前已跟踪补丁摘要、全部 tau2 Python 源码（包含未跟踪 Memory 源码）、当前执行封装、稳定目录中的公共运行时、依赖版本、Task、政策、工具、Retail 数据和 User 指南指纹。启动前逐项校验内容。

双组配置删除 `group`、`run_id` 和 `memory` 后必须全量相等；模型、输出预算、政策或评分等任何非策略差异都抛错，不只警告。关键生效条件还与当前适配器及实际环境校验。

## 请求与停止边界

每次请求沿用原 CountedTransport 的 SDK/HTTP 检查和发送前预留，先持久化本组账本与全局账本，再允许 I/O。失败、超时和不确定尝试不返还配额。重开账本会读取现有次数，不清零；一次启动锁及固定的 live 执行目录阻止重复启动。不是用 max_steps 代替请求上限。

上游会在等待工具结果时延后步骤终止检查。本入口给原 `step` 加调用前资源检查，达到 30 步或任务期限后不再执行下一步/新工具，不改写原循环。300 秒是任务期限配置及执行边界检查；已开始的 HTTP 仍受各阶段 30 秒超时约束，不能宣称绝对墙钟 300 秒时刻强制中断。

分类分别保留：

- 正常终止且官方 DB 不匹配：业务失败，保留实际官方分数，不自动重跑。
- BudgetWindow 最低集合不适配、计数未知、输出超额、步骤/时间/请求上限：资源停止，保持未评分，不伪造业务 reward0。
- FullHistory 的已知上下文超限错误码：资源停止。当前已修复门控会阻止第二组启动；不能沿用本阶段早期的继续执行规则。无法识别的 HTTP 错误不猜成上下文问题。
- 认证/权限、网络超时、协议或共享执行故障：暂停第二组，保留已有轨迹和账本。
- usage 无法验证：保留 unknown，并保守暂停后续收费链路。
- 金标重放等评分阶段警告：保留原官方返回值并标记需复核，不能以高分宣称有效通过。

资源停止不运行新评分来补零。没有自动第三次运行，也不为获得成功自动重试。

## 记录与离线执行

默认只准备并检查构造，不运行对话：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py
```

执行同一入口的离线 Mock：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --mock
```

定向验证及原有 Memory 回归：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests
```

离线入口先以标准环境变量白名单启动隔离子进程，不读取/传递 `AMH_QIANWEN_API_KEY`，不改调用方环境或用户级持久配置；随后复用原网络、模型及文件防护。全部响应只由 MockTransport 提供，不连接平台或本地代理。

每次写到新的 `validation/phase1_comparison/<run_id>/`，不覆盖旧材料。保存校准表、冻结计划、未批准授权草案、源码指纹及构造检查。Mock 的 A/B 独立子目录另保存：生效配置、请求账本、原始模型响应、完整轨迹/Agent 状态、模型可见上下文及其原始位置关系、工具事件、官方评分原值、warnings、usage、资源决定、耗时和汇总；全局账本保存每次尝试归属及共享暂停原因。

Mock 响应是使用历史内容生成的 **synthetic** 固定响应，usage 改为明显固定的 Mock 值，原历史响应不变。新结果明确 `mode=mock`、`usage_source=mock`、`quality_comparison_valid=false`。即使固定响应重放得到 Reward=1，也只证明调用、工具和评分链路，不能证明裁剪无损或任务成功率提升。

## 下一轮授权与费用

下一轮待批准范围是 A、B 各一次、顺序执行、每组最多 20 请求、合计最多 40 请求；不沿用此前已消费的 train34 或两次接口授权。

本阶段只生成 `authorization.pending.json`：`approved=false`、金额为 null，不是有效收费授权。真实入口必须先核对本次 comparison_id、完整身份摘要、新收费金额目标和请求上限，再允许读取进程凭据及构造真实传输。缺少、旧、未批准或身份不匹配的授权都在读取 Key 之前拒绝；源码变化也使旧草案失效。

凭据仍由用户私有终端继承已有本地配置，未来获批时只传给对应子进程，结束后释放临时引用；不需要重新保存或在聊天提供 Key。本轮没有读取它。

平台可信单价和真实账单仍为 **unknown**，本轮不查询。不能按其他平台价格估价，也不能声称金额已被硬限制在 2 元。未来金额目标必须单独批准；即使每组最多 20 请求，输入 Token、供应商输出限制执行及实际计费仍存在未知。两组各一次仅支持小样本工程观察，不支持稳定成功率或可靠性结论。
