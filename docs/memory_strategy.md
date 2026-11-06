# MemoryStrategy、FullHistory 与 BudgetWindow

Phase1-01 新增任务内记忆接入点。`FullHistory` 复现普通 `LLMAgent` 的完整历史输入，是策略对照基线。Phase1-02 新增输入预算窗口 `BudgetWindow`；没有进行真实成功率、Token 费用或成本对比实验。

## 复用与最小接入

- 复用上游 `LLMAgent` 的输入检查、用户/工具消息追加、生成调用和 Assistant 消息追加。
- 复用 `Environment.get_response`、`ToolCall`、`ToolMessage`、`MultiToolMessage`、factory/registry 和 runner 构造入口。
- `llm_agent.py` 只把原来的 `state.system_messages + state.messages` 提取为 `_build_model_context`；默认实现完全相同。
- `registry.py` 只新增 `memory_agent` factory 的导入和注册。默认 `llm_agent` 保持原样。
- 不修改 UserSimulator、Retail 数据/工具/政策、评分器、全局模型默认值或依赖。

## 策略接口和状态

`src/tau2/agent/memory.py` 提供三个真实调用的方法：

- `initialize(history) -> MemoryState`：创建独立状态；再次调用即重置为新状态。
- `observe(state, message)`：接收新出现的 Agent 可见对话事件。
- `build_context(state)`：返回下一次生成使用的对话上下文。

`MemoryAgentState` 保留继承的完整 `messages`，另有独立的 `memory`、`memory_cursor`、策略名称和版本。策略实例不持有共享会话列表。

调用顺序为：

1. `get_init_state` 复制初始历史并初始化策略状态。
2. 原 LLMAgent 追加用户消息，或按原顺序展开多个工具结果。
3. 上下文 hook 用 cursor 把尚未接收的可见消息交给策略，再读取策略上下文。
4. Agent 在策略上下文前加入自己的系统政策，沿原 `generate` 路径生成。
5. 原 LLMAgent 追加 Assistant 消息，MemoryAgent 随即接收这一新事件。

构造历史、策略输入和返回上下文均做深复制，避免调用方、原始轨迹、策略状态之间通过可变对象互相污染。系统政策和工具定义由 Agent 管理，不由策略改写；BudgetWindow 仅通过计数回调取得它们的副本。factory 不读取或转交 runner 的 `task` 参数。策略元数据不进入模型提示词。

## 选择与扩展

通过既有 `build_agent("memory_agent", environment, llm=..., llm_args=...)` 选择新 Agent，默认使用 `full_history`。注册名也可用于已有运行配置的 `agent` 字段。本轮没有新增 CLI 参数。

Python factory/构造器支持显式 `memory_strategy="full_history"` 或实现 `MemoryStrategy` 的对象；未知名称会在生成之前失败。不要把策略配置塞进 `llm_args`，该用法会明确报错。

BudgetWindow 在同一 `build_context` 扩展点接入，原始 `state.messages` 和 `state.memory.messages` 继续保留完整历史。输入上下文预算与 `max_completion_tokens` 总输出预算是两个职责。

## Phase1-01 独立对照和历史验证

在修改上游文件之前，冻结 HEAD `fc0055dc4e0a316c3f83133267fbd6faaa770992` 下的普通 LLMAgent 实际经过 LiteLLM/HTTP MockTransport 生成了 6 个场景、10 个请求。

对照位于 `tests/fixtures/memory/full_history_fc0055dc.json`，记录源 Git blob、依赖版本和场景源哈希；测试另固定快照 SHA256。捕获脚本在 LLMAgent 偏离固定上游来源 commit 后拒绝重新捕获，不会自动更新预期文件。

修改后的普通 Agent 与 FullHistory Agent 都逐次对照该快照，比较初始化、所有生成参数、完整 HTTP JSON、工具定义、tool_choice、调用 ID、消息顺序、最终完整历史、请求数及工具执行顺序。不是只让两条新路径彼此比较。

确定性测试还覆盖交错状态、reset、新 trial、可变对象隔离、策略方法实际生效、非法配置、隐藏 Task 信息不影响可见输入，以及现有 total-output-v2 的超额/未知 usage 暂停。资源暂停保持未评分状态，不伪造成业务成功。

离线复跑命令（仓库根目录、项目虚拟环境，不需要 Key）：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_tests.py
```

该入口先安装网络、凭据文件读取和目录写入防护，再导入模型库；禁用真实生成出口，只允许已有 total-output-v2 的 MockTransport。输出保存到新的 `validation/phase1_memory/<run_id>/`。

Phase1-01 当时实际结果：31 项通过、0 失败、0 跳过；验证进程使用 26 次 HTTP Mock，加上捕获对照的 10 次，合计 36 次 Mock、0 次真实 API。

这些离线测试没有验证真实供应商对 total-output-v2 的执行，也没有运行真实 Retail 任务。现有环境未安装 ruff，未安装依赖补跑 lint；`git diff --check` 通过。测试有上游 audioop 弃用提示，以及安全禁用自动插件加载后既有 asyncio 配置项的提示，不影响这组同步测试结果。

## Phase1-02 配置与调用链

正式新增模块：

- `src/tau2/agent/budget_window.py`：全历史工具配对检查、用户轮次分块、完整候选请求重计数、连续后缀选择、暂停异常。
- `src/tau2/agent/input_tokens.py`：可注入计数契约和本地完整 JSON 的 BPE 估算器。
- `memory_agent.py`：新增 `memory_config`，构造策略；上下文 hook 提供固定输入计数回调并把诊断写入本会话 `context_selections`。

没有再修改 `llm_agent.py` 或 `registry.py`；保留 Phase1-01 扩展点和注册。普通 Agent、默认 FullHistory、UserSimulator、工具执行及评分流程保持原样。

通过既有注册 factory 显式配置（以下只构造，不执行生成）：

```python
from tau2.registry import registry

factory = registry.get_agent_factory("memory_agent")
agent = factory(
    tools=environment.get_tools(),
    domain_policy=environment.get_policy(),
    llm=model_name,
    llm_args=model_args,  # 两种策略使用同一生成配置，包括既有总输出预算
    memory_strategy="budget_window",
    memory_config={"max_input_tokens": 4096, "safety_margin_tokens": 128},
)
```

`max_input_tokens` 必填且为正整数，余量为非负整数；未知选项、未知策略或计数契约错误在生成前明确失败。不可把 `memory_config` 放进 `llm_args`。没有新增 CLI 或共享 runner 参数；目前显式预算从 Python factory/构造器接入，普通 `build_agent("memory_agent", ...)` 仍默认 FullHistory。

调用链：registry factory → MemoryAgent → 原 LLMAgent 追加可见事件 → cursor 同步 MemoryState → BudgetWindow 全历史校验与选择 → Agent 加入原系统消息 → 原 generate → 实际 LiteLLM。已有 total-output-v2 在相同生成出口执行响应后资源检查，没有复制、改写或绕过。

## 分块、保留与停止规则

1. 先校验全部原始历史的工具调用/结果关系，再做裁剪。调用 ID 必须非空且不重复；结果必须且只能关联一次已有待完成调用。单条 Assistant 的全部工具返回和连续工具链均不得拆分。旧历史错误不能因后来被移出窗口而被隐藏。
2. 每个用户消息到下一个用户消息之前为一块。首次用户发言前的 Assistant/工具历史单独作为 `preamble` 块；没有用户消息时全部作为前导块。初始历史与新消息使用同一规则。
3. 最新块和它紧邻的前一块必须保留；前导块恰好位于此前位置时也保留。该规则与“Yes”、任务 ID 或业务内容无关。最新块可仍在进行，但工具调用若有未齐全结果，生成入口抛出 `context_incomplete_tools`，等待调用方补齐；不编造返回。
4. 完整输入超预算时，从最旧的可移除块逐块删除。每次重新序列化整个候选请求并计数，只保留连续后缀。恰好满足 `估算 + 余量 <= 预算` 即停止，不做中间跳选、摘要或字段内截断。
5. 最低集合仍超预算时抛出 `ContextBudgetError(stop_reason="context_budget_exceeded")`，在调用模型之前停止。计数失败、非法返回值或缺少固定输入回调则为 `context_budget_unknown`；非法工具关系为 `context_protocol_error`。

异常携带诊断和 `official_reward=None`，不是业务 reward0。调用方应保存轨迹并结束当前执行，不自动重试或执行未生成的新工具。这里没有改写共享 runner 的异常处理。补齐工具结果后的显式恢复允许重新构建上下文；资源问题不能伪装成任务成功。

策略只改变下一次输入，不删改原始轨迹。每次成功选择或暂停记录预算、余量、计数器/模型、裁剪前后估算、候选计数序列、原始区间、块数/消息数、最新块保留和工具配对结果。区间为从 0 开始、右端不包含的 `[start,end)`。诊断归属具体 `MemoryAgentState.context_selections`，不注入提示词、不保存 Key 或隐藏任务数据。

## 本地计数口径与限制

实际使用已安装 LiteLLM 1.82.6 的 `default_encoding.encoding`（`cl100k_base`，tiktoken 0.14.0），离线使用本机包内已有 BPE 资源。未安装或下载 tokenizer。

计数器 `complete-json-bpe-surrogate` 版本 1 对整个规范 JSON 视图一次编码，包括模型、全部 system/政策、消息角色、内容、调用 ID、参数、工具结果、工具 schema、tool_choice，以及显式 response_format。固定输入只加入一次。候选视图沿用项目 `to_litellm_messages`，因此遵守原上游消息转换；它是生成边界的完整输入估算视图，并非声称等于供应商最终 HTTP/ChatML 字节或原生分词。生成参数和输出预算不转成输入内容，也不把秘密配置字典交给策略。

这不是 Qwen 原生 tokenizer，供应商封装开销和计费 Token 均未校准。余量用于保守控制，不能提供供应商侧硬封顶保证。计数器失败不会返回 0；非文本/工具内容明确暂停，未支持语音、多模态。`InputTokenCounter` 可注入，测试中的固定权重仅为算法边界测试，不是供应商 Token。

## Phase1-02 实际离线验证

先运行节点测试：30 项通过，0 失败，0 跳过。

随后集成验证最终通过 42 项：Phase1-01 原有 31 项回归，加 BudgetWindow 11 项集成（含 6 个大预算逐请求 FullHistory 对照）。使用真实 Agent、消息转换、LiteLLM、HTTP MockTransport 和 `Environment.get_response`；最终集成进程共 52 次 Mock HTTP，真实 API 0 次。

首次集成有 1 项断言错误：把转换前带 `tool_calls=None` 的消息视图与 LiteLLM 已省略该空字段的发送结构直接比较。保留失败产物；修正为独立 FullHistory 对照分别在生成边界和完整 HTTP 边界严格比较，不过滤字段、不更新冻结快照。修正后定向复验通过。

实际裁剪例：输入预算 339，余量 8。两个旧块共 6 条消息 `[0,6)` 整块移出，紧邻前轮 `[6,8)` 和当前轮保留。相同计数器的两次输入由 5377→274、5434→331；第二次 `331+8=339` 恰好满足。旧工具调用与结果一同移出；新工具 `current-id` 恰好执行一次，其返回进入第二次输入，完整原始历史仍保留。

阻止例：输入预算 1，最低集合估算 274，保留两块、不删政策/工具，`context_budget_exceeded`；Mock 请求数 0、工具执行 0、Reward 为未评分。计数失败同样未发送。裁剪路径还验证 total-output-v2 的超额与 unknown 暂停，响应中的工具未执行，后续请求被既有暂停门阻止。

上面的缩减只表示同一离线估算方法下的模型输入变化，不是成功率提升、真实输入 Token 或费用下降。较早事实会随块移出，不保证语义无损；尚未验证长对话业务质量、供应商原生计数或真实 Memory 任务效果。

可复跑命令（不需要 Key，仅运行定向测试）：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_budget_window_tests.py
```

可分别加 `--node-only`、`--integration-only`。每次写入新的 `validation/phase1_budget_window/<run_id>/`；复用原安全防护与预算封装，真实网络出口、未验证生成出口和目录外写入均受阻，使用虚构 Key，不读取用户持久凭据。最终集成结果在 `20261103T143526220595+0800/`；本轮原始基线和增量清单在 `20261103T1428202772264+0800/`。
