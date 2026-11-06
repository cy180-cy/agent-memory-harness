# StructuredMemory 第一版

同一个`memory_agent`工厂现可选择`full_history`、`budget_window`和`structured_memory`。
普通LLMAgent、默认FullHistory、BudgetWindow算法及模型输出配置不变。

```python
agent = registry.get_agent_factory("memory_agent")(
    tools, policy, llm=model, llm_args=model_args,
    memory_strategy="structured_memory",
    memory_config={"max_input_tokens": 7168, "safety_margin_tokens": 128},
)
```

本模块不收Task、金标、目标DB或隐藏情境，也不额外调用模型。
统一入口已通过 `--suite` 支持三策略，原固定A/B入口继续保留；详见
[三策略统一评测](memory_evaluation.md)。工厂可选与Mock回放不代表已经完成三组真实评测。

## 状态与来源

`structured_memory.py`新增`StructuredMemory`、`StructuredMemoryState`、`MemoryFact`及
`VisibleCall`。策略继承现有BudgetWindow的配置和计数器接口，单独实现事实维护及上下文选择。
MemoryAgent仅增加策略分支和可序列化的状态类型；没有复制Agent循环。

每个task/trial的MemoryState独立保存完整可见消息、事实版本、工具调用关联及待确认提议。
每条事实携带原始消息下标、角色，以及适用的tool_call_id和工具名。
MemoryAgent原始轨迹不删改，构建上下文返回独立副本。

维护六类记录：目标候选/显式目标、用户原文与结构化事实约束、实体字段、
具体操作确认、工具操作及结果引用、待确认或未解决事项。

## 确定性规则

- 自然语言不做语义猜测：首个请求和后续用户原文保留，标记为未解析表述。
  可见JSON中的`goal`、`facts`、`constraints`按字段记录用户声明。
- 成功ToolMessage按调用ID对应原ToolCall；错误、孤立或重复结果不产生已确认实体字段。
- 实体优先依据返回ID与调用参数中的ID匹配；身份冲突留待核实。
  ID不明确时使用call_id作用域，不将不同查询的结果强行合并。
- 同一实体字段的后续观察覆盖当前值，旧版本标记superseded而保留。
  同一Assistant多工具批次返回冲突值时，两者标记conflict，直到后续独立观察更新。
  字段未再次出现不等于被删除，实体值始终标注为“最后观察”。
- 普通“Yes”不会自动生成权限。第一版仅能可靠识别可见JSON形式的精确确认：
  Assistant的`proposed_action={tool,arguments}`与紧邻User的
  `confirmation={tool,arguments,approved}`完全对应才形成范围明确的记录。
  该格式是可选证据格式，没有要求真实UserSimulator改提示词或采用它。
- 新用户消息使未使用的旧确认转为needs_reconfirmation；只有参数完全匹配的工具调用
  可关联确认，首次返回后记为consumed，不把确认扩展给其他实体或操作。
  缺少可解析确认记录表示未知，不替Agent决定自然语言是否符合业务政策。
- 记录工具已返回、error标志和结果来源；这不等于业务目标已完成，不产生Reward。

例如，两次可见查询分别返回同一个实体：

```json
{"order_id":"O1","status":"pending"}
{"order_id":"O1","status":"cancelled"}
```

更新后，`O1 /status = pending`保留为superseded，`cancelled`为最后观察值；
两者分别引用各自的调用及结果消息。`O2`的status不会被更新。

## 输入预算

完整输入适配预算时直接使用全历史，同时在状态内维护结构化记录。
超限时保留最新两个完整交互块，将较早记录注入一个明确标记的assistant记忆消息。
用户表述、约束、确认及未解决冲突优先保留；实体和工具记录按新近程度及字段深度选择，
最多32条可选记录，省略ID及数量明确写入选择审计；上下文声明省略信息未知，不把未知信息当成不存在。

复用现有完整JSON输入计数器，将system政策、工具定义、记忆消息和近期对话一起计算，
加独立安全余量。必需记录与最低交互集合仍放不下时，抛出既有ContextBudgetError，
不删政策、不拆工具配对、不发送模型请求。输入估算不是千问原生计费Token。

### 紧凑注入格式（Phase1-06.1）

`compact-evidence-v1`只改变选择与序列化，不改变事实维护、用户确认规则或7168/128配置。
模型记忆段保留`source_id`、类别、值、置信类型、版本状态，以及适用的实体作用域和语义字段路径。
完整来源数组、批次、原始消息下标和省略诊断留在`context_selections[].memory_sources`等审计字段；
`source_id`对应该次选取时的完整MemoryFact快照，后续事实更新不会污染过去的来源记录。
superseded版本不注入；未解决conflict保留各个值及其不确定状态。确认状态和精确操作范围始终保留。

只有active实体事实存在同值合并记录，且其匹配工具调用与返回完整处于近期保留块中，才省去记忆段重复项。
没有采用模糊文本去重，也没有把普通“Yes”解释为操作授权。无法可靠解析的旧用户表述继续保留原文，
不会因新的用户话题或推测的“已失效”而删除旧约束。近期两个完整块、政策、工具定义和原始轨迹不变。
若只能删尽所有受支持的旧记忆才能适配，则停止，不返回空记忆段冒充结构化记忆。

原train34阻塞前缀有194条状态事实，最终旧候选实际只有一条241字符的初始用户原文；不是194条全部注入。
同一完整请求边界的计数：固定输入4273，固定输入与近期块6975，旧记忆候选7174，紧凑候选7109。
这些是分别重计的完整候选，不是独立分项Token相加。7109+128=7237仍大于7168，保持context_budget_exceeded。
不带说明与来源的裸原文对照已达7038；该对照不作为有效模型上下文使用。
此结果没有证明所有可能编码都无法适配，但现有保留要求下不能靠丢失原文、来源或语义标记强行通过。

定向验证入口（复用现有离线防护，包含原阻塞位置和必要共享回归）：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B tests/run_structured_context_checks.py
```

## 同轮账本公共修复

`spec.atomic_json`串行提交，使用独立临时文件、flush/fsync及原子替换。
仅对Windows共享冲突最多执行4次本地替换，间隔25/50/75毫秒；没有API重试。
持续失败保留旧账本、预留名额和失败临时文件，仍停止执行。

现有CountedTransport的底层响应处理增加就地观察：HTTP返回后先保存脱敏的
`received_responses/<序号>.json`，再让原有传输层更新账本。历史传输脚本未修改。
账本持续失败时，这些响应证据仍可核对，不伪造业务成功或继续请求。

监控只读发布后不再修改的`monitor/*.json`小快照，使用现有入口的`--status`读取。
不要对运行中的`comparison_request_ledger.json`使用Get-Content轮询。

## 定向离线复跑

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --development-tests ledger
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --development-tests structured
```

本阶段没有真实模型请求、自然语言语义提取模型、跨任务记忆或外部检索。
没有验证任务成功率或Token节省。三策略统一入口、记录分析和离线Demo已具备；
原train34前缀仍存在已确认的预算停止，后续真实实验须另行授权，不能将固定响应回放当作效果结果。
