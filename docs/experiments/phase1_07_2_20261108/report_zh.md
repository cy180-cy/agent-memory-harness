# Phase1-07.2：三策略 Retail 真实实验结果

日期：2026-11-08；授权：`AMH-PHASE1-07-2-RETAIL-01`。
实验：`20261108T223632706573+0800_50f1c5631a`。运行代码 HEAD：`3bd276666d2e03e2fd5c15ecd781f663d60b6c6d`；分支：`feature/agent-memory-evaluation`。

本轮实际执行8组：7组正常结束并获得有效官方评分（5成功、2业务失败），1组输入预算停止；最后1组因现有门控未执行。102次真实请求全部HTTP 200；原始usage完整。总Token为466,974，按指定单价估算¥0.8074365，供应商账单未核实。没有Mock混入、自动重试或追加实验。

## 九组结果

Reward与DB在本轮已评分组中相同；“—”表示未评分或未执行，不是0。费用列是本地估算，时间为每组端到端执行时间。

| 任务 | 策略 | 状态 | Reward / DB | 请求 | 输入Token | 输出Token | 总Token | 工具 / 错误 | 耗时(s) | 估算费用(元) |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| 69 | FullHistory | 完成，业务失败 | 0 / 0 | 12 | 50668 | 2482 | 53150 | 5 / 0 | 57.70 | 0.0913465 |
| 69 | BudgetWindow | 完成，业务成功 | 1 / 1 | 17 | 75717 | 4043 | 79760 | 6 / 0 | 89.91 | 0.1363939 |
| 69 | StructuredMemory | 完成，业务成功 | 1 / 1 | 17 | 75557 | 4007 | 79564 | 6 / 0 | 76.73 | 0.1364849 |
| 73 | FullHistory | 完成，业务成功 | 1 / 1 | 11 | 44313 | 2137 | 46450 | 4 / 0 | 50.29 | 0.0806276 |
| 73 | BudgetWindow | 完成，业务成功 | 1 / 1 | 13 | 52333 | 2691 | 55024 | 4 / 0 | 61.73 | 0.0955038 |
| 73 | StructuredMemory | 完成，业务成功 | 1 / 1 | 11 | 44253 | 2191 | 46444 | 4 / 0 | 49.24 | 0.0810038 |
| 93 | FullHistory | 完成，业务失败 | 0 / 0 | 15 | 72065 | 4373 | 76438 | 6 / 0 | 87.66 | 0.1348263 |
| 93 | BudgetWindow | 资源停止，未评分 | — | 6 | 29239 | 905 | 30144 | 5 / 0 | 22.91 | 0.0512497 |
| 93 | StructuredMemory | 未执行 | — | 0 | — | — | — | 0 / 0 | — | — |

## 可核对的结论

- FullHistory：3组有效评分，成功1/3；BudgetWindow：2组有效评分，成功2/2，另1组资源停止；StructuredMemory：2组有效评分，成功2/2，另1组未执行。分母不同，不能用这些比例声称稳定胜出。
- train69与train93的FullHistory均以`user_stop`正常结束，但User Simulator在确认操作的同一回复中发出`###STOP###`，取消/换货工具尚未执行；DB未变化，官方Reward=0。这是本次真实交互中的业务失败证据，不是账本或API失败，不能仅归因为记忆策略。
- train69的BudgetWindow与StructuredMemory均在明确用户确认后各执行一次取消工具；train73三组均在确认后各执行一次退货工具。40次工具调用均有唯一调用ID和正确配对的成功返回；5次业务写操作无重复，工具结果均进入后续Agent流程。
- 8组初始DB哈希相同，使用独立构造状态；成功组的DB与官方参考终态匹配。FullHistory的两个业务失败组及train93预算停止组DB未变化。未修改原始数据库、政策或评分器。

## 上下文和结构化记忆

| 任务 / 策略 | Agent生成位置 | 完整输入估算：前→后 | 删除块数 | 注入事实 | 结果 |
|---|---:|---|---:|---:|---|
| 69 / BudgetWindow | 11 | 7425 → 5450 | 3 | 0 | trimmed |
| 69 / StructuredMemory | 11 | 7398 → 7004 | 4 | 26 | structured |
| 93 / BudgetWindow | 6 | 7328 → 7328 | 0 | 0 | context_budget_exceeded |

以上是同一请求边界上的本地cl100k_base估算，不是供应商Token。train69的BudgetWindow裁剪1次，删除3块；StructuredMemory裁剪1次，删除4块并注入26条有来源的记忆。两次裁剪都发生在取消工具成功之后的最终Agent回复前，因此不能宣称裁剪改善了前面的取消决策。

StructuredMemory的train69状态保存138条事实（131 active、2 superseded、5 response_received_unverified），train73保存92条（87 active、2 superseded、3 response_received_unverified），均保留来源引用。状态总事实数不等于提示词注入数。train73未触发裁剪或事实注入，不能用其结果证明压缩效果。

train93 / BudgetWindow在第6个Agent上下文构建位置停止：最低集合估算7,328，加128为7,456 > 7,168。当前完整用户交互块含连续5次工具查询与返回，没有可删除的旧交互块；未截断工具JSON，未发送第7次组内请求。保持Reward/DB为空。

现有继续规则只豁免满足条件的StructuredMemory自身上下文超限；本次发生在BudgetWindow，`gate_8.json`明确拒绝后续组。故train93 / StructuredMemory为not_executed，不代表该策略业务失败。未修改门控、提高预算或恢复授权。

## Token、费用与协议

| 角色 | 请求 | 输入Token | 输出Token | 合计 | 思考Token（输出子集） |
|---|---:|---:|---:|---:|---:|
| agent | 67 | 412558 | 16334 | 428892 | 8959 |
| user | 35 | 31587 | 6495 | 38082 | 4391 |

总输入444,145；总输出22,829；总计466,974。思考Token未重复相加。全部102次usage有效且费用预留已结算，无待定/未知请求；供应商最终账单仍为unknown。

沿用用户指定的同平台价格：Agent输入≤256K时输入¥1.60/百万、输出¥6.40/百万；更高档为¥4.80/¥19.20；User为¥0.80/¥2.70。本轮所有请求适用较低输入档。发送前按实际请求体估算输入并加20%，输出按3072/1024预留；未使用缓存或免费额度折扣。金额为本地估算而非平台扣费承诺，没有查询余额或账单。

8组累计执行时间约499.08秒。所有请求均使用指定端点和两种模型；Agent实际发送max_completion_tokens=3072与thinking_budget=1536，User=1024，无max_tokens冲突。重试、Judge、Review及额外模型请求均为0，所有响应的输出预算检查通过。

本轮原始响应为62次纯文本stop、40次纯工具tool_calls，转换后对应标记一致，未实际触发混合文本规范化或stop→tool_calls兼容例外。上一轮混合响应离线证据保留，本轮不据此新增未经观测的兼容性结论。

train73三组均成功且未裁剪：BudgetWindow比FullHistory多8,574总Token（约18.46%），请求多2次；StructuredMemory仅少6总Token（约0.013%），不足以表示优化收益。差异伴随真实User回复和对话长度分化。train69的两个成功组约79.6K–79.8K总Token，高于53.2K的FullHistory失败组，不能进行同等质量节省比较。train93不计算策略节省比例。

## 冻结配置与证据保护

- 任务69/73/93，各策略1次，顺序执行；每组20请求/30步/300秒，全局180请求/¥10本地估算停止线，HTTP各阶段30秒，重试0。
- Agent qwen3.7-plus，User qwen3.8-flash；输出配置comparison-output-agent3072-thinking1536-v1。temperature=0，seed沿用42规则，未新增供应商seed参数或修改思考模式。BudgetWindow与StructuredMemory输入7168/余量128；FullHistory不裁剪，不称为相同输入预算实验。
- 三个任务均无需真实NL Judge；官方评分对空NL断言直接返回，不调用默认GPT-4.1。已评分7组没有金标重放/评分警告。
- 新授权只启动一次，绑定本comparison_id和源码指纹，102/180名额已消费；各组最多17/20。授权状态仍为claimed_no_automatic_resume，不能重新消费剩余名额。
- 运行前后261个冻结源码/运行依赖文件、8个数据及用户指引文件哈希一致；抽核6份既有授权、账本、响应和脱敏历史结果哈希一致。源码工作区原本干净，本次仅新增这3份公开汇总；持久凭据未修改。

原始证据仅保留本地：`validation/phase1_comparison/20261108T223632706573+0800_50f1c5631a/live_execution/`；离线重算导出：`validation/phase1_comparison/20261108T224740577049+0800_3473389b28/exports/`。公开JSON只含汇总、配置与哈希，不含密钥、请求ID、客户资料、原始对话或账本。

可离线重新汇总（不发送API）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --export validation/phase1_comparison/20261108T223632706573+0800_50f1c5631a/live_execution
```

本轮没有新增/重跑测试，没有进行后续收费实验。停止后仅做原始记录核对与脱敏导出。需要下一轮单独授权才能补齐缺失组；不建议直接以这批小样本宣称成功率提升。优先保留并说明User提前结束与完整工具交互块超预算这两个已观察到的问题。

提交前核对：CSV/JSON的9行、102次请求、466,974 Token与费用总额一致，未评分字段保留为空；公开字段及密钥模式检查通过。已尝试仓库要求的`make check-all`，但本机没有make，未安装依赖或替代运行全量测试。本次只提交三份脱敏结果文档，不含源码改动。
