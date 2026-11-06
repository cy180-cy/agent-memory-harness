# 三策略统一评测与离线回放

正式入口：`validation/run_memory_comparison.py`。新增三策略编排和结果分析，继续使用原有 MemoryAgent、Orchestrator、Retail 工具、官方评分和持久请求账本；不另建模型客户端。原双策略默认入口和既有命令保持兼容。

正式提交后的版本校验区分固定上游来源与当前开发HEAD：当前提交必须继承上游，
实验仍绑定实际源码、数据、补丁和当前开发提交。已有收费授权不会因迁移目录而恢复。
运行时位于`validation/runtime_support/`，离线素材位于`tests/fixtures/memory/`，
不依赖本机历史运行目录。可运行`--tests --development-tests portable`检查提交可移植性、
独立FullHistory对照、校准、截断停止和授权边界。

冻结对照文件通过`.gitattributes`保留原始字节及固定SHA。历史数据原始SHA仍记录在
素材来源中，另以明确标记的LF规范化SHA校验跨平台文本检出；不改变原始数据、
模型输入或Token计数方式。

## 正式源码与版本身份

公共模型适配、总输出预算和离线防护位于 `validation/runtime_support/`。
`validation/repository_state.py` 校验仓库根目录及上游来源祖先关系，并分别记录固定上游来源
`fc0055dc4e0a316c3f83133267fbd6faaa770992`、当前开发 commit、分支和未提交补丁。
正常后续提交不会因 HEAD 改变而被拒绝；源码、数据指纹及收费授权绑定继续校验。

必要样本位于 `tests/fixtures/memory/`，区分冻结上游对照、合成输入和带来源的脱敏历史摘录。
完整历史运行目录、真实请求账本、有效授权、私有配置和缓存不随 Git 发布。
新 checkout 无需复制原开发者的 `validation/phase0_*` 或 `phase1_*` 目录。
结果仍写入新的独立运行目录，默认离线；所有真实执行必须另外取得新授权。

以下命令假设已按上游安装说明准备本仓库依赖及 Python 3.12/3.13 环境；
离线命令不会安装依赖或下载资源，也不需要 API Key。

## 可复跑命令

在仓库根目录的 PowerShell 执行，均不需要 Key、不联网：

```powershell
# 冻结四任务 × 三策略计划，并执行独立内存参考工具预检；不生成模型回复
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --suite

# 固定 Mock 响应经过真实本地 Agent/User/工具/评分执行链；不是自主模型效果实验
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --suite --mock --task-ids 34

# 新功能及少量账本/原A-B门控回归
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --development-tests evaluation

# $run 替换为上述输出的完整结果目录（含 comparison_plan.json）
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --demo $run --task 34
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --export $run
```

`--suite` 默认固定任务34、69、73、93。`--task-ids`可选择该清单的非空子集，不能随意扩充任务。未指定`--suite`时保留原 FullHistory/BudgetWindow 双组入口。`--demo`也支持历史单组目录；会明确显示旧普通LLMAgent、旧配置和历史真实来源。

`--export`支持多个来源，输出到新目录的`metrics.json`、`metrics.csv`、`summary_zh.md`。不覆写来源。Demo只展示Agent可见上下文片段、工具及事实来源，不展示User Simulator隐藏情境。

## 冻结实验定义

- 三组策略顺序：full_history、budget_window、structured_memory；每任务每策略1次，串行。
- Agent：openai/qwen3.7-plus，max_completion_tokens=3072，thinking_budget=1536。
- User：openai/qwen3.8-flash，max_completion_tokens=1024。
- 配置版本：comparison-output-agent3072-thinking1536-v1；temperature、seed和原有思考模式不变；重试全部0；禁止Judge/Review。
- BudgetWindow/StructuredMemory：7168输入预算、128余量。FullHistory不裁剪；这是工程取舍比较，不能称为相同输入预算比较。
- 每组请求20次、步骤30、任务超时300秒、HTTP阶段30秒。四任务共12组，最大240次请求；单任务三组最大60次。
- 每组由现有build_group重新构建独立Agent、Memory、User、数据库。请求在发送前持久化，失败/超时不退名额，不可重启清零。
- 沿用修复后的持久化启动检查。业务Reward=0但执行有效时可继续；资源、协议、评分、记录异常时后续组记为not_executed并停止整个矩阵。
- 当前仅有离线计划，没有收费授权。真实执行仍需新授权绑定本次身份、组内/全局上限及金额；缺授权时在读取Key前拒绝。旧授权不得复用。

## 固定任务选择

|train任务|业务覆盖|选择依据|
|---|---|---|
|34|修改待处理订单地址|已有官方离线参考验证，路径清晰|
|69|取消订单、礼品卡退款|已有离线验证，覆盖余额变化|
|73|已送达商品退货申请|覆盖退货操作，不使用已知缺陷工具|
|93|已送达单商品换货|覆盖商品变体和付款方式|

按操作覆盖、数据及工具缺陷选择，未使用模型成功率。准备阶段在新内存环境运行参考工具并立即抛出异常，避免官方参考重放吞异常的问题；此过程不把金标传给Agent，也不是Agent实验结果。清单完整保留任务、政策、工具、数据及代码补丁指纹。

含`modify_pending_order_items`的任务因已确认且未修复的变体复用缺陷排除；含非空NL断言的任务因需要Judge排除。24、57等空动作任务是合法任务，只因本轮聚焦数据库写操作而未选入，不能认定为金标无效。69、73、93声明NL评分项但断言为空，官方评分器直接返回，不产生Judge请求。

低成本分批建议：先批准单任务三策略（最多60次），确认链路及预算适用后再批准其余三任务（最多180次）。整份预先固定清单最多240次，不按分数临时改任务。可信同平台价格/账单尚缺，预计人民币费用为unknown，金额目标待用户批准；数量和输出上限不保证供应商扣费硬封顶。不得把此前¥4授权延用为本轮授权。

## 指标口径

- Reward/DB读官方结果；评分异常、资源停止、未执行保留空值，不补reward0。原分数另行保留。
- 请求数从原始尝试账本计算；Token从原始供应商响应计算，缺失/非法时保留已知小计和缺失次数。思考Token作为输出细分，不重复相加。
- Mock、历史真实、新真实、本地cl100k_base估算分别标记。按来源、策略、配置版本分组，旧成功运行不拼进新实验。
- 工具统计来自工具轨迹；记忆裁剪、删除块、保留事实及上下文估算来自上下文审计和状态记录。
- 只有条件一致且三个真实组都完成、评分和usage有效，才提供描述性Token/耗时差值；不输出统计显著性、稳定成功率或普遍节省结论。
- StructuredMemory第一版维护确定性事实和来源，普通自然语言保持原句；未实现通用语义摘要或自然语言授权推理。

目前已观察到train34历史响应Mock回放中StructuredMemory最低集合无法容纳7168/128预算的情形，应如实保留资源停止。不能为了取得完整三组分数临时提高预算或改算法；后续应先解决其必要保留集合的通用适用性，再申请完整真实矩阵。
