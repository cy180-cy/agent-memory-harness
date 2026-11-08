# Agent Memory Harness：最终实验分析与工程结论

版本：第一版研究与工程报告；分析日期：2026-11-09；真实实验日期：2026-11-08。

本报告基于已保存的真实请求、工具轨迹、官方评分及脱敏CSV/JSON。Phase1-08只进行离线分析、文档和展示整理，没有新增真实API、重跑任务或调整记忆算法。

## 1. 项目目标与研究问题

长对话Agent不仅要保存聊天记录，还要在有限上下文中正确关联用户意图、实体、授权范围及工具执行结果。本项目将任务内记忆变为独立策略，并对模型可见上下文与完整原始轨迹分别记录。

研究问题是：完整历史、完整交互块滑窗、带来源的结构化事实，在相同任务/模型/工具环境中有什么工程取舍？输入预算何时真正触发，是否损坏工具配对？哪些失败属于业务行为，哪些是资源或执行限制？实际usage能否支持成本结论？

本次是全历史参考组与受输入预算约束策略之间的比较；FullHistory没有套用7168裁剪目标，不是同输入预算的能力实验。每组合一次，仅支持描述性观察。

## 2. 复用边界与个人实现

| 部分 | 上游提供 | 本项目实现 |
|---|---|---|
| 任务环境 | Retail任务、政策、数据库、工具、参考评分依据 | 固定小任务清单、隔离组状态、独立参考预检与证据记录 |
| Agent链路 | LLMAgent、UserSimulator、Orchestrator、消息与工具分发 | MemoryAgent最小接入与registry选择，不复制整套循环 |
| 记忆 | 上游普通完整对话历史 | MemoryStrategy / MemoryState、FullHistory、BudgetWindow、StructuredMemory |
| 评测 | 官方数据库/动作/NL评价入口 | 有效评分与资源停止分类、请求/usage/费用统计、CSV/JSON及中文Demo |
| 模型 | LiteLLM及SDK基础 | 千问适配、已保存响应回放、输出预算、授权与持久计数 |

上游为Sierra Research的tau2-bench，固定来源`fc0055dc4e0a316c3f83133267fbd6faaa770992`，包版本1.0.1。MIT及作者版权不变；[保留的上游说明](../README.upstream.md)与[许可证](../LICENSE)可追溯。没有把上游任务/评分器或作者论文结果当成个人原创。

接口为`initialize(history)`、`observe(state, message)`、`build_context(state)`。状态属于具体task/trial，接收Agent可见消息；政策、工具schema由Agent保留职责。MemoryAgent在`_build_model_context`调用策略，原始轨迹不受裁剪影响。详见[架构图](architecture.md)。

| 策略 | 已实现机制 | 当前边界 |
|---|---|---|
| FullHistory | 原序返回全历史副本，不增提示词、不变工具ID | 是等价性参考，不是性能优化 |
| BudgetWindow | 按用户交互分块，保留最新和前一完整块，从最旧可移除块删减，完整候选重计 | 最低集合放不下就发送前停止，不拆工具JSON |
| StructuredMemory | 目标/用户表述/实体/确认/工具操作/未解决事项，事实带来源和版本；超限时紧凑注入 | 确定性规则，无通用NL语义抽取；未解析自然语言保留原文；不做跨任务记忆 |

StructuredMemory不把普通“Yes”无条件解释为操作许可。仅对精确匹配的显式结构化确认形成明确记录；无法可靠提取时保留未知。旧值superseded保留审计，冲突不择一伪装确定。系统没有额外模型摘要或提取请求。

## 3. 实验身份、环境和冻结配置

- comparison_id：`20261108T223632706573+0800_50f1c5631a`。
- 授权：`AMH-PHASE1-07-2-RETAIL-01`，单次消费；运行代码HEAD：`3bd276666d2e03e2fd5c15ecd781f663d60b6c6d`。
- 实验结果已由`50d1b730`独立提交；本报告不会把后续文档提交当作运行代码版本。完整源码/数据指纹与原始证据哈希见[结构化结果](experiments/phase1_07_2_20261108/metrics.json)。
- 环境：Windows，本轮只读核验Python3.12.14；历史冻结依赖为tau2 1.0.1、LiteLLM1.82.6、OpenAI SDK3.24.0、HTTPX0.28.1、tiktoken0.14.0。项目声明Python>=3.12,<3.14，未新增或升级依赖。

| 条件 | 冻结值 |
|---|---|
| 范围与顺序 | train69 → 73 → 93；每题FullHistory → BudgetWindow → StructuredMemory；各1次，串行 |
| Agent | memory_agent；openai/qwen3.7-plus；max_completion_tokens=3072，thinking_budget=1536 |
| User | 上游UserSimulator；openai/qwen3.8-flash；max_completion_tokens=1024 |
| 输出版本 | comparison-output-agent3072-thinking1536-v1；无max_tokens冲突 |
| 生成条件 | temperature=0，非流式，保持原思考模式；内部seed42，不传供应商seed |
| 输入 | BudgetWindow / StructuredMemory：7168，余量128；FullHistory完整历史 |
| 时间与计数 | 每组20请求/30步/300秒，总180请求；HTTP各阶段30秒 |
| 其他调用 | SDK/HTTP/任务重试0；Judge/Review0；无额外模型或健康探测 |
| 费用 | 全批¥10本地预估停止线；输入估算加20%，输出按总上限预留 |

模型通过指定OpenAI兼容端点及LiteLLM调用，不声称复现作者论文的模型配置。同seed规则不能固定供应商采样或用户模拟器全部回复；各组真实对话允许自然分化。

任务按业务覆盖预先选择：69取消订单/礼品卡退款，73已送达商品退货，93单商品换货。没有按本轮模型成败挑题。避开已知未修复的多商品修改工具问题；三题NL断言均为空，官方NL评价短路返回，不调用默认Judge。所有正常结束组均经官方评分入口，未另写成功判定代替Reward。

## 4. 结果与分类口径

9组计划中实际执行8组；7组正常结束并取得有效评分（5成功、2业务失败），1组资源停止未评分，1组未执行。未评分/未执行保留null，绝不补Reward=0。公共执行异常与策略资源停止另列，本批没有HTTP失败、超时、工具错误或评分异常。

| 任务 | 策略 | 状态 | Reward / DB | Agent/User请求 | 输入Token | 输出Token | 总Token | 工具 | 秒 | 估算元 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 69 | FullHistory | 完成 / 业务失败 | 0 / 0 | 8/4 | 50668 | 2482 | 53150 | 5 | 57.70 | 0.0913465 |
| 69 | BudgetWindow | 完成 / 成功 | 1 / 1 | 11/6 | 75717 | 4043 | 79760 | 6 | 89.91 | 0.1363939 |
| 69 | StructuredMemory | 完成 / 成功 | 1 / 1 | 11/6 | 75557 | 4007 | 79564 | 6 | 76.73 | 0.1364849 |
| 73 | FullHistory | 完成 / 成功 | 1 / 1 | 7/4 | 44313 | 2137 | 46450 | 4 | 50.29 | 0.0806276 |
| 73 | BudgetWindow | 完成 / 成功 | 1 / 1 | 8/5 | 52333 | 2691 | 55024 | 4 | 61.73 | 0.0955038 |
| 73 | StructuredMemory | 完成 / 成功 | 1 / 1 | 7/4 | 44253 | 2191 | 46444 | 4 | 49.24 | 0.0810038 |
| 93 | FullHistory | 完成 / 业务失败 | 0 / 0 | 10/5 | 72065 | 4373 | 76438 | 6 | 87.66 | 0.1348263 |
| 93 | BudgetWindow | 资源停止 / 未评分 | — | 5/1 | 29239 | 905 | 30144 | 5 | 22.91 | 0.0512497 |
| 93 | StructuredMemory | 未执行 | — | 0/0 | — | — | — | 0 | — | — |

有效评分中FullHistory成功1/3；BudgetWindow成功2/2，另1组资源停止；StructuredMemory成功2/2，另1组未执行。不能把缺失组算作失败，也不能忽略缺失组后声称2/2就是普遍100%成功率。

## 5. 逐任务解释

### train69：区分成功决策与事后裁剪

FullHistory完成身份和订单查询，但User在明确确认取消的同一回复里发送`###STOP###`。上游以user_stop结束；取消工具没有执行，数据库未变，官方Reward/DB=0。这里是有效业务失败；不能将其归结为长历史导致遗忘，也不能为了改善分数删除结束标记。

BudgetWindow和StructuredMemory分别在明确用户确认后执行一次`cancel_pending_order`，工具成功返回并进入后续Agent回复，官方DB与参考终态一致。这两组均17请求；FullHistory仅12请求且业务未完成，不能用后者更少的Token说明全历史更省，也不能把成功差异直接归因于记忆。

| 策略 | Agent生成位置 | 本地完整输入估算 | 删除块 | 注入事实 | 发生时机 |
|---|---:|---|---:|---:|---|
| BudgetWindow | 11 | 7425 → 5450 | 3 | 0 | 取消成功后，最终Agent回复前 |
| StructuredMemory | 11 | 7398 → 7004 | 4 | 26 | 取消成功后，最终Agent回复前 |

两次最终候选都满足“估算+128≤7168”，并保留必要工具关系。该证据证明真实路径中裁剪与注入生效；**不能证明裁剪促成了此前取消成功**。

StructuredMemory保存138条事实（131 active、2 superseded、5 response_received_unverified），此次只注入26条。旧订单观察`f125`来自消息14/15，在近期保留块`[17,19)`与`[19,22)`之外，仍通过source_id进入记忆段；完整来源快照留在本地。订单状态更新则保留旧pending版本、将新cancelled观察设为active。原始状态文件的哈希记录在[结构化结果](experiments/phase1_07_2_20261108/metrics.json)中。

### train73：成功但不支持节省结论

三组均执行一次`return_delivered_order_items`并成功，均未裁剪。StructuredMemory维护92条事实（87 active、2 superseded、3 response_received_unverified），但未向模型注入独立记忆段，因为全历史仍可容纳。

BudgetWindow比FullHistory多2次请求、8574总Token（18.46%）和11.44秒；StructuredMemory同为11次请求，仅少6总Token（约0.013%）、少1.05秒，输出Token反而多54。对话自然分化且样本各一次，没有可证明的显著节省，也不能把未触发的记忆压缩作为原因。

### train93：不可压缩交互块与门控缺失样本

FullHistory在核对两份订单、纠正待换货对象后获得用户确认，但User再次同时输出结束标记；换货写工具未执行，DB未变，官方Reward/DB=0。保留该失败，不现场调整提示词或重新运行。

BudgetWindow已完成6次真实请求和5次只读工具调用，在第6个Agent上下文构建位置，固定输入与最低保留集合估算为7328。加128余量得7456，超过7168；当前完整用户交互块含连续查询及工具返回，没有可删除的旧块。发送第7个组内请求前停止，没有截断JSON或补造评分。

现有门控只对满足条件的StructuredMemory自身上下文超限允许继续独立后续组。本次停止来自BudgetWindow，gate_8拒绝继续，因此train93 StructuredMemory未执行。它的潜在表现未知；本轮不改变门控或提高预算补齐样本。

## 6. usage、费用与可靠性证据

| 角色 | 请求 | 输入 | 输出 | 总Token | 思考Token（输出子集） |
|---|---:|---:|---:|---:|---:|
| agent | 67 | 412558 | 16334 | 428892 | 8959 |
| user | 35 | 31587 | 6495 | 38082 | 4391 |

合计102次HTTP 200、444145输入、22829输出、466974总Token；102份原始响应、局部账本、全局计数一致，usage无缺失。思考Token已经包含在completion_tokens中，没有重复相加。40次工具调用无错误、ID与返回正确配对；5次业务写操作未重复。8组初始DB哈希相同且独立构造，未继承前组状态。

所有实际Agent请求包含3072总输出与1536思考预算，User为1024；未发送冲突max_tokens。此次预算检查全部通过，不代表供应商侧存在绝对硬封顶。62次原始纯文本stop和40次纯工具tool_calls与转换后对应；本批没有混合文本响应，不将此前离线兼容修复冒称为此次新增真实覆盖。

费用按此前用户确认的同平台单价估算：Agent输入≤256K档为输入1.60元/百万、输出6.40元/百万；更高档4.80/19.20；User0.80/2.70。本批均用低档，不计缓存/免费额度折扣。102项费用均完成usage结算，合计¥0.8074365；供应商账单未知，未调用账单、余额或模型列表接口。¥10是本地估算停止线，不是供应商绝对扣费上限。

累计执行耗时499.08秒。按策略合计Token为176038/164928/126008，但完成任务范围和对话长度不同，这三个总量不能用于宣称成本节约。

| 数据类型 | 本报告用途 | 不能混淆为 |
|---|---|---|
| 供应商原始usage | 实际已发生Token与按单价估费 | 本地tokenizer估算或已核实账单 |
| 本地cl100k_base完整输入估算 | 输入选择、裁剪和发送前停止 | 千问原生分词或直接计费Token |
| 固定Mock / 历史响应回放 | 接口、协议、预算与隔离逻辑的确定性验证 | 自主Agent成功率或新真实实验结果 |
| 未评分 / 未执行 | 保留null和具体原因 | Reward=0或同等质量节省 |

## 7. 工程结论与局限

已验证的工程成果是三种策略通过同一MemoryAgent、真实模型、官方工具和评分链路工作；输入选择可追溯到原始消息位置，结构化事实可追溯到工具与版本，资源停止未被包装成业务评分。请求数、费用、数据隔离和可公开结果之间也有明确边界。

尚不能回答哪种策略普遍更优。限制包括每组合一次、只有三个train任务、没有随机化重复、User回复分化、FullHistory与有界组输入目标不同、输入计数为本地估算、一个策略任务缺失、完整工具块可能不可压缩、自然语言提取能力有限及账单未核实。无统计显著性、稳定成功率或pass^k结论。

本轮结束主要实验开发，不通过改算法、调预算或收费重跑美化结果。第一版的工程价值是可替换的记忆设计、严格区分失败类型的评测和能解释负结果的证据。若未来扩展效果主张，必须另行固定方案、获得授权并增加独立重复；不是当前交付的前提。

## 8. 证据与复现

- [阶段原报告](experiments/phase1_07_2_20261108/report_zh.md)、[CSV](experiments/phase1_07_2_20261108/metrics.csv)、[JSON](experiments/phase1_07_2_20261108/metrics.json)原样保留，后者记录源代码、数据及原始文件哈希。
- [README中的离线Demo](../README.md#环境与离线demo)可直接读取公开派生JSON；无需模型、Key或历史运行目录。它展示已有结果，不重新执行工具或计算官方评分。
- 本机原始目录为`validation/phase1_comparison/20261108T223632706573+0800_50f1c5631a/live_execution/`；完整原始账本和轨迹不随Git公开。从原响应重新核算usage需要这些本地材料，公开汇总不足以独立重算官方Reward。
- 原始状态证据：`t69_full_history/trajectory.json`、`t93_full_history/trajectory.json`记录结束标记；`t69_structured_memory/agent_state.json`记录来源和版本；`t93_budget_window/summary.json`与`gate_8.json`记录停止和阻断。目录均相对于上述本地根目录。
- 源码边界：[记忆接口](../src/tau2/agent/memory.py)、[MemoryAgent](../src/tau2/agent/memory_agent.py)、[BudgetWindow](../src/tau2/agent/budget_window.py)、[StructuredMemory](../src/tau2/agent/structured_memory.py)、[结果分析](../validation/memory_comparison/results.py)。

本报告不包含真实凭据、Authorization头、原始请求账本、客户个人信息或未脱敏轨迹。上游基准的虚构实体在公开事实示例中使用展示别名；完整证据哈希及本地原件保持不变。

## 9. Phase1-08交付验证

2026-11-09完成6类定向检查：公开Demo分别展示69/73/93、原本机证据Demo兼容、拒绝将Mock来源标成真实结果、拒绝不支持的公开路径。均符合预期；未执行任务、Mock生成或真实API请求。修改的两个展示入口通过Python语法解析，新文档相对链接与敏感模式检查通过。

原阶段CSV/JSON/报告及MIT文件哈希未变；上游README正文单独保留，字节与原正文一致。新增源码仅为同一Demo入口读取公开摘要，未改Memory、模型参数、执行器或评分算法。验证日志只保留本地忽略目录`validation/phase1_delivery/`。

自写文档的Git空白检查通过。保留的上游README中有4处原BibTeX行尾空格，已确认来自原正文；为保持原文不做清理，不将其误称为新文档检查全部无告警。

已尝试仓库要求的`make check-all`，本机未安装make，不能将该检查记为通过；本轮未安装依赖或重跑历史测试集。公开展示与分析报告已具备第一版交付条件，GitHub发布仍待用户确认。
