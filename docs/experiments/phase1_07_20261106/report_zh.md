# Phase1-07 三策略 Retail 真实实验记录

日期：2026-11-06（Asia/Shanghai）。本次触发预定公共故障停止条件：9组中1组开始、0组完成，剩余8组未执行。未产生有效官方评分，不构成三策略性能对比结果。

## 范围与实际结果

授权：`AMH-PHASE1-07-RETAIL69-73-93-01`，仅消费一次。comparison_id：`20261106T174245439114+0800_8e7d5cc84c`。
计划为 train69、73、93 × FullHistory、BudgetWindow、StructuredMemory，每组1次、顺序执行；请求上限20/组、180/批，费用停止线¥10。

| 任务 | 策略 | 实际状态 | Reward / DB | 请求数 | 实际总Token |
| --- | --- | --- | --- | ---: | ---: |
| 69 | full_history | 异常停止 | — / — | 4 | 12178 |
| 69 | budget_window | 未执行 | — / — | 0 | — |
| 69 | structured_memory | 未执行 | — / — | 0 | — |
| 73 | full_history | 未执行 | — / — | 0 | — |
| 73 | budget_window | 未执行 | — / — | 0 | — |
| 73 | structured_memory | 未执行 | — / — | 0 | — |
| 93 | full_history | 未执行 | — / — | 0 | — |
| 93 | budget_window | 未执行 | — / — | 0 | — |
| 93 | structured_memory | 未执行 | — / — | 0 | — |

`—`表示未评分或未运行，不是0分。FullHistory开始1/3组、执行异常1组；另两种策略各开始0/3组。三者有效评分数均为0，因此评分样本成功率均为不可计算。资源停止数均为0，不能把未运行或中断的低Token解释为优化收益。

## 供应商usage与费用

| 角色 | 请求 | 输入Token | 输出Token | 合计Token | 按用户价格估算费用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent | 2 | 10,386 | 434 | 10,820 | ¥0.0193952 |
| User | 2 | 1,074 | 284 | 1,358 | ¥0.0016260 |
| 合计 | 4 | 11,460 | 718 | 12,178 | ¥0.0210212 |

4次均为HTTP 200，usage缺失0次，网络错误/超时0次，自动重试0次。433个reasoning Token属于718输出Token的细分，没有重复相加。FullHistory组耗时15.540秒；整个执行器耗时15.872秒。

价格来自本轮用户提供的控制台报价：Agent输入/输出¥1.60/6.40每百万Token，超过256K按¥4.80/19.20；User为¥0.80/2.70。本地将256K门槛保守解释为256,000 Token。本轮Agent单次输入最大5,249，使用低档；不计任何缓存优惠或免费额度。供应商实际账单仍为unknown。

每次实际HTTP请求体以既有cl100k_base计数器估算输入、向上取整预留20%，输出按3072/1024完整预算预留；4次都已根据供应商usage结算，未结算预留为0。JSON保留逐次估算、实际usage及费用。本地停止线不是供应商绝对扣费硬封顶。

## 停止原因与业务轨迹

第4次请求为Agent，原始和LiteLLM转换后finish_reason均为`tool_calls`，输出314（含reasoning 187），没有输出截断或预算超额。响应同时含非空普通文本和一个`find_user_id_by_name_zip`调用。工具结构、ID、参数转换没有丢失，但兼容层在`validation/runtime_support/compatibility.py:70`拒绝混合消息，原因`mixed_text_and_tools`。上游`src/tau2/orchestrator/orchestrator.py:722`也禁止单条消息同时包含文本和工具调用。

这是当前接入链路与上游消息约定的兼容性异常；不能据此认定供应商HTTP或整个OpenAI格式无效。原始分类字段仍原样保留，没有改成通过。

对话进入身份信息收集阶段。Agent产生1个身份查询工具候选，实际执行工具0次、工具错误0次，尚未执行订单查询、操作确认、取消、退款或工具结果回传。初末数据库哈希相同，非目标数据也未变化。未运行评分，不写Reward=0。正式组间门控拒绝下一组，其余8组均保留`not_executed`。

## 本轮获准的最小修复与离线证据

复用原执行器与账本，新增发送前金额预留、按usage结算、未知费用暂停及授权ID跨目录一次性绑定；只为证据完整的StructuredMemory `context_budget_exceeded`允许继续下一独立组，其他公共故障仍停止。进程变量不存在时仅读取本项目Windows当前用户级变量，未打印、持久化或修改密钥。

11项新增定向测试通过，失败0；34次Mock传输、真实API0次。覆盖预留/释放、价格阶梯、usage未知、持久化失败不发送、真实传输边界拒绝费用越限、合成上下文停止后继续、公共故障停止、重复启动拒绝。Mock不计入以上真实请求与费用。

提交前已尝试仓库约定的`make check-all`，因本机没有`make`未能运行；未安装依赖，也未把它记为通过。3份公开汇总已核对9行、4次请求、Token及空评分，并扫描密钥、认证头和原始个人字段。

获准修改的7个文件为`validation/memory_comparison/fees.py`、`evaluation.py`、`execution.py`、`spec.py`，`validation/runtime_support/budget_transport.py`，`validation/run_memory_comparison.py`及`tests/test_suite_fees.py`。没有修改记忆核心算法、模型参数、任务、政策或评分算法。

## 固定配置与证据保护

Agent `openai/qwen3.7-plus`：总输出3072、thinking_budget1536；User `openai/qwen3.8-flash`：总输出1024；没有max_tokens冲突参数。配置版本`comparison-output-agent3072-thinking1536-v1`，temperature=0，内部seed=42，供应商seed沿用原封装不发送。预算策略7168/128；FullHistory不裁剪。每组30步、300秒，HTTP各阶段30秒，零重试，Judge/Review为0。

运行分支`feature/agent-memory-evaluation`；运行时HEAD `ce5e649d264b6ca3f2122c07c6ae7544b14871ba`；上游来源`fc0055dc4e0a316c3f83133267fbd6faaa770992`。获准修复以未提交补丁参与运行，补丁哈希`172c641c0402e7186d3f1b85c9d156997fcb0863613d64a7698b095d0603aeef`，完整实验身份`24e3d038047bc16d7d95d9a38b67fe7fb2ed1eacc8433f0092905c2575d16a58`。JSON另存必要源码指纹和依赖版本。运行后核对冻结源码、Retail原始文件和User政策指纹一致。

本次结果提交仅纳入本目录3份脱敏汇总，执行器修复保留未提交；源码补丁及新文件不可仅凭运行HEAD还原。原始响应、身份信息、reasoning正文、请求ID、授权文件与费用账本均仅留本地忽略目录，没有复制到本报告。旧实验与持久凭据未修改，没有推送。

本地原始证据：`validation/phase1_comparison/20261106T174245439114+0800_8e7d5cc84c/live_execution/`，重点文件为`suite_summary.json`、`comparison_request_ledger.json`、`t69_full_history/model_responses.json`、`t69_full_history/received_responses/0004.json`；摘要保留其内容哈希。

## 结论

本轮按停止条件结束，未重发或恢复授权。下一项必要工作是使用已保存第4次响应离线确定混合文本/工具消息的适配方式，遵守上游消息约定并保留原始证据，完成定向验证后再申请新的真实实验授权。本轮不能判断三种记忆策略优劣，也不能报告成功率或成本节省。每任务每策略一次即使完成也只支持探索性观察。
