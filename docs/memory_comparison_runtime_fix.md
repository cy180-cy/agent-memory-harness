# Phase1-04.5：真实模式门控与Agent输出预算

> 本文保留对应阶段的设计、配置和结果记录；“本轮/新配置”指该历史阶段。当前配置与可复跑入口见 [三策略统一评测](memory_evaluation.md)。命令从仓库根目录运行，公共运行时已整理到 `validation/runtime_support/`，不依赖本机历史运行目录。

## 已确认的问题

Phase1-04.4真实对比在第9次请求停止：Agent的实际请求为
`max_completion_tokens=1024`，未显式发送`thinking_budget`。
供应商原始响应及LiteLLM转换结果均为`length`；completion_tokens=1024，
其中reasoning_tokens=1024，可见content为空。原始数据、9次请求账本、
B未执行状态和已消费授权全部保留。

另一个公共问题是：全局账本及组结果写`live`，历史CountedTransport写`real`，
A→B门控直接比较这两个字符串，因此正常真实模式也可能被保守拒绝。

## 模式修正

内部模式固定为`live`或`mock`，只允许历史别名`real -> live`。
未知名称、非布尔live标志、互相冲突的标志在执行前报错。
`run_pair(mode="real")`与`live=True`进入同一授权检查，不能转成未授权真实调用。
CLI仍使用现有`--execute`，没有另起客户端或任务循环。

新执行记录在现有持久化回调中归一模式；来自传输层的`real`另保存在
`transport_mode`供溯源。读取旧记录时仅在内存中比较已知别名，不改写历史文件。
未知模式、组/账本模式不一致仍拒绝。门控继续检查完成状态、有效评分、usage、
资源限制、工具/轨迹及产物读回；Reward=0不构成拒绝理由。
发送前计数、20/40上限、消费授权和防重复启动规则保留。

## 新输出配置

版本：`comparison-output-agent3072-thinking1536-v1`。
数值预算检查机制仍是原来的`total-output-v2`。

| 参数 | 旧版本comparison-output-user1024-v1 | 新版本 |
|---|---|---|
| Agent max_completion_tokens | 1024 | 3072 |
| Agent thinking_budget | 未显式发送 | 1536 |
| User max_completion_tokens | 1024 | 1024 |
| enable_thinking / reasoning_effort | 未显式发送 | 保持未显式发送 |
| max_tokens | 不发送 | 不发送 |

两组使用同一新配置。模型、temperature、seed规则、思考模式、提示词、
输入7168/128、时间限制及零重试均不调整。

配置经已有Agent/User llm_args进入角色绑定；历史qianwen_adapter文件只读复用。
顺序执行的组作用域内临时绑定角色参数和预算，结束或异常时恢复原函数及字典别名。
在LiteLLM参数中使用`extra_body={"thinking_budget":1536}`，HTTP JSON中应为顶层字段：

```json
{"model":"qwen3.7-plus","max_completion_tokens":3072,"thinking_budget":1536}
```

现有执行器新增的HTTP请求钩子在传输发送前核对实际输出字段，参数丢失、冲突或
出现残留extra_body包装时拒绝；不安装依赖、不绕过LiteLLM。
回复先保存，再执行原有资源与协议检查。length截断、输出超额或usage未知
仍会停止，禁止将该回复中的工具调用继续执行。

## 平台文档依据与限制

2026-11-05查阅[思考模式](https://platform.qianwenai.com/docs/developer-guides/text-generation/thinking)
及[OpenAI Chat API参考](https://platform.qianwenai.com/docs/api-reference/chat/openai-chat)。
文档说明Qwen3.7支持thinking_budget；该非标准参数通过extra_body传入。
max_completion_tokens限制含思考的总输出，thinking_budget限制思考部分。
Qwen3.7-Plus文档默认开启思考，本轮保持原请求中未显式切换的行为。

本机LiteLLM 1.82.6经MockTransport验证了参数到达HTTP JSON。
这不证明供应商实际执行新思考预算，也不保证未来真实任务完整结束。
3072是总输出预算，不能再把1536加上去当成允许总量。
原有本地零容差与暂停规则不变；响应后检查不能撤销已发生的Token或费用。

## 定向测试

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --runtime-fix-only
```

只运行`tests/test_comparison_runtime_fix.py`。真实任务及Key均不使用。
普通工具闭环使用实际MemoryAgent、UserSimulator、Orchestrator、Mock域工具和
现有执行器，全部模型响应来自MockTransport。
模式门控另使用明确标记的合成live账本和测试内准入替身，不创建有效收费授权；
这些合成名额不计为真实API请求。未知模式、未授权请求、Reward=0放行、
资源/评分异常阻断、截断/超额/usage异常、参数丢失、请求上限与防重复启动均覆盖。

既有Phase1-03和Phase1-04.3测试夹具显式固定到各自原配置版本，
预算数值断言和冻结预期不变。没有重跑全量历史回归。

新配置单独冻结为未批准草案，必须有新的实验身份绑定授权才可真实执行。
Phase1-04.4授权已经消费，本轮不恢复或复用它。
