# Phase1-04.3：User 总输出配置修正

> 本文保留对应阶段的设计、配置和结果记录；“本轮/新配置”指该历史阶段。当前配置与可复跑入口见 [三策略统一评测](memory_evaluation.md)。命令从仓库根目录运行，公共运行时已整理到 `validation/runtime_support/`，不依赖本机历史运行目录。

Phase1-04.2 第12次请求是 User Simulator 的生成：HTTP 200，实际请求只有
`max_completion_tokens=512`，供应商与 LiteLLM 均返回 `finish_reason=length`。
`completion_tokens=512` 已包含 `reasoning_tokens=344`，按供应商的
`text_tokens=512` 细分可得可见回答部分168；不能再把344加到512上。
这次是达到总输出上限后的截断，没有超额。截断回复未交给 Agent，B未启动，
官方评分为空。原始记录和已消费授权保持不变。

## 下一轮配置

| 项目 | 历史配置 | 新配置 |
|---|---|---|
| 输出检查机制 | total-output-v2 | total-output-v2，不变 |
| 比较配置版本 | 无显式版本，按 legacy 1024/512 读取 | comparison-output-user1024-v1 |
| Agent max_completion_tokens | 1024 | 1024 |
| User max_completion_tokens | 512 | 1024 |

两组使用同一新版配置；不发送 `max_tokens`。模型、temperature、seed规则、
思考模式、提示词、7168/128输入预算、20/40请求上限和零重试均不改变。
旧配置不会升级或覆盖；新配置、新源码指纹构成新的实验身份，必须另外授权真实运行。

## 最小适配

`validation/memory_comparison/spec.py` 统一定义允许的比较配置版本并校验全部角色参数。
默认准备入口生成新版；缺少版本的历史计划仍按旧预算验证。

历史 `qianwen_adapter` 和 `budget_bindings` 会核对固定角色预算，因此
`execution.run_group` 在顺序执行的组作用域中临时绑定该组的预算别名，
同时给现有 CountedTransport 传入同一份预算。结束或异常后恢复别名，
不修改历史适配器文件、原始默认字典、输出检查算法或模型客户端。
这沿用已有的进程内角色绑定方式，入口仍限顺序执行。

提高配置上限不能保证供应商一定完整回答。`length` 仍在原兼容检查中拒绝，
回复保存后停止，不交给 Agent，也不放行B。超额、usage未知等暂停机制不变。
响应后检查不能撤销已经发生的 Token 或费用。

## 定向离线复跑

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --user-output-only
```

只运行 `tests/test_comparison_user_output.py`。使用实际本机 LiteLLM、MockTransport、
MemoryAgent、UserSimulator、Orchestrator 和现有执行器；任务为合成 Mock 域，
不运行 Retail train34。历史截断响应仅作离线回放，不能作为新版预算在供应商端
生效或已解决真实任务问题的证据。所有 Mock usage 单独标记，不混入历史真实统计。

离线入口先用环境变量白名单启动子进程，不读取或传递真实 Key，随后安装原有
网络、模型及写入防护。结果保存于新的独立目录，测试失败也不覆盖历史记录。
