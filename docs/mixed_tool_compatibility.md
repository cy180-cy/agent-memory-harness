# 混合文本与工具调用适配（Phase1-07.1）

本轮仅离线修复接入层；未改 Orchestrator、Message、Agent 提示词、Memory 策略或 Retail 业务。

## 转换边界

供应商同时返回 `content` 与 `tool_calls` 时，保留原始 HTTP 响应、LiteLLM 原始结构、usage、finish_reason 和完整有序工具调用。仅供 tau2 使用的 AssistantMessage 副本将过渡性 `content` 置为 `None`。

`qianwen_adapter.invoke` 在原有输出预算检查之后、消息释放之前调用 `normalize_mixed_response`。先沿用工具名、调用ID、参数类型和转换一致性校验，再用 `mixed_response.transition_audit` 逐句检查文本；审计落盘成功后才交给上游。`budget_bindings` 的兼容性检查会重新核对该审计，不是直接删除 mixed 检查。

当前为保守的确定性规则，不是通用自然语言理解：

- 只接受有 READ 且不修改状态的工具，有限的礼貌语、查询意图和有可核对来源的输入复述。
- 姓名候选必须同时匹配查询参数及最近可见用户消息中的用户名组成；仍标记为“暂定查询参数”，不升级为身份确认或授权。
- 涉及实质答复、确认问题、写操作、未知表达或无法逐句核对的内容停止。不能承诺未来任意供应商表述都可转换。
- 多工具保持ID、名称、参数及顺序；不同工具的同名不同值参数不能用于证明文本内容，但不会阻断无参数引用的过渡说明。

被移出的原文及逐句依据保存为 `mixed_content_normalization`，放在审计记录中；它不进入下一轮模型输入。原始协议检查仍标记 `mixed_text_and_tools` 差异，适配后检查单独报告通过。不改写旧失败报告或已消费授权。

## 实际验证

命令（仓库根目录，现有虚拟环境）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B validation/run_memory_comparison.py --tests --development-tests mixed
```

最终14项通过、失败0、跳过0；首轮发现的两个实现/测试适配问题已修复，首轮失败材料保留。两次测试合计29次Mock传输，真实API请求0；最终一轮15次Mock。没有读取真实Key、联网或重跑train69。

历史第4次响应从 HTTP MockTransport 进入实际 LiteLLM → MemoryAgent → 上游通信校验 → 上游工具分发。只执行一次只读身份查询，不运行任务会话或评分；ToolMessage ID关联正确，下一轮Agent输入含工具结果，并以明确标记的合成文本结束。此回放中的两次模型响应分别为历史捕获响应和合成响应，不能证明新的真实多轮运行成功。

原文可追溯、调用身份/参数不变、下一轮不重复注入过渡说明、普通文本及纯工具响应、多工具顺序、实质答复/确认/非法JSON/缺失或重复ID/截断/过滤/写操作/usage缺失等均有定向验证。原始usage原样保留；现有费用账本仅按其总输出结算，reasoning未重复计费；Mock费用为离线算术验证，不是新增实际费用。

最终结果：`validation/phase1_comparison/20261107T145851475509+0800_22e421746e/test_summary.json` 与同目录 `mixed_replay_summary.json`。首轮材料保存在 `20261107T145811700236+0800_779a334898`。

私有历史响应不进入Git；克隆后缺少该本地文件时，对应真实捕获回放用例明确跳过，合成用例仍可运行。已有Phase1-07汇总和4次真实请求账本保持不变，预算未恢复、授权未重新消费。

仓库要求的 `make check-all` 因本机没有 `make` 未能执行；未安装依赖。定向测试通过，不等于全量测试或真实实验已验证。
