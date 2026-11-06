# 可公开的离线测试素材

这些文件不是收费授权、完整真实请求账本或新实验成绩。原始历史文件仍只保留在本机历史目录。

- `full_history_fc0055dc.json`：修改前上游普通 LLMAgent 的独立冻结请求对照，原文件及测试SHA不变。
- `train34_replay/`：早期普通Agent train34运行的允许字段摘录；保留Agent可见请求输入、可见轨迹、工具调用关联及回放所需模型文本。`source.json`记录原文件SHA及供应商usage数字。请求头、密钥、真实响应ID、隐藏User系统情境、收费授权和账本状态均未纳入。模型响应的外层ID/时间是明确的fixture标记，工具调用ID为维持配对保留。实体姓名、地址及商品来自上游虚构Retail基准。
- `truncation_responses.json`：两条历史截断响应的协议字段与usage摘录；供应商ID、时间、推理原文和请求元数据不保留。只能验证离线协议处理，不能证明新参数在供应商端生效。
- `structured_obstruction.json`：过去Mock执行中Agent可见的状态和原预算请求边界，保留事实版本、来源和完整可见交互；不是Task金标或隐藏情境。
- `legacy_metrics_excerpt.json`：旧普通Agent真实运行的指标数字与评分摘录，用于验证结果读取分类。测试临时生成的ledger明确标注`synthetic_no_http`，此次HTTP计数为0；历史Token数字不计作本轮消耗。
- `structured_demo_excerpt.json`：过去合成Mock执行的结构化事实、来源及裁剪诊断，用于离线Demo读取。
- `historical_status_excerpt.json`：旧停止状态、次数及不可重启标记的最小摘录；不含`approved`、有效授权ID或授权签名，不能用于模型调用。

运行配置由`tests/portable_support.py`结合当前仓库身份生成，始终未批准；拒绝旧授权测试使用明确无效的合成文件。所有原冻结预期和业务断言保留，不按新输出重新生成expected。
