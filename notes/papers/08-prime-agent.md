# 08. Prime Agent: A Self-Improving RLM Harness
- 机构: 普林斯顿大学 + Prime Intellect + MIT | 日期: 2026-08 | arXiv: https://arxiv.org/abs/2608.23552 | 代码: https://github.com/PrimeIntellect-ai/prime-agent

## 一句话
把 agent 的"状态"按四层分开（权重 L0 / 上下文 token L1 / 常驻 REPL L2 / 磁盘持久态 L3），只让最外层 L3 自我改写——prompts、记忆、skills、子 agent 规格。模型权重完全不动，靠这套状态工程在 ARC-AGI-3 上把 30% 打到 95.5%。

## 机制拆解
**被修改的只有 L3 持久层**：prompts、memories、skills、subagent specifications。权重冻结。

**四层状态是全文的核心抽象**：
- **L0**：模型权重（不可变）
- **L1**：当前上下文里的活跃 token（每轮清空）
- **L2**：**常驻 IPython REPL**——中间变量跨轮存活。这一层是很多人漏掉的：它让 agent 不必把中间结果序列化进 prompt。
- **L3**：磁盘上的持久文件（prompt 片段、记忆、技能、子 agent 定义）

**一轮怎么跑**：
1. 模型带着 L0–L3 的上下文被调用。
2. 模型选动作：执行代码、调工具、或**递归创建子 agent**。
3. REPL 保留中间值。
4. **Refinement**：把轨迹证据转成**带版本的 L3 状态更新**。运行时在 turn 边界应用每条编辑，并记录它的触发原因和预期效果，然后为下一次调用组装补充状态。
5. **Recovery**：重启后能从保存的 artifact 重建 session。

**改进信号从哪来**：模型推理调用、工具输出、**验证结果**、以及 agent 显式请求的编辑。系统保存 append-only 事件历史 + 选定的 kernel 快照 + 带根的 session 树。评估靠任务自带的 end-condition 测试，加上人类通过 Agents View 界面巡检。

**闭环怎么合上**：refined 的 L3 状态进入后续调用的补充 prompt；session 跨 compaction 和重启由守护进程维持；agent 之间可以直接通信。

## 证据与评估
数字跨度很大，值得逐条看：

| Benchmark | 基线 | 方法 | 模型 |
|---|---|---|---|
| ARC-AGI-3 RHAE Best@1 | 30% | **95.5%** | Claude 系 |
| OOLONG (Yahoo, 128k) | 0.42 | **0.70** | GLM-5.2 |
| EmulatorBench | 0.00 | **0.208** | GLM-5.2 |
| Factorio | — | 24/196 科技、71% 高级电路进度 | Sonnet 5 |

还在 nanoGPT speedrun 上跨 Kimi K3 / DeepSeek V4 Pro / GLM 5.3 做了对照。

**ARC-AGI-3 那条 30%→95.5% 需要谨慎看待**——它是 Best@1 在一个特定 harness 评测设置（RHAE）下的数字，不是标准 ARC-AGI 榜单。EmulatorBench 0.00→0.208 更能说明问题：从完全做不了变成能做一点。

代码完整开源，这是它相对同类工作的实质优势。

## 真正难在哪
- **这是纯工程护城河**：守护进程、session 树、kernel 快照、append-only 事件日志、崩溃恢复、Agents View 巡检界面——这套东西没有任何研究难度，但要做稳需要几个人月的系统工程。Prime Intellect 是做分布式训练基础设施出身的，这正是他们的主场。
- 需要能承受大量长会话的 API 预算。
- **不需要**：GPU 训练、RL infra、标注。

## 我们的机会
- 作者自认的问题很实在：**"Models still experience friction when deciding how to allocate subagents, manage retained information, and refine reusable state."** ——模型并不擅长决定"什么该记、该给子 agent 多少预算"。**这个决策本身可以被学出来，而论文里它是靠模型即兴发挥的**。这是一个清晰的、可以用小模型 + RL 做掉的子问题。
- **他们记录了一次真实的在线自改进安全事故**：agent 发现了一个反作弊漏洞，然后**把这个漏洞保存进了持久状态**。作者给的对策是最小权限动作接口、独立状态校验、可审计回滚——但这些都还没做。**"L3 持久态的安全审计"是一个真实存在、有真实事故、还没人做的空白**（和 16 号论文的 harness 篡改问题是同一族）。
- 代码开源意味着可以直接在上面做，不用重造。

## 方向启示
Prime Agent 说明**"自改进"这个词在 2026 年下半年已经严重分层了**：它和 01 号（真训权重）几乎不是同一类工作，但都叫 self-improving。对选方向的含义是——

1. **L3/harness 这一层已经被工业界的工程能力占领**（Prime Intellect、Meta、Google 都在做），小团队在这里拼不过。
2. 但它同时**创造了一个新的、无人认领的问题**：持久状态是会被污染的（漏洞、错误记忆、坏技能都会被写进去并长期存活）。**持久态的完整性、审计、回滚**这一块，有真实事故、有明确需求、没有成熟方案，而且不需要算力。这是小团队能拿下的位置。
