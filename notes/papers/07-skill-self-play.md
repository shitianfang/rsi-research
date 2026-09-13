# 07. Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills
- 机构: 阿里巴巴 Qwen 大模型应用团队 + 香港中文大学 + 中国人民大学 + 中山大学 + 北京大学 + ETH Zürich | 日期: 2026-07 | arXiv: https://arxiv.org/abs/2607.22529 | 代码: https://github.com/Qwen-Applications/skill-self-play

## 一句话
纯粹的 self-play 出题很容易退化成"出一堆相似的题"。这篇给出题人加了一个**会进化的技能库**当结构骨架：出题时按技能采样，出完题再根据结果精炼/淘汰/新增技能——结果在弱模型上带来了 +42.9 分的极端提升（把一个连 schema 都写不对的模型救回来了）。

## 机制拆解
三个角色：**Proposer**（出题）、**Solver**（解题）、**Controller**（管技能库 𝒮）。

**一轮（共 5 轮）**：
1. 从技能库采一个技能，走两条流生成候选题：**skill-routed**（条件在该技能上）和 **exploration**（不条件）。
2. **验证 + 算 reward**：环境做三件事——schema 合规检查、技能自带的专用 validator（技能元组里的 ν）、以及 **probe consistency**（当前 solver 采 K 次 rollout，多数答案必须和参考答案一致）。最后一条是防止出题人生成"看起来有答案其实答案错了"的题。
3. **按前沿难度排序**，构造 50% skill-routed + 50% exploration 的混合课程。
4. Proposer 用 GRPO 更新；Solver 在课程上更新。
5. **技能库演化**：精炼已有技能；低于 γ_prune 的淘汰；exploration 流里高于 γ_induce 的**归纳出新技能**。

**改进信号从哪来**：
- Proposer 的 reward = 合法性二值闸 × **前沿难度 `1 − 2|v_solve − 0.5|`**（即 solver 胜率越接近 50% 分越高）。这是个动态前沿——solver 变强，同一批题的 reward 自动下降。
- Solver 的 reward = 环境验证（精确正确性 + 格式合规）。**机器可验证，不靠 LLM 裁判**。

**闭环怎么合上**：生成失败的执行反馈触发技能精炼；合法的前沿题进课程；solver 提升改变难度地形，逼 proposer 出更难的；exploration 流不断归纳新技能扩张前沿。**技能库就是那个"跨轮的记忆"，也是它和普通 self-play 的唯一区别**。

## 证据与评估
- **任务/模型**：工具调用（API-Bank L1–L3、BFCL 的 JS/Python/Java/live）、逻辑推理（ZebraLogic 四档规模）；backbone 覆盖 Qwen3-4B/8B、Ministral-3-8B/14B、Granite-4.1-3B——**跨三个模型家族**，不是只在自家模型上刷。
- **工具调用**：Qwen3-4B **+6.5**；Ministral-3-8B **+42.9**（论文说是从 schema 失败里"救"回来的）。
- **最关键的对照**：**Unguided SP（无技能引导的 self-play）只有 +0.3**。这一条直接证明"结构化的技能库"才是增益来源，而不是 self-play 本身。
- **逻辑推理**：Qwen3-4B **+5.8**；Ministral-3-14B **+12.0**（小规模档上 +35 以上）。
- 规模：5 轮迭代，工具调用每轮 8000 题，推理每轮 1920 题。
- 代码开源。

## 真正难在哪
- **技能元组里的专用 validator ν**：每个技能要配一个能自动判对错的验证器。工具调用/逻辑推理之所以能做，是因为这类任务天然可验证。**换到没有 validator 的域，整套机制就失效**——这是隐藏的最大门槛。
- **probe consistency 很贵**：每道候选题都要 solver 采 K 次 rollout 才能确认答案可靠。出题成本是解题成本的 K 倍。
- RL infra（GRPO、多模型家族、5 轮 × 8000 题）——中等偏上的算力。
- 阿里 Qwen 团队的优势在于"能同时在五个 backbone 上跑完整实验"，这是资源而非想法上的护城河。

## 我们的机会
- 论文自认弱点非常具体：**"pure self-play requires a minimal capability threshold to bootstrap"**——在 Large / X-Large 规模上，初始弱的模型始终起不来。技能的精炼与归纳全靠 backbone 自己，没有外部更强的老师。**"能力门槛之下怎么冷启动"是明确的空白**。
- 另一个没做的对照：技能库的**三个操作（精炼 / 淘汰 / 归纳）各自值多少分**没有拆开消融。+42.9 那个极端值到底来自哪一个？这个消融便宜且结论有用。
- 小团队切口：**把技能库这个骨架移植到一个没有天然 validator 的域**，用 05 号论文的"最小锚点 + 确定性检测算子"补上验证缺口。这是把两篇的强项接起来，而且是各自都承认的空白处。

## 方向启示
Skill Self-Play 证明了一件重要的事：**self-play 的收益不来自对抗，而来自"有结构的课程"**（unguided SP 只有 +0.3 是铁证）。这意味着"技能库/记忆/经验"这类工作真正的价值不是"让 agent 记住东西"，而是**给自我生成的数据提供一个不会坍缩的组织结构**。

方向上：技能库本身已经是超级红海（同期至少二三十篇"SkillXXX"）。有差异化的问题是上游的那个——**在没有 validator 的域里，前沿难度怎么估？** 因为一旦估不了难度，整个课程机制就没了。
