# 09. Procedural Graphs: Self-Evolving Execution Structures for LLM Agents
- 机构: Google（含 Sercan Ö. Arık）+ Georgia Tech + 北京大学 | 日期: 2026-09 | arXiv: https://arxiv.org/abs/2609.09153 | 代码: 论文中未给出仓库链接

## 一句话
不让 agent 在不断变长的历史上自由生成动作，而是把"怎么做事"沉淀成一张**有向属性图**（程序性知识），然后离线迭代地增删图的节点和边——每次改动必须在 held-out 验证集上不掉分才被接受。在 24 个"模型 × benchmark"组合里 21 个排第一。

## 机制拆解
**被修改的**：Procedural Graph 本身（节点、边、属性）。模型和权重都不动。

**离线四步循环**：
1. **Diagnostic Rollout**：用当前 solver 在一批训练任务上跑，记录完整轨迹和分数。
2. **Feedback-Driven Mutation**：一个 LLM refiner **对比失败轨迹和成功轨迹**，提出图的拓扑和属性编辑（加/删节点、加/删边）。
3. **Validation Gating**：候选图在 **held-out 验证集**上测；**只有平均任务分不下降才被采纳**。不合法候选在评测前就丢掉。
4. **Rejection Memory**：把被拒的候选存下来，避免反复提同一个坏主意。

**改进信号从哪来**：执行反馈——成功/失败轨迹的对比。refiner 看到的是"哪条路走通了、哪条没走通"，而不是一个标量分。

**闭环怎么合上**：验证分决定是否继续；连续若干轮没有被采纳的更新就停。

**为什么这个设计值得注意**：三个小设计各自堵一个坑——validation gate 堵"改动看起来好实际变差"（这是 harness 演化最常见的假阳性），rejection memory 堵"反复兜圈子"，合法性预筛堵"浪费评测预算"。这三条加起来就是 04 号 RHI 缺的那一半（RHI 只有 LLM 成对偏好，没有 held-out 闸门）。

## 证据与评估
- **benchmark 覆盖面是全文最强的地方**：HotpotQA、MultiChallenge、GDPval、ALFWorld、τ-bench、BFCL v3、EnterpriseArena——从 QA 到具身到工具调用到企业场景。
- **模型**：Claude Sonnet 4.6、Gemini 3.1 Pro、Gemini 3.5 Flash、Grok 4.1 Fast——**跨厂商，不是只在自家 Gemini 上刷**。
- **结果**：24 个"模型 × benchmark"设置里 21 个第一/并列第一；对最强 baseline 是 19 胜 2 平 3 负（p = 4.3×10⁻⁴）。
- 单点增益：BFCL v3 + Gemini 3.5 Flash **58% → 67%（+9）**；GDPval + Gemini 3.1 Pro **71.37 → 78.78（+7.41）**。
- **最亮的一条**：EnterpriseArena 生存率 **0% → 85%**（scratch + evolution，20 episode 设置）。从完全做不了到大部分能做。
- 未给代码仓库——Google 的工作常态，但对复现是硬伤。

## 真正难在哪
- **门槛低得反常**：不训练、不要 GPU、只要 API 预算和一个干净的验证协议。这是它的优点也是它的风险（容易被复制）。
- 真正的成本在**评测预算**：每个候选图都要在 held-out 集上跑一遍，24 个设置 × 多轮 × 多 benchmark，API 账单不小。
- Google 的护城河在于"能在四家厂商的旗舰模型上都跑一遍"这种资源，而非算法。

## 我们的机会
作者自认的两条限制：
- **"Guidance increases token use even when it reduces solver steps"**——图带来的指导文本增加了 token 消耗。**成本/收益的帕累托前沿没人画过**，而 04 号 RHI 恰恰在成本侧有漂亮结果（token 不涨、缓存降 33–64%）。把两者合起来做"既涨分又降本的结构演化"是明确的缝隙。
- **"Evaluating transfer across solvers and tool interfaces"**——演化出来的图能不能换个 solver 继续用？完全没测。**"程序性知识的可迁移性"是个干净的、可独立发表的问题**，而且如果答案是"能迁移"，那就意味着可以做一个跨模型复用的程序知识库，这个东西有产品价值。
- 没有开源：做第一个可复现实现有先发优势。

## 方向启示
Procedural Graphs + RHI + HELIX + Prime Agent 四篇放在一起看，结论很清楚：**"不改权重、只改外部结构"这条路在 2026 年已经被彻底验证有效，并且已经挤满了人**（Google、Sakana、HKU、Princeton/Prime Intellect 各一家，同期还有十几篇 "XXXHarness"）。这是标准的红海。

它对"选难方向"的价值在于反向提示：**这一层已经不缺方法，缺的是纪律**。Procedural Graphs 的 validation gate 是这堆工作里唯一一个把"防止假阳性改进"做进机制的——而这恰恰指向下一个真问题：**当演化预算大到一定程度，在 held-out 上做几百次选择本身就会过拟合 held-out**。谁先把这个"演化的多重检验问题"用统计手段解决（而不是再加一个验证集），谁就有了别人绕不开的结果。
