# 12. HELIX: Model-Harness Co-evolution for Recursive Self-Improvement
- 机构: 香港大学（HKUDS，Chao Huang 组）| 日期: 2026-08 | arXiv: https://arxiv.org/abs/2608.13951 | 代码: https://github.com/HKUDS/HELIX

## 一句话
提出一个关键论点：harness 不是中性的包装，它同时决定"今天怎么执行"和"明天能拿到什么训练数据"——所以模型和 harness 应该 build→update→rebuild 地交替演化。**但这篇自己没跑完这个循环**，它只做到了"把训练数据物化出来"。

## 机制拆解
**核心论点**：`harness 塑造轨迹 (φ_t → τ_t)` → `被验证的轨迹塑造下一个模型 (τ_t → θ_{t+1})` → `更新后的能力画像决定新的 harness (θ_{t+1} → φ_{t+1})`。三条有向依赖构成递归。

**harness 被拆成八个维度**：shell、session/hooks、config、prompt、tools、turn loop、acceptance、policy。组合成 typed ports / atoms / recipes / product shells，枚举出 **4⁵=1024 个耦合配方**（acceptance 独立选时是 4⁶=4096）。这个分解是全文最有复用价值的产出——它把"改 harness"从玄学变成了可枚举的组合空间。

**三阶段**：
- **Build**：对固定模型，构造并评测多个**可溯源的** harness 候选，跑在匹配任务上。
- **Update**：把**verified sibling trajectories**（同一任务不同 harness 产生的兄弟轨迹）转成学习记录，微调模型。
- **Rebuild**：为更新后的模型重新演化 harness，因为新的能力画像可能偏好不同的运行时设计。

**谁来评估**：一个 verifier V 观察 任务 / 轨迹 / 工作区 diff / 测试证据 / 策略证据，产出结果标签，区分 **resolved、target-miss、regression、no-action、policy violation、patch-noise** 六类。用 trace-strict 标准（必须观察到规范化的 bash 执行并有通过输出），交叉验证时用 SWE-bench Verified 官方评测器。**六类标签比"过/不过"信息量大得多——这是它给学习信号加的最有用的一层结构**。

**改进信号从哪来**：同一批 rollout **一鱼两吃**——执行信号用来选今天最好的固定 harness，学习信号（成功、回归、险些成功、替代解法）结构化成 SFT / critic / filter / preference 四类记录。作者原话："The same harness interventions that change execution today structure the learning signal available tomorrow."

## 证据与评估
- **模型/任务**：MiniMax-M2.7-highspeed，LiveCodeBench（100 题）+ SWE-bench Verified（55 实例）。
- **固定 harness 表现**：π 基线 50/100 (LCB)、44/55 (SWE)；最佳演化候选 52/100（+4.0%）、46/55。**单个 harness 的提升很小**。
- **真正的发现是 portfolio**：完整候选组合覆盖 **79/100**（LCB）、49/55（SWE），相对覆盖增益 **58.0%**。也就是说——**不同 harness 解开的是不同的题，而不是同一批题解得更好**。这是全文最有价值的实证。
- **数据物化**：200 槽切片产出 438 条已验证记录（63 clean SFT、56 需清理 SFT、52 critic 负例、113 filter 样本、154 偏好对）。
- **⚠️ 循环没闭合**：作者明说 "This paper materializes model-update data but does not train an updated model"——**只评测了单轮，多轮改进没有实验验证**。
- 代码开源。

## 真正难在哪
- **verifier 的工程**：六类结果标签 + trace-strict 判定（要真的看到 bash 执行和通过输出，而不是信 agent 自述）——这是防 agent 自我报告造假的核心，做扎实很费劲。
- **1024–4096 个 harness 配方的评测预算**：每个配方都要在任务集上跑，这是这篇真正的花销所在。
- 下一步（真的训模型并 rebuild）需要 RL/SFT infra + 更多数据，作者因"data scaling constraints"没做。

## 我们的机会
论文自认的限制单子很长，而且很诚实，每条都是切口：
- **循环没闭**——最大的空白。**第一个真正跑完 build→update→rebuild 两轮以上并给出曲线的人，就拿到了这篇想要而没拿到的结论**。代码开源，数据管线现成，这是一个"接着往下做"成本很低的机会。
- **"cannot tell which individual component caused the change"**——每个演化 harness 一次改多个组件，归因不了。**八维度 × 四取值的组合空间正好适合做因子实验设计**，这是标准统计方法能解决、但没人做的事。
- **portfolio 覆盖 79/100 是靠事后结果选的（oracle coverage），不可部署**——**"在不知道答案时怎么选 harness"（路由问题）完全空白**，而这是把 58% 覆盖增益变成真实收益的唯一途径。这个问题界定清晰、可训练、价值直接。
- 评测面窄：100 道 AtCoder 题（取前 100 而非随机抽样）、只用公开测试、只测编码 agent。

## 方向启示
HELIX 最有价值的不是它做成了什么，而是它**无意中证明了一件事：harness 演化的收益主要是"覆盖不同的题"，而不是"把同样的题做得更好"**（52/100 vs 组合覆盖 79/100）。

这把整个 harness 演化方向的叙事改了：如果收益来自多样性而非单点质量，那么
1. **最优策略不是"演化出一个最好的 harness"，而是"维护一个互补的 harness 组合 + 一个好的路由器"**——这是完全不同的技术路线，目前没人在做。
2. 这也解释了为什么 04/09 号那些"演化出单个更好 harness"的工作增益会见顶。

对选方向：**"agent 能力的组合与路由"比"agent 的自我改进"更可能是下一个真问题**，而且它和 quality-diversity（质量-多样性搜索）那一支天然接得上。
