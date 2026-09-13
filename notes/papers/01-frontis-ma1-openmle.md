# 01. Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering (OpenMLE / OpenRSI)
- 机构: 清华大学 + Horizon Research + Frontis.AI | 日期: 2026-07 | arXiv: https://arxiv.org/abs/2607.28568 | 代码: https://github.com/FrontisAI/OpenRSI （另有模型权重 https://huggingface.co/collections/FrontisAI/frontis-ma1 ）

## 一句话
他们把"让 AI 做机器学习工程"这件事整套打通了：自己造了 5758 个可执行的 ML 任务环境，用执行结果训练出一个 35B 的"专门改模型训练代码"的模型，再让这个模型去驱动进化式搜索——这是目前公开工作里少数几个**权重、数据、搜索算子三层都在同一个闭环里**的 RSI 系统，而且权重和代码真开源了。

## 机制拆解
闭环分三段，首尾咬合：

1. **OpenMLE-Gym（造环境）**：从 Kaggle 数据集/比赛里自动构造 5758 个可执行任务，每个任务配一个确定性的 `metric.py` 评分脚本和 Docker 沙箱。这一步是把"ML 工程"变成可验证域的关键。
2. **OpenMLE-ERL（训算子）**：把 agent 的动作抽象成四个算子——Draft（写初版）、Improve（改进）、Debug（修错）、Crossover（把两个方案杂交）。每次算子调用产生的代码在沙箱里真跑，拿到执行状态 + 任务分数 + 日志，作为结构化反馈。这些反馈用来做 SFT + RL（按归一化 reward 加权），训出 Frontis-MA1-35B。
3. **OpenMLE-Evo（搜索）**：训好的模型本身成为进化搜索的"变异引擎"，配合 experience card（记录每个节点的来源与表现）做 experience-guided parent selection，跑长周期搜索。

**闭环怎么合上**：论文的说法是"verified evolutionary transitions supervise the same transformations that search later composes"——搜索过程中被验证有效的那些"父→子"代码变换，正好就是训练算子的监督信号；算子变强 → 搜索轨迹变好 → 产出更好的训练数据 → 下一轮后训练。这是一个 meta-evolutionary loop，改进信号全部来自沙箱真实执行，不依赖 LLM 裁判。

**什么被修改**：模型权重 θ、agent 的搜索脚手架、以及任务环境本身（环境是自动构造并过滤的）。

## 证据与评估
- **MLE-Bench Lite（22 任务，Medal Average）**：MA1-35B 基线 39.39% → 方法 60.61%；加大预算（+Max）到 71.21%。30B 版本 34.85% → 53.03%，说明不是单一 checkpoint 的偶然。
- 横向位置：超过 GPT-5.5+Codex 的 68.18%，逼近 GPT-5.6 Sol 的 72.73%——一个 35B 开源模型打到接近闭源前沿的水平，这是这篇最有说服力的地方。
- **NatureBench Lite（10 任务）跨域迁移**：模型把 Match-SOTA 从 50% 提到 70%；单看 OpenMLE-Evo 这个 harness，把 20% 提到 50%。模型和 harness 的贡献是分开测的。
- **开源程度**：模型权重、数据集、训练/评测代码、沙箱基础设施、harness 全放出。任务数据因授权只放了 5758 中的 1415 个——这是复现时的实际缺口。

## 真正难在哪
- **环境工程是主要护城河，不是算法**。5758 个可执行任务、每个都有确定性评分脚本和 Docker 沙箱，这是几个人月到十几个人月的脏活。授权问题让别人拿不到完整数据，只能拿到 1415 个。
- **RL infra**：在 5758 个需要真跑训练脚本的任务上做 RL，rollout 极慢且成本高（每个 rollout 都是一次真实的 ML 训练）。异步 rollout + 沙箱调度是硬工程。
- **长周期实验**：进化搜索的收益来自"跑得久"，而不是单步聪明。作者自己承认单 GPU 12GB 预算下的结论未必外推到前沿算力配置。
- 反过来说，**算法本身不难**：四个算子 + SFT/RL + 进化搜索，没有新原理。

## 我们的机会
- 论文自认的弱点：迁移只验了 10 个任务、6 个领域，广度不足；算力预算小；模态分组少。
- 小团队的可行切口：**不要重建 5758 个 Kaggle 任务**，而是换一个环境成本低得多、但评分同样确定的域（例如编译器优化、SQL 查询改写、数值库 kernel），复用它的四算子 + experience card 设计。环境少但更深，能把"长周期搜索到底有没有复利"这个它没回答的问题回答掉。
- 另一个切口：它的 Crossover 算子是最独特的部分（把两个方案杂交），但论文没有单独消融 Crossover 的贡献。谁把"LLM 做代码杂交"这件事做出可靠的收益归因，就拿到了一个别人没有的结论。

## 方向启示
这篇证明了 RSI 在**可验证域**里已经不是概念——有确定性评分器 + 足够多环境 + 中等规模模型，就能把开源模型推到接近前沿闭源的水平。所以"做一个 RSI 系统"本身不再是差异化方向；差异化只能来自两处：**(a) 你能造出别人造不出的环境**（成本、许可、领域壁垒），**(b) 你能在没有确定性评分器的域里造出可信信号**。前者是工程战，后者是本文完全没碰的开放问题——也是真正难的那一半。
