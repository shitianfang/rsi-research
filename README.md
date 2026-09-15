# RSI 阅读指南 / Recursive Self-Improvement Reading Guide

> A decision-oriented reading guide to recursive self-improvement (RSI) and self-evolving LLM agents.
> 24 structured notes — 16 recent papers (2026-06 to 2026-09) and 8 frontier-lab technical reports (2026-03 to 2026-09) — plus one synthesis page that ranks candidate research directions.
> Everything is written in Simplified Chinese. Open `index.html` locally or browse it on GitHub Pages.

**快照日期：2026-09-13**

---

## 这是什么

一份**为了做方向决策而做的阅读笔记**，不是文献综述。

材料是 24 篇：16 篇近半年的 RSI / 自演化 agent 论文，8 篇前沿实验室技术报告。每一篇按同一套模板拆解，最后交叉阅读得出一页判断，回答三个问题：

1. **什么是真正难的、别人无法轻易做到的**（护城河盘点：算力 / 数据与环境 / RL infra / 评估器 / 思想）
2. **什么是我们能重新做出来并且效果更好的**（具体切入口，逐条引用论文自认的弱点）
3. **3–5 个候选方向的排序**，每个说清：为什么难、为什么有根本性差异化、第一步做什么、最大风险是什么

它刻意不做两件事：不给"该领域有 N 种流派"这类分类学，不复述摘要。只关心——**这篇的护城河在哪、它自己承认的弱点是什么、它对选方向意味着什么**。

## 筛选标准

- **时效**：论文以近半年为界，最好三个月内。收录的 16 篇 arXiv v1 全部在 **2026-06-13 之后**，逐篇核对过日期与作者单位。技术报告窗口放宽到六个月（2026-03-13 起）。
- **机构**：前沿实验室（OpenAI / Anthropic / Google DeepMind / Meta / DeepSeek / Moonshot / 阿里 Qwen / 字节 Seed 等）；美国 top-30 高校；国内清北华五；港三新二。
- **公司技术报告权重高**——它们写的是真正跑通的生产系统，护城河与取舍写得比论文诚实，且常附带可下载的权重。
- **例外与剔除都记录在案**：两个机构例外（剑桥的 Red Queen Gödel Machine、AWS 的 Who Grades the Grader）在各自条目顶部标注了纳入理由；被剔除的高关注度候选及原因列在论文页的"收录说明"里。

## 阅读的透镜

每篇笔记固定回答同样几个问题，这样 24 篇才能横向比较：

| 段落 | 它在回答什么 |
|---|---|
| 一句话 | 如果只能记住一件事，是哪件 |
| 机制拆解 | 什么被修改、谁来评估、改进信号从哪来、循环怎么闭合 |
| 证据与评估 | 数字、对照组、开放程度，以及哪些数字要打折看 |
| 工程护城河（仅报告） | 抄不走的到底是什么：钱、数据、系统，还是组织 |
| 真正难在哪 | 门槛的类型——区分"要花钱"和"要想清楚" |
| 我们的机会 | 作者自认的弱点，逐条转成可做的切口 |
| 方向启示 | 这篇对"选一个难方向"意味着什么 |

重心在最后三段。绝大多数论文会把自己的局限写在末节或附录里，那里往往是最有价值的信息。

## 建议的阅读顺序

1. **`direction.html` — 方向判断**：先看结论。护城河的五类分法、一张"可重做清单"、5 个候选方向的排序与风险。其余页面都是它的证据。
2. **`reports.html#overview` — 报告综述**（十六条判断）：八家实验室在做什么、谁的 playbook 对小团队最可学。
3. **`papers.html#overview` — 论文综述**（地形 / 红海 vs 无人区 / 三个瓶颈）：学术侧的分层，以及哪些方向已经挤满了人。
4. **按需精读**：从两页顶部的索引表里挑。每条都有稳定锚点（`papers.html#p01`…`#p16`，`reports.html#r01`…`#r08`），方向判断页里的每个引用都直接跳到对应条目。

## 结构

```
index.html        首页：这是什么、筛选规则、阅读透镜、地形速览、注意事项
direction.html    方向判断（核心产出）
papers.html       16 篇论文笔记 + 综述 + 索引表，锚点 #p01…#p16
reports.html      8 篇技术报告笔记 + 综述 + 索引表，锚点 #r01…#r08
notes/
  papers/         16 篇论文笔记的 Markdown 原文 + _overview.md
  reports/        8 篇报告笔记的 Markdown 原文 + _overview.md
  directions/     原创方向判断（非论文笔记），01 = 验证器应该是度量而不是打分函数
README.md
LICENSE           CC BY 4.0
```

HTML 页面是从 `notes/papers/` 与 `notes/reports/` 渲染的，内容与之一致。保留 Markdown 原文是为了方便用编辑器检索或做二次加工。

`notes/directions/` 是快照之后新增的原创判断，**尚未渲染进 HTML**——它不是对某篇材料的笔记，而是横向读完之后的结论，会随后续轮次增补。

## 怎么浏览

- **本地**：克隆后直接用浏览器打开 `index.html`。四个页面完全自包含——内联 CSS、无外部 CDN、无字体请求、无 JS 框架，`file://` 下离线可用。
- **GitHub Pages**：仓库 Settings → Pages → 选择分支根目录即可，无需构建步骤。
- 页面适配深色模式，宽表格独立横向滚动，移动端可读。

## 几条必须先讲的提醒

- **Red Queen Gödel Machine（论文 02）**：摘要与数字是从 arXiv 摘要页逐字核对的，但 epoch 结构、anchor 替换流程等**机制细节转引自第三方索引站，未从原文正文确认**。引用前请打开 PDF 复核。
- 论文 04 / 06 / 09 / 14 / 16 正文中未给出代码仓库链接；其余 11 篇的代码/项目链接已逐条验证可访问。
- **DeepSeek-V4（报告 04）**：笔记记载的日期与所给 arXiv 编号前缀不一致，条目内已标注，引用发表日期前请自行核对。
- 部分亮眼数字有前提条件，笔记里都写明了。例如 Prime Agent 的 ARC-AGI-3 `30% → 95.5%` 是特定 harness 评测设置（RHAE）下的 Best@1，不是标准榜单；EdgeBench 的"学习速度每三个月翻倍"基于 18 任务切片、单条模型线、两点外推，方向性可信但倍数要打折。
- 方向判断页的排序是**一家之言**，该页末尾专门列了"这份判断在什么情况下是错的"，便于日后检查而不是重写。

## 透明度声明

本仓库的检索、笔记与方向判断由 **Claude（Anthropic）完成——由一个 Fable 会话调度多个 Opus 子 agent 分头研究与撰写**，筛选标准（时间窗口、机构白名单、报告优先）由仓库所有者设定。所有数字与链接以原论文/报告为准；发现错误欢迎提 issue。

## 许可

内容以 **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** 发布，完整法律文本见 [`LICENSE`](LICENSE)。你可以自由分享和改编，包括商业用途，只需署名并注明是否做了修改。被引用的论文与技术报告各自的版权归原作者所有。
