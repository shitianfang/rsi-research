"""Generate demo-poster.html from demo.json. Nothing is hand-typed: every
number and every poster on the page comes out of the simulation."""
import json, os

d = json.load(open(os.path.join(os.path.dirname(__file__), "demo.json")))
H, C = d["honest"]["rounds"], d["corrupted"]["rounds"]
CF = d["corrupted"]["corrupt_from"]
N = len(H)
caps = sorted(d["honest"]["posters"], key=int)


def at(rs, r):
    return rs[r - 1]


def line(rows, key, color, dash=""):
    pts = " ".join(f"{40 + (r['round'] - 1) / (N - 1) * 560:.1f},"
                   f"{150 - r[key] * 118:.1f}" for r in rows)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" {dash}/>')


def chart(series, caption_marks=True):
    g = ['<svg viewBox="0 0 640 190" role="img" aria-label="迭代曲线">']
    g.append('<g font-family="\'IBM Plex Mono\',monospace" font-size="10.5" fill="currentColor">')
    for v, lab in ((0, "0"), (0.5, "0.5"), (1, "1.0")):
        y = 150 - v * 118
        g.append(f'<line x1="40" y1="{y}" x2="600" y2="{y}" stroke="currentColor" '
                 f'stroke-opacity="0.16" stroke-width="1"/>')
        g.append(f'<text x="32" y="{y + 3.5}" text-anchor="end" fill-opacity="0.55">{lab}</text>')
    if caption_marks:
        x = 40 + (CF - 1) / (N - 1) * 560
        g.append(f'<line x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="150" stroke="#9C4221" '
                 f'stroke-width="1" stroke-dasharray="4 3"/>')
        g.append(f'<text x="{x + 6:.1f}" y="20" fill="#9C4221" font-size="10.5">'
                 f'第 {CF} 轮：裁判被污染</text>')
    for rows, key, color, dash, lab, ly in series:
        g.append(line(rows, key, color, dash))
        g.append(f'<text x="606" y="{150 - at(rows, N)[key] * 118 + 3.5:.1f}" '
                 f'fill="{color}" font-size="10.5">{lab}</text>')
    g.append(f'<text x="40" y="168" fill-opacity="0.55">第 1 轮</text>')
    g.append(f'<text x="600" y="168" text-anchor="end" fill-opacity="0.55">第 {N} 轮</text>')
    g.append("</g></svg>")
    return "".join(g)


def posters(run, rounds, labels):
    cells = []
    for r, lab in zip(rounds, labels):
        row = at(d[run]["rounds"], int(r))
        cells.append(
            f'<div class="pcell"><div class="pframe">{d[run]["posters"][r]}</div>'
            f'<div class="pmeta"><b>第 {r} 轮</b><span>{lab}</span>'
            f'<span class="pnum">真实 {row["true"]:.2f}　报告 {row["reported"]:.2f}</span></div></div>')
    return f'<div class="prow">{"".join(cells)}</div>'


CSS = """
:root{--ground:#F5F6F4;--panel:#FFFFFF;--sunk:#EDEFEC;--ink:#171A18;--muted:#5C625E;
--faint:#868C88;--rule:#DCE0DB;--rule-soft:#E9ECE8;--accent:#14534A;--accent-wash:#E3EDEA;
--alarm:#9C4221;--alarm-wash:#F6E7DF;--calm:#3F6B8C;--calm-wash:#E4ECF2;
--serif:"Noto Serif SC",Georgia,serif;--sans:"Noto Sans SC",-apple-system,"PingFang SC",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#131614;--panel:#1A1E1B;--sunk:#20241F;--ink:#E6E9E5;--muted:#A0A6A1;--faint:#7C837D;
--rule:#2C312D;--rule-soft:#232722;--accent:#5FBFAE;--accent-wash:#18302C;
--alarm:#D98A5F;--alarm-wash:#33221A;--calm:#7FA8C9;--calm-wash:#1C2833}}
:root[data-theme="dark"]{--ground:#131614;--panel:#1A1E1B;--sunk:#20241F;--ink:#E6E9E5;
--muted:#A0A6A1;--faint:#7C837D;--rule:#2C312D;--rule-soft:#232722;--accent:#5FBFAE;
--accent-wash:#18302C;--alarm:#D98A5F;--alarm-wash:#33221A;--calm:#7FA8C9;--calm-wash:#1C2833}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
font-size:16.5px;line-height:1.85}
.wrap{max-width:44rem;margin:0 auto;padding:0 1.35rem}
.wide{max-width:58rem;margin:0 auto;padding:0 1.35rem}
.scale{height:11px;margin:0;border:0;background-image:
repeating-linear-gradient(to right,var(--rule) 0 1px,transparent 1px 12px),
linear-gradient(to bottom,transparent 0 10px,var(--rule) 10px 11px);background-repeat:repeat-x}
header.top{padding:3.4rem 0 1.6rem}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);margin:0 0 1.1rem}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2.1rem,6vw,3.1rem);line-height:1.24;
margin:0 0 1rem;letter-spacing:-.01em;text-wrap:balance}
.stand{font-size:1.06rem;color:var(--muted);margin:0;max-width:34rem}
.byline{font-family:var(--mono);font-size:.74rem;color:var(--faint);margin:1.6rem 0 0}
main{padding:1.2rem 0 4rem}section{padding:2.6rem 0 .4rem}
h2{font-family:var(--serif);font-weight:600;font-size:1.42rem;line-height:1.45;margin:0 0 .3rem;
text-wrap:balance}
.h2num{font-family:var(--mono);font-size:.74rem;color:var(--accent);letter-spacing:.14em;
display:block;margin-bottom:.45rem}
h3{font-size:1rem;font-weight:700;margin:2rem 0 .3rem}
p{margin:.85rem 0}em{font-style:normal;color:var(--accent);font-weight:500}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:.08em .32em;border-radius:3px}
ul{margin:.8rem 0;padding-left:1.15rem}li{margin:.45rem 0}li::marker{color:var(--faint)}
.lead-claim{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--accent);
border-radius:2px;padding:1.15rem 1.35rem;margin:1.6rem 0;font-size:1.02rem}
.lead-claim p{margin:0}
.warn{background:var(--alarm-wash);border:1px solid var(--rule);border-left:3px solid var(--alarm);
border-radius:2px;padding:1.1rem 1.3rem;margin:1.6rem 0}.warn p{margin:0}
figure{margin:2rem 0;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
padding:1.4rem 1.2rem 1rem}
figure svg{display:block;width:100%;height:auto}
figcaption{font-size:.87rem;color:var(--muted);margin-top:1rem;padding-top:.8rem;
border-top:1px solid var(--rule-soft);line-height:1.7}
.prow{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.6rem 0}
.pframe{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:.5rem}
.pframe svg{width:100%;height:auto;display:block}
.pmeta{margin-top:.55rem;font-family:var(--mono);font-size:.74rem;color:var(--muted);
display:flex;flex-direction:column;gap:.12rem}
.pmeta b{color:var(--ink);font-weight:500}
.pnum{color:var(--faint)}
.tablewrap{overflow-x:auto;margin:1.5rem 0;border:1px solid var(--rule);border-radius:3px;
background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.87rem;line-height:1.55}
th,td{text-align:left;padding:.62rem .8rem;border-bottom:1px solid var(--rule-soft)}
thead th{background:var(--sunk);font-family:var(--mono);font-weight:500;font-size:.72rem;
letter-spacing:.09em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.what{color:var(--muted)}tr.dead td{background:var(--alarm-wash)}tr.hi td{background:var(--calm-wash)}
.note{background:var(--sunk);border:1px solid var(--rule);border-radius:3px;padding:.95rem 1.15rem;
margin:1.5rem 0;font-size:.93rem;color:var(--muted)}.note strong{color:var(--ink)}
footer{border-top:1px solid var(--rule);margin-top:3.5rem;padding:1.8rem 0 3.5rem;font-size:.83rem;
color:var(--faint);font-family:var(--mono)}
@media (max-width:680px){body{font-size:16px}.prow{grid-template-columns:1fr;gap:1.6rem}}
"""

score_chart = chart([
    (H, "true", "#3F6B8C", "", "诚实", 0),
    (C, "true", "#9C4221", "", "污染·真实", 0),
    (C, "reported", "#9C4221", 'stroke-dasharray="5 3"', "污染·报告", 0),
])
ruler_chart = chart([
    (H, "resolution", "#3F6B8C", "", "诚实", 0),
    (C, "resolution", "#9C4221", "", "污染", 0),
])

html = f"""<title>海报会变好，也会被骗</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600&display=swap">
<style>{CSS}</style>

<header class="top"><div class="wrap">
<p class="eyebrow">演示 · 同一套机制，换成看得见的东西</p>
<h1>海报会变好，也会被骗</h1>
<p class="stand">同一台机器跑两遍。一遍裁判是诚实的，海报从草稿一轮轮变好。另一遍裁判在第 {CF} 轮被污染——分数一路涨到满分，海报却肉眼可见地烂掉。我们的尺子能不能当场抓住它？</p>
<p class="byline">2026-09-16 · rsi-research · 页面上每个数字和每张海报都由 demo-poster/poster.py 生成</p>
</div></header>

<hr class="scale">
<main>
<div class="wrap">

<section>
<span class="h2num">01 / 定义</span>
<h2>"好海报"在这里有一个机器能算的定义</h2>
<p>先写下你想让人看到的<strong>阅读顺序</strong>——这就是意图。本次演示用的是最常规的一种：</p>
<div class="lead-claim"><p><code>{' → '.join(d['intent'])}</code>　　标题先入眼，然后副标题，然后"报名"，再是正文，最后页脚。</p></div>
<p>顺序是你自己定的，所以它<strong>天然是真值</strong>，不需要任何人来评判好坏。然后让一个<em>只瞥几眼的观众</em>去看这张海报——限定 5 次注视，他按视觉显著度（字号、对比度、字重、位置）依次看过去，读出一个顺序。他读出来的顺序和你想要的对上几分，就是这张海报的分数。</p>
<p>请注意这个分数里<strong>没有任何审美意见</strong>，全是机械比对；而它衡量的又确实是真品质——<em>在有限注意力下，你想传达的秩序还传不传得到</em>。这正是"限制逼出真品质"的意思：不设注视上限，任何海报都能被读懂，分数全满，信号为零。</p>
</section>

<section>
<span class="h2num">02 / 迭代</span>
<h2>诚实裁判下：从草稿到成品</h2>
<p>起手是一张典型的糟糕初稿——页脚又大又居中，标题小、淡、沉在下面。然后循环开始：随机改一点字号、位置、对比度，谁让观众读对的顺序更多就留下谁。</p>
{posters("honest", caps, ["糟糕的初稿", "爬升中", "成品"])}
<p>第 1 轮分数 <strong>{at(H,1)['true']:.2f}</strong>，第 {N} 轮 <strong>{at(H,N)['true']:.2f}</strong>。看海报本身：标题被放大提亮推到顶部，页脚被压小压暗——机器并不知道什么叫"标题应该大"，它只是发现<em>这样改之后，那个只瞥几眼的观众才会按你要的顺序读</em>。</p>
</section>

<section>
<span class="h2num">03 / 被骗</span>
<h2>裁判被污染之后：分数飙升，海报烂掉</h2>
<p>第二遍完全一样，只有一处不同：从第 {CF} 轮起，裁判有 80% 的概率<strong>根本不看海报</strong>，直接报出"正确答案"。这对应现实中最危险的裁判失效——记住了测试集、或者和出题的一方串通。</p>
{posters("corrupted", caps, ["同一张初稿", f"污染前一轮：好海报", "污染之后"])}
<div class="warn"><p><strong>第 {caps[1]} 轮那张，和上面诚实线的那张是同一张海报</strong>（前 {CF-1} 轮两条线完全相同）。真正的分叉发生在之后：到第 {N} 轮，这条线的<strong>报告分数是 {at(C,N)['reported']:.2f}（满分）</strong>，而它的<strong>真实分数只有 {at(C,N)['true']:.2f}</strong>——比诚实线的 {at(H,N)['true']:.2f} 差了一大截。它报告的分数甚至<em>高于</em>诚实线，而海报实际上被毁了。</p></div>
</section>
</div>

<div class="wide">
<figure>{score_chart}
<figcaption>三条线：诚实线的真实分数（蓝）稳步爬升；污染线的真实分数（红实线）在第 {CF} 轮后崩落并乱走；污染线报告的分数（红虚线）却一路顶到满分。<strong>只盯着报告分数的团队，看到的是一条完美的成功曲线。</strong></figcaption>
</figure>
</div>

<div class="wrap">
<section>
<span class="h2num">04 / 抓住它</span>
<h2>尺子当场抓住了它</h2>
<p>除了给海报打分，同一个裁判每轮还要考一套<strong>冻结不变的标准题</strong>：一组事先准备好的海报对，两张的阅读顺序确实不同。问题只有一个——"这两张读起来顺序一样吗？"</p>
<p>诚实的裁判分得出。被污染的裁判对两张海报都报同一个答案，于是它"分得出"的比例塌掉。<strong>标准题一个像素都没变，塌掉的只可能是裁判。</strong></p>
</section>
</div>

<div class="wide">
<figure>{ruler_chart}
<figcaption>分辨力读数。诚实线（蓝）全程平稳在 {at(H,N)['resolution']:.2f} 上下；污染线（红）在第 {CF} 轮<strong>一轮之内</strong>从 {at(C,CF-1)['resolution']:.2f} 塌到 {at(C,CF)['resolution']:.2f}，此后贴地。这一刻，报告分数还在往上走。</figcaption>
</figure>
</div>

<div class="wrap">
<section>
<h3>一个必须有的细节：先扣掉裁判自己的噪声</h3>
<p>标准题里混进了一部分<strong>两张完全相同</strong>的海报。裁判对这种题本该答"一样"；它在这里答错的比例，就是它自己的随机噪声底。</p>
<div class="tablewrap"><table>
<thead><tr><th>第 {N} 轮</th><th>真实对答"不一样"</th><th>相同对答"不一样"（噪声底）</th><th>分辨力＝前者−后者</th></tr></thead>
<tbody>
<tr class="hi"><td class="what">诚实裁判</td><td class="num">{at(H,N)['hit']:.2f}</td><td class="num">{at(H,N)['floor']:.2f}</td><td class="num">{at(H,N)['resolution']:.2f}</td></tr>
<tr class="dead"><td class="what">污染裁判</td><td class="num">{at(C,N)['hit']:.2f}</td><td class="num">{at(C,N)['floor']:.2f}</td><td class="num">{at(C,N)['resolution']:.2f}</td></tr>
</tbody></table></div>
<p>看污染那一行：它对真实对还有 {at(C,N)['hit']:.0%} 答"不一样"——<strong>单看这个数，会以为它还有一部分功能</strong>。但它对<em>完全相同</em>的两张海报也有 {at(C,N)['floor']:.0%} 答"不一样"。两者一减，分辨力是 {at(C,N)['resolution']:.2f}：它根本没在分辨，只是在乱答。<strong>不做这一步扣除，这场崩塌是看不见的。</strong></p>
</section>

<section>
<span class="h2num">05 / 边界</span>
<h2>这个演示是什么、不是什么</h2>
<ul>
<li><strong>这是演示，不是新的实验结论。</strong>真正的验证跑在领域通用的设定上（合成程序结构、以及真实 LLM 裁判加公开语料），判据都是事先写死的。这一页只是把同一套机制换成看得见的东西。</li>
<li><strong>海报不是研究域。</strong>设计是动机来源；把机制绑死在海报上会失去通用性，这条线要的是任何域都能用的机制。</li>
<li><strong>观众是模型，不是人。</strong>这里的"只瞥几眼的观众"是一个显著度模型，不是真人眼动。它让机制可复现、可核对，但它不等于真实观众。</li>
<li><strong>它测的是传达，不是美。</strong>这套机制判断不了一张海报美不美，它只判断你想传达的秩序在有限注意力下还传不传得到。</li>
</ul>
</section>
</div>
</main>

<hr class="scale">
<footer><div class="wrap">rsi-research · demo-poster/poster.py 可复现（0.5 秒，纯 numpy，无网络）</div></footer>
"""

out = os.path.join(os.path.dirname(__file__), "..", "demo-poster.html")
open(out, "w", encoding="utf-8").write(html)
print(f"wrote {os.path.abspath(out)} ({len(html)} bytes)")
