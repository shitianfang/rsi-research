"""Build E-R's frozen blocks from HelpSteer2 (public, CC-BY-4.0, third-party).

Nothing here is authored by this project. HelpSteer2 ships 10,160 prompts,
each with two responses carrying HUMAN helpfulness ratings (0-4), so both
the artifacts and the quality labels come from outside the loop. It is also
the corpus used by the closest drift-monitoring paper (Who Drifted: the
System or the Judge?, 2606.15474), which makes our numbers comparable to
the baseline we position against.

What each half supplies:
  battery     frozen pairs judged only SAME/DIFFERENT quality. The human
              score gap gives a GRADED ladder the previous corpus lacked:
                floor  (X, X)        identical text  -- stochastic noise
                d0     gap 0         same quality, different text
                d2     gap 2         moderate quality difference
                d4     gap 4         large quality difference
              The monitor never sees a score: it only knows which frozen
              pair it is looking at.
  production  single responses rated 1-5 by the judge. base = the LOWER
              rated response of a prompt, improved = the HIGHER rated one
              (gap >= 2), a human-verified capability gain paired by prompt.

Labels are used only here, offline, and in the precheck -- the role
simulation truth played in the toy domain.
"""
import gzip, json
from collections import defaultdict

CAP = 800
N_BLOCKS, N_BAT_PER_LEVEL, N_PROD = 10, 2, 8
LEVELS = (0, 2, 4)

def cap(t):
    t = " ".join(str(t).split())
    return t if len(t) <= CAP else t[:CAP].rsplit(" ", 1)[0]

rows = [json.loads(l) for l in gzip.open("hs2.jsonl.gz", "rt")]
by = defaultdict(list)
for r in rows:
    by[r["prompt"]].append(r)

pools = {g: [] for g in LEVELS}
prod_pool = []
for p, v in sorted(by.items()):
    if len(v) != 2:
        continue
    lo, hi = sorted(v, key=lambda r: r["helpfulness"])
    gap = hi["helpfulness"] - lo["helpfulness"]
    a, b = cap(lo["response"]), cap(hi["response"])
    if min(len(a), len(b)) < 150 or len(cap(p)) < 20:
        continue
    if gap in pools:
        pools[gap].append(dict(question=cap(p), a=b, b=a, gap=gap))   # a = higher rated
    if gap >= 2:
        prod_pool.append(dict(question=cap(p), base=a, improved=b, gap=gap))

print("pool sizes:", {g: len(v) for g, v in pools.items()}, "production:", len(prod_pool))
blocks = []
for bi in range(N_BLOCKS):
    bat = []
    for g in LEVELS:
        bat += pools[g][bi * N_BAT_PER_LEVEL:(bi + 1) * N_BAT_PER_LEVEL]
    blocks.append(dict(battery=bat, production=prod_pool[bi * N_PROD:(bi + 1) * N_PROD]))

json.dump(dict(corpus="HelpSteer2 (CC-BY-4.0)", cap=CAP, levels=LEVELS, blocks=blocks),
          open("blocks.json", "w"), indent=1)
lens = [len(t) for blk in blocks for p in blk["production"] for t in (p["base"], p["improved"])]
print(f"wrote blocks.json: {N_BLOCKS} blocks x ({len(blocks[0]['battery'])} battery + {N_PROD} production)")
print(f"response chars: min {min(lens)} median {sorted(lens)[len(lens)//2]} max {max(lens)}")
