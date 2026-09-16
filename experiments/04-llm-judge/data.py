"""Build E-R's frozen blocks from JudgeBench (public, MIT, third-party).

Corpus choice is empirical, not aesthetic. HelpSteer2 was tried first for
its graded human helpfulness gaps, and its precheck failed a precondition:
on subjective quality gaps -- even the largest, gap 4 -- both judges agreed
with the human label only ~50% of the time and showed heavy position bias,
so there was no healthy baseline resolution to degrade. JudgeBench's labels
are OBJECTIVE (LiveBench, LiveCodeBench, MMLU-Pro), and the probe confirmed
a clean instrument: identical-text pairs read TIE 6/6, real pairs read
non-TIE 6/6, and injected provenance metadata collapsed that to 0.5.

  battery     frozen (A, B) pairs plus identical (A, A) floor trials, judged
              only as a forced choice A / B / TIE. The monitor sees no label:
              resolution is the non-TIE rate on real pairs above the
              identical-pair floor -- E-D3's protocol, unchanged.
  production  single responses rated 1-5. base = each question's INCORRECT
              response; improved = the objectively CORRECT one for a fraction
              of the population (a realistic partial improvement, sized at
              precheck so the injected leak's ceiling can reach it).

Labels are used only here, offline, and in the precheck gates.
"""
import json, os, subprocess

CAP, QCAP = 800, 500
N_BLOCKS, N_BAT, N_PROD = 10, 6, 8
BASE = "https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/main/data"
SPLITS = ("gpt-00000-of-00001.jsonl", "claude-00000-of-00001.jsonl")


def cap(t, n=CAP):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0]


rows = []
for fn in SPLITS:
    if not os.path.exists(fn):
        subprocess.run(["curl", "-sL", "-o", fn, f"{BASE}/{fn}"], check=True, timeout=180)
    rows += [json.loads(l) for l in open(fn) if l.strip()]
print(f"loaded {len(rows)} JudgeBench items")

items = []
for r in rows:
    a, b = cap(r["response_A"]), cap(r["response_B"])
    if min(len(a), len(b)) < 150:
        continue
    good, bad = (a, b) if r["label"] == "A>B" else (b, a)
    items.append(dict(question=cap(r["question"], QCAP), a=a, b=b, good=good, bad=bad,
                      source=r["source"], pair_id=r["pair_id"]))
items.sort(key=lambda x: x["pair_id"])
print(f"{len(items)} usable after capping")

blocks = []
for bi in range(N_BLOCKS):
    chunk = items[bi::N_BLOCKS][:N_BAT + N_PROD]
    blocks.append(dict(
        battery=[dict(question=i["question"], a=i["a"], b=i["b"]) for i in chunk[:N_BAT]],
        production=[dict(question=i["question"], base=i["bad"], improved=i["good"])
                    for i in chunk[N_BAT:]]))
json.dump(dict(corpus="JudgeBench (MIT)", cap=CAP, blocks=blocks),
          open("blocks.json", "w"), indent=1)
print(f"wrote blocks.json: {N_BLOCKS} blocks x ({N_BAT} battery + {N_PROD} production)")
