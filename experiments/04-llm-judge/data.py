"""Build E-R's frozen blocks from JudgeBench (public, MIT, third-party).

Why an external corpus: nothing in this experiment should be authored by
the loop being studied. JudgeBench ships 620 (question, response_A,
response_B) items whose correctness labels come from LiveBench,
LiveCodeBench and MMLU-Pro -- verified outside this project.

How each half is used:
  battery     frozen pairs asked only as SAME/DIFFERENT quality judgments.
              The monitor never sees a label: a "real" pair is just (A, B),
              a floor pair is (A, A). This is the label-free half.
  production  single responses rated 1-5. The base population is each
              question's INCORRECT response, the improved population the
              CORRECT one -- an objectively verified capability gain on the
              same questions, which is the control arm the whole design
              turns on.

Labels are used only here, offline, to build the arms and (in the precheck)
to confirm the gain is real -- the same role simulation truth played in the
toy domain. Responses are capped to CAP characters at build time, applied
identically to every condition, so prompt cost is bounded and the R2
truncation arm still bites.

Run:  ../../.venv/bin/python data.py
"""

import json
import os
import subprocess

CAP = 1200
N_BLOCKS = 10
N_BATTERY = 6          # frozen pairs per block
N_PROD = 12            # production questions per block
BASE = "https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/main/data"
SPLITS = ("gpt-00000-of-00001.jsonl", "claude-00000-of-00001.jsonl")


def fetch():
    rows = []
    for fn in SPLITS:
        if not os.path.exists(fn):
            subprocess.run(["curl", "-sL", "-o", fn, f"{BASE}/{fn}"], check=True, timeout=180)
        rows += [json.loads(l) for l in open(fn) if l.strip()]
    return rows


def cap(text):
    t = " ".join(str(text).split())
    if len(t) <= CAP:
        return t
    return t[:CAP].rsplit(" ", 1)[0]


def main():
    rows = fetch()
    print(f"loaded {len(rows)} JudgeBench items")
    items = []
    for r in rows:
        a, b = cap(r["response_A"]), cap(r["response_B"])
        if min(len(a), len(b)) < 120:          # unusably short after capping
            continue
        good, bad = (a, b) if r["label"] == "A>B" else (b, a)
        items.append(dict(pair_id=r["pair_id"], source=r["source"],
                          question=cap(r["question"]), a=a, b=b, good=good, bad=bad))
    print(f"{len(items)} usable after capping to {CAP} chars")

    need = N_BLOCKS * (N_BATTERY + N_PROD)
    assert len(items) >= need, f"need {need}, have {len(items)}"
    # deterministic, interleaved across sources so no block is single-domain
    items.sort(key=lambda x: (x["pair_id"]))
    blocks = []
    for bi in range(N_BLOCKS):
        chunk = items[bi::N_BLOCKS][:N_BATTERY + N_PROD]
        blocks.append(dict(
            battery=[dict(question=i["question"], a=i["a"], b=i["b"]) for i in chunk[:N_BATTERY]],
            production=[dict(question=i["question"], base=i["bad"], improved=i["good"])
                        for i in chunk[N_BATTERY:]]))
    with open("blocks.json", "w") as f:
        json.dump(dict(corpus="JudgeBench (MIT)", cap=CAP, blocks=blocks), f, indent=1)
    lens = [len(t) for blk in blocks for p in blk["production"] for t in (p["base"], p["improved"])]
    print(f"wrote blocks.json: {N_BLOCKS} blocks x ({N_BATTERY} battery + {N_PROD} production)")
    print(f"response lengths: min {min(lens)} median {sorted(lens)[len(lens)//2]} max {max(lens)}")


if __name__ == "__main__":
    main()
