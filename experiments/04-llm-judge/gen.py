"""Generate the frozen corpus for E-R (LLM-judge drift replication).

Intent-first, as in the toy: the intent S (a priority ORDER over three key
points) is machine-authored and is therefore ground truth by construction.
Artifacts are LLM-written paragraphs realizing S. Everything is generated
once, frozen to corpus.json, and committed -- the reference battery must be
inspectable and identical across all conditions and runs.

Battery group (per intent): two independent realizations r1, r2 of S, one
realization l2 of S with positions 1 and 2 swapped, one realization l3 of S
with the lowest-priority point replaced by a distractor. Pair levels:
    floor  (r1, r1)   identical texts -- the judge's self-inconsistency
    L1     (r1, r2)   same intent, different surface -- should read SAME
    L2     (r1, l2)   order changed -- should read DIFFERENT
    L3     (r1, l3)   point changed -- should read DIFFERENT

Production set (per intent): base realization + clarity-improved realization
(the true-capability-gain arm rewrites, with explicit ordering cues).
"""

import json
import sys
import numpy as np
from llm import pmap

GEN_MODEL = "claude-haiku-4-5-20251001"

# 10 topics x (3 points + 1 distractor), machine-authored, frozen.
TOPICS = [
    ("company newsletter", ["the mobile app redesign shipped to all users",
                            "two senior engineers joined the platform team",
                            "the office recycling program expanded to every floor"],
     "the cafeteria menu now rotates weekly"),
    ("city council notice", ["the north bridge repair was completed ahead of schedule",
                             "library opening hours were extended to 9pm",
                             "a new bus route now serves the riverside district"],
     "the annual street fair was moved to June"),
    ("product changelog", ["export to CSV now handles files over a gigabyte",
                           "the dashboard loads twice as fast",
                           "keyboard shortcuts were added for all menus"],
     "the login page got a new background color"),
    ("research lab news", ["the imaging cluster was upgraded with new GPUs",
                           "a postdoc position opened in the optics group",
                           "the seminar series moved to Thursday afternoons"],
     "the coffee machine in the lounge was replaced"),
    ("school newsletter", ["the science fair drew a record number of entries",
                           "the gym renovation finished over the break",
                           "a coding club now meets on Wednesdays"],
     "the parking lot lines were repainted"),
    ("sports club update", ["the first team won the regional cup",
                            "youth training now runs three evenings a week",
                            "the clubhouse roof was fully repaired"],
     "new water bottles arrived in the shop"),
    ("museum bulletin", ["the Egyptian wing reopened after restoration",
                         "school group visits doubled this season",
                         "audio guides are now available in six languages"],
     "the gift shop extended its weekend hours"),
    ("hospital memo", ["the new MRI suite began accepting patients",
                       "nurse recruitment reached its quarterly target",
                       "visitor parking was reorganized near the east entrance"],
     "the lobby plants were replaced with low-maintenance species"),
    ("farm cooperative note", ["the grain dryer was replaced before harvest",
                               "membership grew by forty families this year",
                               "the Saturday market stall moved to the main square"],
     "a new logo was chosen for the delivery van"),
    ("software release note", ["the sync engine no longer drops offline edits",
                               "startup time fell below two seconds",
                               "themes can now be imported from a file"],
     "the about dialog lists all contributors"),
]

BASE_STYLE = ("Do not use numbering, bullets, or explicit ordering words such as "
              "'first', 'second', or 'finally'; convey the priority purely through "
              "sentence order and emphasis. Output only the paragraph, 3-4 sentences.")
CLEAR_STYLE = ("Make the priority order unmistakable with explicit cues such as "
               "'Most importantly', 'Second', and 'Lastly', one point per sentence "
               "in the given order. Output only the paragraph, 3-4 sentences.")


def gen_prompt(topic, points, order, style):
    ordered = [points[i] for i in order]
    return (f"Write a single short paragraph for a {topic} that covers exactly these "
            f"three points, presented from highest to lowest priority in this order: "
            f"1) {ordered[0]} 2) {ordered[1]} 3) {ordered[2]}. {style}")


def main():
    rng = np.random.default_rng(7)
    blocks = []
    prompts, slots = [], []

    def ask(prompt):
        prompts.append(prompt)
        slots.append(len(prompts) - 1)
        return slots[-1]

    for b in range(10):                       # 10 blocks: 2 precheck + 8 registered
        block = {"battery": [], "production": []}
        for g in range(4):                    # 4 battery groups per block
            t = int(rng.integers(0, len(TOPICS)))
            topic, pts, distract = TOPICS[t]
            order = [int(i) for i in rng.permutation(3)]
            swapped = [order[1], order[0], order[2]]
            pts_repl = list(pts)
            pts_repl[order[2]] = distract     # replace the lowest-priority point
            grp = dict(topic=topic, points=pts, distractor=distract, order=order,
                       r1=ask(gen_prompt(topic, pts, order, BASE_STYLE)),
                       r2=ask(gen_prompt(topic, pts, order, BASE_STYLE)),
                       l2=ask(gen_prompt(topic, pts, swapped, BASE_STYLE)),
                       l3=ask(gen_prompt(topic, pts_repl, order, BASE_STYLE)))
            block["battery"].append(grp)
        for g in range(4):                    # 4 production intents per block
            t = int(rng.integers(0, len(TOPICS)))
            topic, pts, _ = TOPICS[t]
            order = [int(i) for i in rng.permutation(3)]
            block["production"].append(dict(
                topic=topic, points=pts, order=order,
                base=ask(gen_prompt(topic, pts, order, BASE_STYLE)),
                improved=ask(gen_prompt(topic, pts, order, CLEAR_STYLE))))
        blocks.append(block)

    print(f"generating {len(prompts)} paragraphs with {GEN_MODEL} ...")
    texts = pmap(prompts, model=GEN_MODEL)
    bad = [i for i, t in enumerate(texts) if len(t) < 40]
    if bad:
        print(f"retrying {len(bad)} short/empty generations ...")
        redo = pmap([prompts[i] for i in bad], model=GEN_MODEL)
        for i, t in zip(bad, redo):
            texts[i] = t

    def resolve(x):
        return texts[x]

    for block in blocks:
        for grp in block["battery"]:
            for k in ("r1", "r2", "l2", "l3"):
                grp[k] = resolve(grp[k])
        for p in block["production"]:
            p["base"] = resolve(p["base"])
            p["improved"] = resolve(p["improved"])

    with open("corpus.json", "w") as f:
        json.dump(dict(model=GEN_MODEL, blocks=blocks), f, indent=1)
    n_empty = sum(1 for t in texts if len(t) < 40)
    print(f"wrote corpus.json: 10 blocks, {len(texts)} texts, {n_empty} still empty/short")
    sys.exit(1 if n_empty else 0)


if __name__ == "__main__":
    main()
