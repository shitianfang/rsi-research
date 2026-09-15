"""Generate the frozen corpus for E-R (LLM-judge drift replication).

Intent-first, as in the toy: the intent S (a priority ORDER over five key
points) is machine-authored and is therefore ground truth by construction.
Artifacts are LLM-written paragraphs realizing S. Everything is generated
once, frozen to corpus.json, and committed -- the reference battery must be
inspectable and identical across all conditions and runs.

Difficulty (amended after precheck round 1, where the judge scored a
perfect 1.000 and left no dynamic range): the ORDER OF APPEARANCE in the
paragraph is deliberately scrambled relative to the priority order, and
priority is conveyed only by how much elaboration each point receives.
Recovering the intent is then an inference, not a transcription -- the
real-domain analogue of the toy's probe budget.

Battery group (per intent): two independent realizations r1, r2 of S, one
realization l2 of S with the top two priorities swapped (appearance order
unchanged), one realization l3 with the lowest-priority point replaced by
a distractor. Pair levels:
    floor  (r1, r1)   identical texts -- the judge's self-inconsistency
    L1     (r1, r2)   same intent, different surface -- should read SAME
    L2     (r1, l2)   priority changed -- should read DIFFERENT
    L3     (r1, l3)   point changed -- should read DIFFERENT

Production set (per intent): base realization + clarity-improved
realization (the true-capability-gain arm: explicit ordering cues).
"""

import json
import sys
import numpy as np
from llm import pmap

GEN_MODEL = "claude-haiku-4-5-20251001"
K_POINTS = 5

# 10 topics x (5 points + 1 distractor), machine-authored, frozen.
TOPICS = [
    ("company newsletter",
     ["the mobile app redesign shipped to all users",
      "two senior engineers joined the platform team",
      "the office recycling program expanded to every floor",
      "customer support moved to a new ticketing system",
      "the quarterly all-hands switched to a hybrid format"],
     "the cafeteria menu now rotates weekly"),
    ("city council notice",
     ["the north bridge repair was completed ahead of schedule",
      "library opening hours were extended to 9pm",
      "a new bus route now serves the riverside district",
      "the winter road salt supply was fully restocked",
      "building permit applications can now be filed online"],
     "the annual street fair was moved to June"),
    ("product changelog",
     ["export to CSV now handles files over a gigabyte",
      "the dashboard loads twice as fast",
      "keyboard shortcuts were added for all menus",
      "the audit log now retains twelve months of history",
      "two-factor authentication supports hardware keys"],
     "the login page got a new background color"),
    ("research lab news",
     ["the imaging cluster was upgraded with new GPUs",
      "a postdoc position opened in the optics group",
      "the seminar series moved to Thursday afternoons",
      "the shared sample freezer was replaced",
      "data archival moved to the university storage service"],
     "the coffee machine in the lounge was replaced"),
    ("school newsletter",
     ["the science fair drew a record number of entries",
      "the gym renovation finished over the break",
      "a coding club now meets on Wednesdays",
      "the bus routes were redrawn for the north neighborhood",
      "library late fees were eliminated"],
     "the parking lot lines were repainted"),
    ("sports club update",
     ["the first team won the regional cup",
      "youth training now runs three evenings a week",
      "the clubhouse roof was fully repaired",
      "the membership fee structure was simplified",
      "a physiotherapist now attends every home match"],
     "new water bottles arrived in the shop"),
    ("museum bulletin",
     ["the Egyptian wing reopened after restoration",
      "school group visits doubled this season",
      "audio guides are now available in six languages",
      "the conservation lab took on two apprentices",
      "evening opening was added on Fridays"],
     "the gift shop extended its weekend hours"),
    ("hospital memo",
     ["the new MRI suite began accepting patients",
      "nurse recruitment reached its quarterly target",
      "visitor parking was reorganized near the east entrance",
      "the pharmacy switched to automated dispensing",
      "discharge summaries now reach GPs within a day"],
     "the lobby plants were replaced with low-maintenance species"),
    ("farm cooperative note",
     ["the grain dryer was replaced before harvest",
      "membership grew by forty families this year",
      "the Saturday market stall moved to the main square",
      "the shared tractor fleet gained two units",
      "soil testing is now offered to every member"],
     "a new logo was chosen for the delivery van"),
    ("software release note",
     ["the sync engine no longer drops offline edits",
      "startup time fell below two seconds",
      "themes can now be imported from a file",
      "crash reports now include anonymised context",
      "the plugin API gained a stable versioning scheme"],
     "the about dialog lists all contributors"),
]


def base_prompt(topic, points, priority, present):
    pres = "\n".join(f"- {points[i]}" for i in present)
    prio = "; ".join(f"({r + 1}) {points[i]}" for r, i in enumerate(priority))
    return (f"Write a single paragraph for a {topic}. Cover exactly these five points, "
            f"introducing them in this order of appearance:\n{pres}\n\n"
            f"Their relative importance is DIFFERENT from that order of appearance. "
            f"From most important to least important: {prio}.\n\n"
            f"Convey the relative importance only through how much detail and emphasis "
            f"each point receives: the most important point gets the most elaboration, "
            f"the least important little more than a passing mention. Do NOT use "
            f"numbering, bullets, or ordering words such as 'first', 'most importantly', "
            f"'finally', or 'lastly'. Output only the paragraph, 6-8 sentences.")


def clear_prompt(topic, points, priority):
    prio = "\n".join(f"- {points[i]}" for i in priority)
    return (f"Write a single paragraph for a {topic}. Cover exactly these five points in "
            f"this exact order, from most important to least important:\n{prio}\n\n"
            f"Make the priority order unmistakable using explicit cues: begin the points "
            f"with 'Most importantly', 'Second', 'Third', 'Fourth', and 'Least significant'. "
            f"One point per sentence, in the given order. Output only the paragraph.")


def main():
    rng = np.random.default_rng(11)
    blocks = []
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return len(prompts) - 1

    def perm():
        return [int(i) for i in rng.permutation(K_POINTS)]

    for _ in range(10):                       # 10 blocks: 2 precheck + 8 registered
        block = {"battery": [], "production": []}
        for _ in range(4):                    # 4 battery groups per block
            topic, pts, distract = TOPICS[int(rng.integers(0, len(TOPICS)))]
            priority = perm()
            present = perm()
            while present == priority:        # appearance must not equal priority
                present = perm()
            swapped = [priority[1], priority[0]] + priority[2:]
            pts_repl = list(pts)
            pts_repl[priority[-1]] = distract  # replace the lowest-priority point
            block["battery"].append(dict(
                topic=topic, points=pts, distractor=distract,
                priority=priority, present=present,
                r1=ask(base_prompt(topic, pts, priority, present)),
                r2=ask(base_prompt(topic, pts, priority, present)),
                l2=ask(base_prompt(topic, pts, swapped, present)),
                l3=ask(base_prompt(topic, pts_repl, priority, present))))
        for _ in range(4):                    # 4 production intents per block
            topic, pts, _ = TOPICS[int(rng.integers(0, len(TOPICS)))]
            priority = perm()
            present = perm()
            while present == priority:
                present = perm()
            block["production"].append(dict(
                topic=topic, points=pts, order=priority, present=present,
                base=ask(base_prompt(topic, pts, priority, present)),
                improved=ask(clear_prompt(topic, pts, priority))))
        blocks.append(block)

    print(f"generating {len(prompts)} paragraphs with {GEN_MODEL} ...")
    texts = pmap(prompts, model=GEN_MODEL)
    bad = [i for i, t in enumerate(texts) if len(t) < 80]
    if bad:
        print(f"retrying {len(bad)} short/empty generations ...")
        for i, t in zip(bad, pmap([prompts[i] for i in bad], model=GEN_MODEL)):
            texts[i] = t

    for block in blocks:
        for grp in block["battery"]:
            for k in ("r1", "r2", "l2", "l3"):
                grp[k] = texts[grp[k]]
        for p in block["production"]:
            p["base"] = texts[p["base"]]
            p["improved"] = texts[p["improved"]]

    with open("corpus.json", "w") as f:
        json.dump(dict(model=GEN_MODEL, k_points=K_POINTS, blocks=blocks), f, indent=1)
    n_bad = sum(1 for t in texts if len(t) < 80)
    print(f"wrote corpus.json: 10 blocks, {len(texts)} texts, {n_bad} still empty/short")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
