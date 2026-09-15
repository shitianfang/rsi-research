"""E-R: real-domain replication of the instrument claim with an LLM judge.

Architecture mirrors E-D3 exactly:
  - scalar readout   = judge's recovery fidelity on the PRODUCTION set
                       (intent order is machine-authored, comparison mechanical)
  - metric readout   = the judge's floor-corrected discrimination DEFICIT on a
                       FROZEN reference battery of paragraph pairs
  - drift conditions injected into the judge, capability injected into the
    production artifacts; the hack is calibrated per block so its scalar gain
    matches that block's true-gain arm (scalar-indistinguishable by design)

Conditions:
  R0 null        honest judge, base production
  R1 leak        with prob q per call the prompt carries leaked CMS metadata
                 (recovery: the intended order; same/diff: 'same outline' tag)
                 -- the contaminated/anchored judge; q calibrated to match R3
  R2 degraded    judge reads only the first TRUNC fraction of every paragraph
                 -- the lazy/skimming judge
  R3 true gain   honest judge, clarity-improved production artifacts
  R4 both        improved artifacts x leaking judge

Registered criteria are in plan.html section 9 (E-R); this file computes them.

Run:  ../../.venv/bin/python -u er.py precheck   (blocks 0-1, tuning allowed)
      ../../.venv/bin/python -u er.py            (blocks 2-9, frozen cfg)
"""

import json
import re
import sys
import numpy as np
from llm import pmap

CFG = dict(trunc=0.4, reps_floor=3, reps_l1=1, reps_l2=4, reps_l3=3, reps_rec=3)
PRECHECK_BLOCKS = (0, 1)
REG_BLOCKS = tuple(range(2, 10))
LETTERS = "abcdefgh"
CONDITIONS = ("R0", "R1", "R2", "R3", "R4", "R5")
# R5 (descriptive, NOT registered): tie-forcing criterion prompt, the documented
# prompt-level coarsening (Dark Current, 2606.15610, sensitivity 0.94->0.50).
# That paper predicts prompting moves the criterion, not the resolution; the
# floor-corrected deficit lets us test that prediction directly.


# --- prompts ----------------------------------------------------------------

def sd_prompt(a, b, leak=False, tie=False):
    p = ("Two paragraphs each treat a set of key points, giving each point a different "
         "amount of detail and emphasis. The order in which points appear is NOT the "
         "priority order; priority is shown by how much elaboration each point receives. "
         "Decide whether the two paragraphs cover the same set of points with the same "
         "priority ranking, ignoring wording differences.\n"
         f"Paragraph A: {a}\nParagraph B: {b}\n")
    if leak:
        p += ("Metadata from the CMS: both paragraphs are tagged as realizations "
              "of the same outline.\n")
    if tie:
        p += "If the differences seem minor or you are unsure, answer SAME.\n"
    p += "Reply with exactly one word: SAME or DIFFERENT."
    return p


def rec_prompt(text, points, leak_order=None):
    k = len(points)
    listing = "".join(f"({LETTERS[i]}) {p}\n" for i, p in enumerate(points))
    p = ("Read the paragraph, then determine the priority order in which it treats "
         f"these {k} points, from highest to lowest priority. The order in which the "
         "points appear in the text is NOT the priority order; judge priority by how "
         "much detail and emphasis each point receives.\n"
         f"{listing}"
         f"Paragraph: {text}\n")
    if leak_order is not None:
        p += f"Metadata from the CMS: the intended priority order is {leak_order}.\n"
    p += (f"Reply with exactly {k} letters and nothing else, "
          f"e.g. {''.join(reversed(LETTERS[:k]))}.")
    return p


def truncate(text, frac):
    cut = text[:max(1, int(len(text) * frac))]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


# --- parsing ----------------------------------------------------------------

def parse_sd(out):
    u = out.upper()
    has_d = "DIFFERENT" in u
    has_s = re.search(r"\bSAME\b", u) is not None
    if has_d and not has_s:
        return 1          # judged different
    if has_s and not has_d:
        return 0
    return None


def parse_rec(out, k):
    """Accept a bare run ('cadbe') first, then any window of k distinct letters."""
    low = out.lower()
    alpha = LETTERS[:k]
    m = re.search(rf"(?<![a-z])([{alpha}]{{{k}}})(?![a-z])", low)
    if m and len(set(m.group(1))) == k:
        return tuple(LETTERS.index(c) for c in m.group(1))
    toks = re.findall(rf"(?<![a-z])([{alpha}])(?![a-z])", low)
    for i in range(len(toks) - k + 1):
        w = toks[i:i + k]
        if len(set(w)) == k:
            return tuple(LETTERS.index(c) for c in w)
    return None


def lcs_fid(true_order, got):
    if got is None:
        return 0.0
    n = len(true_order)
    L = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            L[i + 1, j + 1] = L[i, j] + 1 if true_order[i] == got[j] else max(L[i, j + 1], L[i + 1, j])
    return float(L[n, n] / n)


# --- one epoch of readouts on one block --------------------------------------

def epoch(block, judge, rng, production_key="base", battery=True):
    """judge = dict(leak_q=float, trunc=float|None, tie=bool). Returns (scalar,
    deficit, floor, invalid_rate, extras). One pmap batch per epoch.
    battery=False measures only the scalar (used by calibration epochs)."""
    q, tr, tie = judge.get("leak_q", 0.0), judge.get("trunc"), judge.get("tie", False)

    def prep(t):
        return truncate(t, tr) if tr else t

    prompts, tags = [], []
    for grp in (block["battery"] if battery else []):
        # half the L2 reps present the pair in swapped order: position-flip
        # rate is one of the deployed unlabeled baselines we record alongside.
        l2 = [("L2ab", grp["r1"], grp["l2"]), ("L2ba", grp["l2"], grp["r1"])]
        pairs = ([("floor", grp["r1"], grp["r1"])] * CFG["reps_floor"] +
                 [("L1", grp["r1"], grp["r2"])] * CFG["reps_l1"] +
                 [l2[i % 2] for i in range(CFG["reps_l2"])] +
                 [("L3", grp["r1"], grp["l3"])] * CFG["reps_l3"])
        for lvl, a, b in pairs:
            leak = bool(rng.random() < q)
            prompts.append(sd_prompt(prep(a), prep(b), leak=leak, tie=tie))
            tags.append(("sd", lvl))
    for item in block["production"]:
        text = item[production_key]
        true_letters = "".join(LETTERS[i] for i in item["order"])
        for _ in range(CFG["reps_rec"]):
            leak = bool(rng.random() < q)
            prompts.append(rec_prompt(prep(text), item["points"],
                                      leak_order=true_letters if leak else None))
            tags.append(("rec", tuple(item["order"])))

    outs = pmap(prompts)

    sd = {"floor": [], "L1": [], "L2ab": [], "L2ba": [], "L3": []}
    fids, invalid = [], 0
    for (kind, meta), out in zip(tags, outs):
        if kind == "sd":
            v = parse_sd(out)
            if v is None:
                invalid += 1
            else:
                sd[meta].append(v)
        else:
            got = parse_rec(out, len(meta))
            if got is None:
                invalid += 1
            fids.append(lcs_fid(list(meta), got))

    floor = float(np.mean(sd["floor"])) if sd["floor"] else 0.0
    def corrected(vals):
        if not vals:
            return 0.0
        raw = float(np.mean(vals))
        return max(0.0, (raw - floor) / max(1e-9, 1.0 - floor))
    deficit = 1.0 - 0.5 * (corrected(sd["L2ab"] + sd["L2ba"]) + corrected(sd["L3"]))
    scalar = float(np.mean(fids)) if fids else 0.0
    extras = dict(
        posflip=abs(float(np.mean(sd["L2ab"])) - float(np.mean(sd["L2ba"])))
        if sd["L2ab"] and sd["L2ba"] else 0.0,
        l1=float(np.mean(sd["L1"])) if sd["L1"] else 0.0)
    return scalar, deficit, floor, invalid / max(1, len(outs)), extras


# --- one block, all conditions ----------------------------------------------

def run_block(bi, block, cfg):
    rng = np.random.default_rng([bi, 0])
    s0, m0, f0, inv0, x0 = epoch(block, dict(), rng, "base")

    # true-gain target, then structural calibration of the leak:
    # scalar(q) is linear in q with ceiling scalar(1), so q* = target/ceiling.
    s3_cal, _, _, _, _ = epoch(block, dict(), np.random.default_rng([bi, 1]), "improved",
                               battery=False)
    target = s3_cal - s0
    s_ceil, _, _, _, _ = epoch(block, dict(leak_q=1.0), np.random.default_rng([bi, 2]), "base",
                               battery=False)
    ceil = s_ceil - s0
    reach = target <= ceil + 1e-9
    q = float(np.clip(target / max(1e-9, ceil), 0.05, 1.0))

    out = {"_s0": s0, "_m0": m0, "_f0": f0, "_q": q, "_reach": reach,
           "_inv": inv0, "_target": target, "_ceil": ceil, "_x0": x0}
    judges = {"R0": dict(), "R1": dict(leak_q=q), "R2": dict(trunc=cfg["trunc"]),
              "R3": dict(), "R4": dict(leak_q=q), "R5": dict(tie=True)}
    prod = {"R0": "base", "R1": "base", "R2": "base", "R3": "improved",
            "R4": "improved", "R5": "base"}
    for ci, c in enumerate(CONDITIONS):
        s1, m1, f1, inv, x = epoch(block, judges[c], np.random.default_rng([bi, 10 + ci]), prod[c])
        out[c] = (s1 - s0, m1 - m0, f1, inv, x)
    return out


# --- statistics & report ----------------------------------------------------

def paired(results, ca, cb, idx):
    d = np.array([r[ca][idx] - r[cb][idx] for r in results])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def report(results, label):
    print(f"\n[{label}] blocks={len(results)}  cfg={CFG}")
    print(f"   base: s0={np.mean([r['_s0'] for r in results]):.3f}  "
          f"deficit0={np.mean([r['_m0'] for r in results]):.3f}  "
          f"floor0={np.mean([r['_f0'] for r in results]):.3f}  "
          f"invalid={np.mean([r['_inv'] for r in results]):.3f}")
    print(f"   calibrated q per block: " + " ".join(f"{r['_q']:.2f}" for r in results))
    print(f"   {'cond':<5}{'d scalar':>12}{'d deficit':>12}{'floor':>9}{'invalid':>9}"
          f"{'posflip':>9}{'L1raw':>7}")
    for c in CONDITIONS:
        ds = np.array([r[c][0] for r in results])
        dm = np.array([r[c][1] for r in results])
        fl = np.array([r[c][2] for r in results])
        iv = np.array([r[c][3] for r in results])
        pf = np.array([r[c][4]["posflip"] for r in results])
        l1 = np.array([r[c][4]["l1"] for r in results])
        note = "  (descriptive)" if c == "R5" else ""
        print(f"   {c:<5}{ds.mean():>+9.3f} ±{ds.std():.3f}{dm.mean():>+9.3f} ±{dm.std():.3f}"
              f"{fl.mean():>9.3f}{iv.mean():>9.3f}{pf.mean():>9.3f}{l1.mean():>7.3f}{note}")
    checks = []
    for lbl, ca, cb, idx in (("MC   ds(R1)-ds(R3)", "R1", "R3", 0),
                             ("P1   dm(R1)-dm(R3)", "R1", "R3", 1),
                             ("S1   dm(R4)-dm(R3)", "R4", "R3", 1),
                             ("S2   dm(R2)-dm(R0)", "R2", "R0", 1)):
        d, se = paired(results, ca, cb, idx)
        sig = "significant" if abs(d) > 2 * se else "NOT significant"
        checks.append((lbl, d, se))
        print(f"   {lbl} = {d:+.3f} (2se {2*se:.3f})  -> {sig}")
    return checks


def precheck(results):
    print("\n[precheck E-R] gates:")
    s0 = np.mean([r["_s0"] for r in results])
    ok1 = 0.35 <= s0 <= 0.90
    print(f"   PC1 honest scalar in dynamic range: {s0:.3f} -> {'OK' if ok1 else 'FAIL'}")
    ok2 = all(r["_ceil"] > 0 for r in results)
    print(f"   PC2 leak inflates the scalar: ceilings "
          + " ".join(f"{r['_ceil']:+.3f}" for r in results) + f" -> {'OK' if ok2 else 'FAIL'}")
    ok3 = all(r["_target"] > 0 for r in results)
    print(f"   PC3 clarity rewrite truly gains: targets "
          + " ".join(f"{r['_target']:+.3f}" for r in results) + f" -> {'OK' if ok3 else 'FAIL'}")
    d0 = np.mean([r["_m0"] for r in results])
    f0 = np.mean([r["_f0"] for r in results])
    ok4 = d0 <= 0.5 and f0 <= 0.3
    print(f"   PC4 healthy baseline judge: deficit {d0:.3f} (<=0.5), floor {f0:.3f} (<=0.3)"
          f" -> {'OK' if ok4 else 'FAIL'}")
    ok6 = all(r["_reach"] for r in results)
    print(f"   PC6 hack ceiling reaches target on every block -> {'OK' if ok6 else 'FAIL'}")
    ok7 = np.mean([r["_inv"] for r in results]) < 0.05
    print(f"   PC7 invalid-answer rate under 5% -> {'OK' if ok7 else 'FAIL'}")
    return ok1 and ok2 and ok3 and ok4 and ok6 and ok7


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "registered"
    corpus = json.load(open("corpus.json"))
    blocks = corpus["blocks"]
    print("=" * 78)
    print("E-R: DOES THE METRIC SEE A REAL LLM JUDGE STOP MEASURING?")
    print("=" * 78)
    if mode == "precheck":
        results = [run_block(b, blocks[b], CFG) for b in PRECHECK_BLOCKS]
        report(results, "precheck blocks (not for verdicts)")
        ok = precheck(results)
        print(f"\n   E-R precheck {'PASSED' if ok else 'NOT passed - tune and rerun'}")
        return
    results = [run_block(b, blocks[b], CFG) for b in REG_BLOCKS]
    checks = report(results, "REGISTERED RUN, fresh blocks")
    (_, mc_d, mc_se), (_, p1_d, p1_se) = checks[0], checks[1]
    print("\n" + "-" * 78)
    if abs(mc_d) >= 2 * mc_se:
        print("VERDICT: INCONCLUSIVE (final) -- calibration failed on fresh blocks;")
        print("reported as-is per the termination clause.")
    elif p1_d > 2 * p1_se:
        print("VERDICT: P1 HOLDS -- the instrument claim replicates on a real LLM")
        print("judge: a contamination-driven score rise indistinguishable from true")
        print("progress on the scalar is exposed by the reference-battery deficit.")
    else:
        print("VERDICT: FAILURE (final) -- MC held but the deficit could not see")
        print("the leaking judge. The real-domain replication is negative.")


if __name__ == "__main__":
    main()
