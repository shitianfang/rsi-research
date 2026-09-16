#!/usr/bin/env python3
"""
Poster-layout demo: a self-improving loop whose judge gets corrupted.

What this makes visible
-----------------------
A hill-climbing loop optimizes a 5-element poster layout against a judge.
The judge is a resource-limited viewer: it glances at the poster under a budget
of K fixations and reports the reading order it recovered. A layout is GOOD when
that recovered order matches the machine-authored INTENT -- a real,
machine-checkable design property, not an opinion.

Two runs share a seed and a starting layout:
  honest     -- the judge is the real budgeted observer for all 24 rounds
  corrupted  -- from round 9 the judge leaks the answer (p=0.8 that it does not
                look at the poster at all and simply returns the intended order)

Both runs log what a team WOULD watch (reported_score) next to the ground truth
they CANNOT see (true_score), plus the remedy being illustrated: a FROZEN
reference battery re-tested every round, measuring whether the judge can still
tell two posters apart -- corrected for its own noise on identical inputs.

Pure Python 3 + numpy. No network, no LLM, no external assets.
"""

import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Poster model
# ---------------------------------------------------------------------------

ROLES = ("title", "subtitle", "body", "cta", "footer")
N_ROLES = len(ROLES)
N_PAIRS = N_ROLES * (N_ROLES - 1) // 2  # 10 ordered pairs

TEXT = {
    "title": "AUTUMN LECTURE SERIES",
    "subtitle": "Six evenings on perception",
    "body": "Every Thursday, 7pm, Hall C. Open to all students and staff.",
    "cta": "Reserve a seat",
    "footer": "Department of Cognitive Science",
}

# Layout parameters, in column order, each continuous and bounded.
PARAMS = ("size", "y", "weight", "contrast")
LO = np.array([10.0, 6.0, 300.0, 0.15])
HI = np.array([64.0, 94.0, 800.0, 1.00])
RANGE = HI - LO

CANVAS_W, CANVAS_H = 320.0, 440.0

# Observer
BUDGET_K = 5          # fixations the viewer can spend
NOISE_SD = 0.06       # saliency noise
FIDELITY_REPS = 32    # glances averaged into one fidelity score. A noisy
                      # judge makes hill climbing accept lucky-but-worse
                      # candidates; more glances is the judge looking longer
                      # before it decides, and it is what makes the trace
                      # legible without hiding anything.

# Loop
ROUNDS = 80
CORRUPT_FROM = 52   # late enough that the loop has already built a genuinely good poster: corruption must destroy something, not drift up from a bad start
P_LEAK = 0.8
STEP_FRAC = 0.14      # perturbation sd as a fraction of each parameter's range

# Ruler (frozen reference battery)
N_REAL_PAIRS = 6
N_IDENTICAL_PAIRS = 4
BATTERY_REPS = 32     # passes of the frozen battery per round (measurement only)
MIN_GAP = 0.20        # minimum adjacent saliency gap of a battery stimulus

SETUP_SEED = 20260916  # intent + frozen battery
RUN_SEED = 10          # the loop itself (shared by both runs)


# ---------------------------------------------------------------------------
# The observer (the judge's honest core)
# ---------------------------------------------------------------------------

def saliency_mean(layout):
    """Noiseless saliency. layout is (5, 4): size, y, weight, contrast."""
    size, y, weight, contrast = layout[:, 0], layout[:, 1], layout[:, 2], layout[:, 3]
    return (
        0.55 * (size - LO[0]) / RANGE[0]
        + 0.30 * contrast
        + 0.15 * (weight - LO[2]) / RANGE[2]
        - 0.25 * np.abs(y - 30.0) / 100.0
    )


def saliency(layout, rng):
    return saliency_mean(layout) + rng.normal(0.0, NOISE_SD, N_ROLES)


def recover_order(layout, rng):
    """Fixation sequence under the budget: most salient first."""
    order = np.argsort(-saliency(layout, rng), kind="stable")
    return tuple(int(i) for i in order[:BUDGET_K])


def rank_of(order):
    """rank[role_index] = where that role lands in this order."""
    rank = np.full(N_ROLES, N_ROLES, dtype=int)
    for pos, role in enumerate(order):
        rank[role] = pos
    return rank


def pair_agreement(intent_rank, order):
    """Kendall-tau-like agreement, normalized to [0, 1]: correct pairs / 10."""
    rank = rank_of(order)
    correct = 0
    for a in range(N_ROLES):
        for b in range(a + 1, N_ROLES):
            if (intent_rank[a] < intent_rank[b]) == (rank[a] < rank[b]):
                correct += 1
    return correct / N_PAIRS


def fidelity(layout, intent_rank, rng, reps=FIDELITY_REPS):
    """Average agreement between the intended order and the recovered order."""
    return float(np.mean([
        pair_agreement(intent_rank, recover_order(layout, rng)) for _ in range(reps)
    ]))


# ---------------------------------------------------------------------------
# The judge (honest, or answer-leaking from CORRUPT_FROM on)
# ---------------------------------------------------------------------------

class Judge:
    """
    Wraps the observer. When contaminated, a judge call does not look at the
    poster at all: it returns the intended order, hence score 1.0. The leak coin
    lives on its own RNG stream, so the honest and corrupted runs are
    bit-identical for rounds 1..CORRUPT_FROM-1.
    """

    def __init__(self, intent, intent_rank, rng_noise, rng_leak,
                 rng_probe_noise, rng_probe_leak, corrupt_from=None):
        self.intent = intent
        self.intent_rank = intent_rank
        # production stream: the judge calls the loop actually makes
        self.rng_noise = rng_noise
        self.rng_leak = rng_leak
        # probe stream: the ruler queries the same judge without perturbing the
        # loop, so how hard we monitor never changes what the loop does
        self.rng_probe_noise = rng_probe_noise
        self.rng_probe_leak = rng_probe_leak
        self.corrupt_from = corrupt_from
        self.round = 0

    def _leaking(self, probe=False):
        if self.corrupt_from is None or self.round < self.corrupt_from:
            return False
        rng = self.rng_probe_leak if probe else self.rng_leak
        return rng.random() < P_LEAK

    def recover(self, layout, probe=False):
        if self._leaking(probe):
            return self.intent
        return recover_order(layout, self.rng_probe_noise if probe else self.rng_noise)

    def score(self, layout):
        if self._leaking():
            return 1.0
        return fidelity(layout, self.intent_rank, self.rng_noise)


# ---------------------------------------------------------------------------
# The ruler: a frozen reference battery, built once before any run
# ---------------------------------------------------------------------------

def stimulus(rng):
    """
    A battery stimulus is a layout with a well-separated saliency profile, so
    the honest observer's own noise rarely reorders it. Built by spreading the
    five roles over distinct prominence levels, then rejecting any layout whose
    smallest adjacent saliency gap is under MIN_GAP.
    """
    while True:
        t = np.clip(rng.permutation(np.linspace(0.0, 1.0, N_ROLES))
                    + rng.normal(0.0, 0.03, N_ROLES), 0.0, 1.0)
        layout = np.empty((N_ROLES, 4))
        layout[:, 0] = 12.0 + t * 50.0            # size
        layout[:, 1] = rng.uniform(8.0, 92.0, N_ROLES)  # y
        layout[:, 2] = 320.0 + t * 450.0          # weight
        layout[:, 3] = 0.20 + t * 0.75            # contrast
        layout = np.clip(layout, LO, HI)
        gaps = -np.diff(np.sort(saliency_mean(layout))[::-1])
        if np.min(gaps) >= MIN_GAP:
            return layout


def orders_robustly_differ(a, b, rng, trials=24, min_hits=23):
    """Build-time verification with the honest observer."""
    hits = sum(recover_order(a, rng) != recover_order(b, rng) for _ in range(trials))
    return hits >= min_hits


def build_battery(rng):
    """
    6 REAL pairs whose induced reading orders genuinely differ (verified here,
    once, with the honest observer) + 4 IDENTICAL pairs (the same layout twice).
    Frozen before any run touches it.
    """
    real = []
    guard = 0
    while len(real) < N_REAL_PAIRS:
        guard += 1
        if guard > 20000:
            raise RuntimeError("could not build the reference battery")
        a, b = stimulus(rng), stimulus(rng)
        if orders_robustly_differ(a, b, rng):
            real.append((a, b))
    identical = [(lambda L: (L, L.copy()))(stimulus(rng)) for _ in range(N_IDENTICAL_PAIRS)]
    return {"real": real, "identical": identical}


def ruler_readout(judge, battery, reps=BATTERY_REPS):
    """
    Ask the CURRENT judge, per pair, "same reading order or different?" -- it
    recovers each poster's order and answers "different" if they differ.
      floor      = fraction of IDENTICAL pairs it calls different (its own noise)
      hit        = fraction of REAL pairs it calls different
      resolution = max(0, hit - floor);  deficit = 1 - resolution
    Averaged over `reps` passes of the frozen battery: a more precise estimate
    of the same two fractions, nothing else.
    """
    def frac_different(pairs):
        diff = total = 0
        for _ in range(reps):
            for a, b in pairs:
                diff += judge.recover(a, probe=True) != judge.recover(b, probe=True)
                total += 1
        return diff / total

    hit = frac_different(battery["real"])
    floor = frac_different(battery["identical"])
    resolution = max(0.0, hit - floor)
    return resolution, 1.0 - resolution, hit, floor


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CHAR_W = 0.62  # conservative sans-serif advance width, in em


def _wrap(text, max_chars):
    """Greedy word wrap, by hand. A word too long for the measure is hard-broken
    with a hyphen, so no line can ever run past the frame."""
    words = []
    for w in text.split():
        while len(w) > max_chars:
            words.append(w[:max_chars - 1] + "-")
            w = w[max_chars - 1:]
        words.append(w)

    lines, cur = [], ""
    for w in words:
        cand = w if not cur else cur + " " + w
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def render_svg(layout):
    """
    A standalone <svg> element: no XML declaration, no <style>, no script, no
    external fonts. Colour comes from the host page through currentColor, so the
    same string renders correctly in a light or dark HTML page.
    """
    pad, x = 12.0, 24.0
    avail = CANVAS_W - x - 20.0  # keep a right margin

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 440" '
        'width="320" height="440" fill="currentColor" font-family="sans-serif">',
        '<rect x="0.5" y="0.5" width="319" height="439" fill="none" '
        'stroke="currentColor" stroke-opacity="0.3" stroke-width="1"/>',
    ]

    for i, role in enumerate(ROLES):
        size, y, weight, contrast = (float(v) for v in layout[i])
        # wrap at roughly 30 characters, but never wider than the frame
        max_chars = max(4, min(30, int(avail / (CHAR_W * size))))
        lines = _wrap(TEXT[role], max_chars)

        line_h = 1.15 * size
        block = (len(lines) - 1) * line_h
        lo_base = pad + 0.78 * size                      # room for the ascender
        hi_base = CANVAS_H - pad - block - 0.22 * size   # room for the descender
        base = y / 100.0 * CANVAS_H
        base = lo_base if hi_base < lo_base else min(max(base, lo_base), hi_base)

        out.append(
            f'<text x="{_fmt(x)}" y="{_fmt(base)}" font-size="{_fmt(size)}" '
            f'font-weight="{int(round(weight))}" fill="currentColor" '
            f'fill-opacity="{_fmt(contrast)}" data-role="{role}">'
        )
        for k, line in enumerate(lines):
            out.append(f'<tspan x="{_fmt(x)}" y="{_fmt(base + k * line_h)}">'
                       f'{_escape(line)}</tspan>')
        out.append("</text>")

    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def start_layout():
    """Deliberately mediocre: near-uniform emphasis, so the order a viewer
    recovers carries almost no information about the intent."""
    return np.array([
        [16.0, 74.0, 380.0, 0.35],   # title    -- small, low, faint
        [18.0, 86.0, 400.0, 0.40],   # subtitle -- pushed to the bottom
        [22.0, 46.0, 500.0, 0.75],   # body     -- takes the centre
        [17.0, 60.0, 400.0, 0.45],   # cta      -- lost in the middle
        [30.0, 26.0, 600.0, 0.90],   # footer   -- loudest thing on the page
    ])


def propose(layout, rng):
    return np.clip(layout + rng.normal(0.0, STEP_FRAC, (N_ROLES, 4)) * RANGE, LO, HI)


def run(intent, intent_rank, battery, corrupt_from, capture_rounds):
    """
    One hill-climbing run. Both runs get the same seed and the same starting
    layout; the leak coin has its own stream, so rounds 1..8 are identical.
    """
    rng_prop = np.random.default_rng(RUN_SEED + 1)    # proposals
    rng_judge = np.random.default_rng(RUN_SEED + 2)   # judge's observer noise
    rng_leak = np.random.default_rng(RUN_SEED + 3)    # contamination coin
    rng_truth = np.random.default_rng(RUN_SEED + 4)   # ground-truth readout only
    rng_probe = np.random.default_rng(RUN_SEED + 5)   # ruler probes of the judge
    rng_probe_leak = np.random.default_rng(RUN_SEED + 6)

    judge = Judge(intent, intent_rank, rng_judge, rng_leak,
                  rng_probe, rng_probe_leak, corrupt_from)

    cur = start_layout()
    judge.round = 0
    cur_score = judge.score(cur)

    rounds, posters = [], {}
    for r in range(1, ROUNDS + 1):
        judge.round = r
        cand = propose(cur, rng_prop)
        # Both sides are re-scored fresh every round. Scoring the incumbent once
        # and keeping that number lets a lucky-high stale score block every real
        # improvement for the rest of the run -- a bug in the loop, not a fact
        # about the problem.
        cur_score = judge.score(cur)
        cand_score = judge.score(cand)
        # Plateau moves are accepted. With a leaking judge that is exactly how a
        # loop keeps "improving" while wandering off the good configuration:
        # once the judge stops looking, almost every proposal ties at 1.0.
        if cand_score >= cur_score:
            cur, cur_score = cand, cand_score

        reported = cur_score                              # what a team watches
        true = fidelity(cur, intent_rank, rng_truth)      # what they cannot see
        resolution, deficit, hit, floor = ruler_readout(judge, battery)

        rounds.append({
            "round": r,
            "reported": round(float(reported), 4),
            "true": round(float(true), 4),
            "resolution": round(float(resolution), 4),
            "deficit": round(float(deficit), 4),
            "hit": round(float(hit), 4),
            "floor": round(float(floor), 4),
        })
        if r in capture_rounds:
            posters[str(r)] = render_svg(cur)

    return {"rounds": rounds, "posters": posters}


def main():
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))

    rng_setup = np.random.default_rng(SETUP_SEED)
    # The intent is arbitrary by construction -- any permutation is a valid
    # thing to ask a layout to deliver. A conventional one is chosen here so a
    # reader can see at a glance whether the poster delivers it; a random
    # permutation (e.g. subtitle first, title last) exercises the machinery
    # identically but produces posters that look wrong for reasons that have
    # nothing to do with what is being measured.
    intent_idx = tuple(ROLES.index(r) for r in
                       ("title", "subtitle", "cta", "body", "footer"))
    intent_rank = rank_of(intent_idx)
    intent_names = [ROLES[i] for i in intent_idx]
    battery = build_battery(rng_setup)

    capture = {1, 51, 80}
    honest = run(intent_idx, intent_rank, battery, None, capture)
    corrupted = run(intent_idx, intent_rank, battery, CORRUPT_FROM, capture)
    corrupted["corrupt_from"] = CORRUPT_FROM

    a, b = battery["real"][0]
    demo = {
        "intent": intent_names,
        "honest": honest,
        "corrupted": corrupted,
        "battery_example": {"a": render_svg(a), "b": render_svg(b)},
    }

    out_path = os.path.join(here, "demo.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=1)

    # --------------------------------------------------------------- verify
    def at(res, r):
        return res["rounds"][r - 1]

    TAIL = 20

    def tail_mean(res, key):
        return float(np.mean([row[key] for row in res["rounds"][-TAIL:]]))

    def tail_corr(res):
        a = np.array([row["reported"] for row in res["rounds"][-TAIL:]])
        b = np.array([row["true"] for row in res["rounds"][-TAIL:]])
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print(f"intent (ground truth by construction): {' -> '.join(intent_names)}")
    print()
    MID, END = CORRUPT_FROM, ROUNDS
    print("honest run -- true_score must RISE")
    for r in (1, MID, END):
        row = at(honest, r)
        print(f"  round {r:>2}: true={row['true']:.3f}   reported={row['reported']:.3f}")
    print()
    print(f"corrupted run (leaks from round {CORRUPT_FROM}) -- reported rises, true must not")
    for r in (1, MID, (MID + END) // 2, END):
        row = at(corrupted, r)
        print(f"  round {r:>2}: reported={row['reported']:.3f}   true={row['true']:.3f}")
    print()
    print(f"tail window (last {TAIL} rounds) -- the robust comparison")
    print(f"  honest    true {tail_mean(honest,'true'):.3f}   reported {tail_mean(honest,'reported'):.3f}")
    print(f"  corrupted true {tail_mean(corrupted,'true'):.3f}   reported {tail_mean(corrupted,'reported'):.3f}"
          f"   corr(reported,true) = {tail_corr(corrupted):+.2f}")
    print()
    print(f"ruler resolution -- honest flat, corrupted collapses after round {CORRUPT_FROM}")
    for r in (1, MID, (MID + END) // 2, END):
        h, c = at(honest, r), at(corrupted, r)
        print(f"  round {r:>2}: honest res={h['resolution']:.3f} deficit={h['deficit']:.3f}"
              f"   corrupted res={c['resolution']:.3f} deficit={c['deficit']:.3f}"
              f"   [corrupted hit={c['hit']:.2f} floor={c['floor']:.2f}]")
    print()

    with open(out_path, encoding="utf-8") as f:
        parsed = json.load(f)
    svgs = (list(parsed["honest"]["posters"].values())
            + list(parsed["corrupted"]["posters"].values())
            + [parsed["battery_example"]["a"], parsed["battery_example"]["b"]])
    assert len(svgs) == 8, len(svgs)
    bad = [s[:20] for s in svgs if not (s.startswith("<svg") and s.endswith("</svg>"))]
    assert not bad, bad
    assert set(parsed["honest"]["posters"]) == {str(r) for r in capture}
    assert parsed["corrupted"]["corrupt_from"] == CORRUPT_FROM
    print(f"demo.json parses; {len(svgs)} SVG strings all start '<svg' and end '</svg>'")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")

    checks = [
        ("honest true rises from draft to final (>= +0.30)",
         at(honest, END)["true"] - at(honest, 1)["true"] >= 0.30),
        ("corrupted reported reaches 1.0",
         at(corrupted, END)["reported"] >= 0.999 > at(corrupted, MID - 1)["reported"]),
        # After corruption the true score is a RANDOM WALK: it wanders, and any
        # single round is one draw from it. So the checks are on the tail
        # window, never on a hand-picked round.
        ("corrupted tail mean falls >= 0.20 below its pre-corruption peak",
         tail_mean(corrupted, "true") <= max(row["true"] for row in
                                             corrupted["rounds"][:MID - 1]) - 0.20),
        ("corrupted tail far below honest tail while REPORTING more",
         tail_mean(corrupted, "true") <= tail_mean(honest, "true") - 0.25
         and tail_mean(corrupted, "reported") >= tail_mean(honest, "reported")),
        ("reported and true decoupled after corruption (|corr| < 0.4)",
         abs(tail_corr(corrupted)) < 0.4),
        ("honest resolution stays high",
         min(row["resolution"] for row in honest["rounds"]) >= 0.80),
        ("corrupted resolution collapses after corruption",
         max(row["resolution"] for row in corrupted["rounds"][CORRUPT_FROM:]) <= 0.20),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    elapsed = time.time() - t0
    print(f"  [{'PASS' if elapsed < 60 else 'FAIL'}] runs in under 60s (elapsed {elapsed:.2f}s)")
    return 0 if all(ok for _, ok in checks) and elapsed < 60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
