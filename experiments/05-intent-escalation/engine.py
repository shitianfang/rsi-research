"""E-E: is task-space deepening the engine of continued improvement?

THE CLAIM UNDER TEST. The line's original engine claim -- tighten the
judge's budget and resolution compounds -- was killed by the four-arm
control in experiments/03: escalating and fixed budgets improved
identically. Two independent findings then pointed at a different engine:
our own saturation (a 3-segment intent space has 6 members and is
exhausted almost immediately) and Survive or Collapse (2605.22217), whose
controlled matrix found the binding constraint on self-play stability to
be task admission, with a proposer-capacity ceiling -- run out of genuinely
new problems and the loop stagnates regardless of reward design.

So the surviving hypothesis is: what keeps a self-improving loop improving
is that its TASK GENERATOR keeps producing genuinely harder tasks, and --
the part that is not tautological -- that the resulting CURRICULUM is
NECESSARY, i.e. you cannot reach the same skill by training on the hard
tasks directly at equal compute. That necessity is the claim this file
tests. It is the same shape as the auto-curriculum literature (PowerPlay,
POET, GoalGAN: generate tasks in a success-rate band and show transfer to
held-out hard tasks); the contribution here is a domain with an
independently computable truth, so the learner cannot fool itself.

THE DOMAIN. A hierarchical piecewise-constant program on [0,1]. A binary
tree of depth D: every node splits its interval at a jittered midpoint,
and its two children take values separated by a gap. The INTENT is the
tree of orientations (at each node, which side is higher) -- machine-
authored, hence ground truth by construction. A budgeted observer probes
the program and recovers the tree TOP-DOWN: it must localise a node's
split before it can look inside either child, so a node recovered wrong
destroys everything beneath it. That cascade is the stepping-stone
structure the claim needs; without it a curriculum could not matter.

THE SKILL. Value range is finite, so the policy must ALLOCATE it across
levels: theta = (gap_1 ... gap_D), sum bounded. Deep levels hold fewer
samples per cell, so they need larger gaps to survive noise -- but every
unit spent deep is taken from shallow. There is a real optimum and it is
not obvious. Crucially, gap_l only receives signal when levels above it
are already recovered correctly, which is why depth-D tasks give a
randomly-initialised policy almost no gradient.

THE ARMS, all at equal total evaluations:
  A  escalate   start at depth 1; deepen when the success rate clears a
                band, exactly the minimal-criterion rule of the curriculum
                literature
  C  direct     train at depth D from the start -- the arm that decides
                whether the curriculum is NECESSARY. This is the primary
                comparison.
  B  shallow    never leaves depth 1 -- the saturation floor, present to
                show the task space really does run dry

PRE-REGISTERED CRITERIA (written before the first run; the verdict is read
once, from the full seed set):
  primary    A's held-out depth-D fidelity exceeds C's by more than 2
             standard errors, paired by seed  -> the curriculum is
             necessary, and task-space deepening is a real engine
  failure    C >= A within 2 se -> training directly on hard tasks does as
             well, the curriculum buys nothing, and the engine claim dies
             in this domain too
  secondary  A > B by more than 2 se (the task space does saturate)
  discipline paired contrasts, 8 seeds, differences below 2 se are written
             "not significant" and not interpreted by direction

INSTRUMENT GATES (must pass before any verdict is read; these exist
because three earlier rounds in this project nearly read numbers off
broken code):
  G1  the observer's recovery degrades with depth at fixed budget -- if it
      does not, depth is not a difficulty axis and nothing here is a test
  G2  a random policy scores near chance at depth D -- if it does not, the
      task is already solved and there is nothing to learn
  G3  the learner actually learns: arm A's depth-1 fidelity rises above
      its own initialisation
  G4  equal compute: every arm spends the same number of evaluations

Run:  ../../.venv/bin/python -u engine.py gates     (instrument gates only)
      ../../.venv/bin/python -u engine.py           (registered run)
"""

import sys
import numpy as np

DEPTH = 4                 # held-out evaluation depth; 2^4 = 16 leaves
BUDGET = 64               # probes per program
NOISE = 0.05
EVALS = 900               # total program evaluations per arm (equal compute)
N_SEEDS = 8
HELDOUT = 24              # fresh depth-D intents for the held-out measure


# --- intent, policy, program -------------------------------------------------

def random_intent(rng, depth):
    """Orientation at every internal node of a full binary tree."""
    return [rng.integers(0, 2, size=2 ** l).tolist() for l in range(depth)]


def build(intent, theta, rng):
    """Realise the intent. theta[l] is the value gap spent at level l;
    splits are jittered so the observer must localise them."""
    depth = len(intent)
    segs = [(0.0, 1.0, 0.5)]                    # (lo, hi, value)
    for l in range(depth):
        nxt = []
        for i, (lo, hi, v) in enumerate(segs):
            cut = lo + (hi - lo) * (0.5 + rng.uniform(-0.12, 0.12))
            g = theta[l] / 2.0
            hi_first = intent[l][i] == 1
            nxt.append((lo, cut, v + (g if hi_first else -g)))
            nxt.append((cut, hi, v + (-g if hi_first else g)))
        segs = nxt
    return segs


def evaluate(segs, x):
    y = np.empty_like(x)
    for lo, hi, v in segs:
        m = (x >= lo) & (x < hi)
        y[m] = v
    y[x >= segs[-1][1] - 1e-12] = segs[-1][2]
    return y


def probe(segs, budget, rng):
    x = np.sort(rng.uniform(0, 1, size=budget))
    return x, evaluate(segs, x) + rng.normal(scale=NOISE, size=budget)


# --- top-down recovery: a wrong split destroys everything beneath it ----------

def recover(x, y, depth):
    out = []
    frontier = [(0.0, 1.0)]
    for _ in range(depth):
        level, nxt = [], []
        for lo, hi in frontier:
            m = (x >= lo) & (x < hi)
            xs, ys = x[m], y[m]
            if len(ys) < 4:
                level.append(0)
                mid = 0.5 * (lo + hi)
                nxt += [(lo, mid), (mid, hi)]
                continue
            best, cut = -1.0, 0.5 * (lo + hi)
            for j in range(2, len(ys) - 1):
                d = abs(ys[:j].mean() - ys[j:].mean())
                if d > best:
                    best, cut = d, 0.5 * (xs[j - 1] + xs[j])
            left = ys[xs < cut]
            right = ys[xs >= cut]
            level.append(1 if (len(left) and len(right) and left.mean() > right.mean()) else 0)
            nxt += [(lo, cut), (cut, hi)]
        out.append(level)
        frontier = nxt
    return out


def fidelity(intent, got):
    """Fraction of orientations recovered, averaged over levels so that a
    deep tree is not dominated by its widest level."""
    per = []
    for l in range(len(intent)):
        a, b = np.array(intent[l]), np.array(got[l])
        per.append(float((a == b).mean()))
    return float(np.mean(per))


def score(intent, theta, rng, reps=3):
    return float(np.mean([fidelity(intent, recover(*probe(build(intent, theta, rng), BUDGET, rng),
                                                   len(intent))) for _ in range(reps)]))


# --- policy: allocate a bounded value range across levels ---------------------

BUDGET_SUM = 0.9


def norm(theta):
    t = np.clip(theta, 0.02, None)
    s = t.sum()
    return t * (BUDGET_SUM / s) if s > BUDGET_SUM else t


def init_theta(rng):
    return norm(rng.uniform(0.05, 0.15, size=DEPTH))


def train(rng, arm, evals=EVALS):
    """One arm. Hill-climb theta on tasks drawn at the arm's current depth.
    Every arm spends exactly `evals` program evaluations."""
    theta = init_theta(rng)
    depth = {"escalate": 1, "direct": DEPTH, "shallow": 1}[arm]
    cur, spent, trace = None, 0, []
    while spent < evals:
        intents = [random_intent(rng, depth) for _ in range(4)]
        if cur is None:
            cur = float(np.mean([score(s, theta, rng) for s in intents]))
            spent += 4
            continue
        cand = norm(theta + rng.normal(scale=0.03, size=DEPTH))
        val = float(np.mean([score(s, cand, rng) for s in intents]))
        spent += 4
        if val > cur:
            theta, cur = cand, val
        if arm == "escalate" and val > 0.85 and depth < DEPTH:
            depth += 1                      # minimal-criterion deepening
            cur = None
        trace.append((spent, depth, cur if cur is not None else val))
    return theta, trace


def heldout(theta, rng, n=HELDOUT):
    """The measure no arm optimises: fresh depth-D intents, policy only."""
    return float(np.mean([score(random_intent(rng, DEPTH), theta, rng, reps=4)
                          for _ in range(n)]))


# --- instrument gates --------------------------------------------------------

def gates():
    rng = np.random.default_rng(0)
    print("=" * 76)
    print("E-E INSTRUMENT GATES")
    print("=" * 76)
    flat = norm(np.full(DEPTH, BUDGET_SUM / DEPTH))
    print("\n[G1] does depth actually make recovery harder at fixed budget?")
    prev, ok1 = 1.1, True
    for d in range(1, DEPTH + 1):
        f = np.mean([score(random_intent(rng, d), flat[:d], rng, reps=4) for _ in range(20)])
        print(f"   depth {d}: fidelity {f:.3f}")
        ok1 &= f <= prev + 0.02
        prev = f
    print(f"   -> {'OK' if ok1 else 'FAIL'} (must be non-increasing)")

    print("\n[G2] is a random policy near chance at depth D?")
    r = np.mean([heldout(init_theta(np.random.default_rng(s)), rng, n=8) for s in range(6)])
    ok2 = r < 0.75
    print(f"   random-policy held-out fidelity {r:.3f} (chance = 0.500) -> {'OK' if ok2 else 'FAIL'}")

    print("\n[G3] does the learner learn? (arm A at its starting depth)")
    rr = np.random.default_rng(1)
    th0 = init_theta(rr)
    base = np.mean([score(random_intent(rr, 1), th0[:1], rr, reps=4) for _ in range(12)])
    th, _ = train(np.random.default_rng(1), "escalate", evals=300)
    after = np.mean([score(random_intent(rr, 1), th[:1], rr, reps=4) for _ in range(12)])
    ok3 = after > base + 0.02
    print(f"   depth-1 fidelity {base:.3f} -> {after:.3f} -> {'OK' if ok3 else 'FAIL'}")
    print(f"\n   gates {'PASSED' if (ok1 and ok2 and ok3) else 'NOT passed'}")


# --- registered run ----------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "gates":
        gates()
        return
    print("=" * 76)
    print("E-E: IS TASK-SPACE DEEPENING THE ENGINE?")
    print("=" * 76)
    res = {a: [] for a in ("escalate", "direct", "shallow")}
    depths = []
    for s in range(N_SEEDS):
        for arm in res:
            th, trace = train(np.random.default_rng([s, 7]), arm)
            res[arm].append(heldout(th, np.random.default_rng([s, 99])))
            if arm == "escalate":
                depths.append(trace[-1][1])
    print(f"\n   held-out depth-{DEPTH} fidelity, {N_SEEDS} seeds, {EVALS} evaluations per arm")
    print(f"   {'arm':<12}{'held-out':>16}")
    for a, lbl in (("escalate", "A escalate"), ("direct", "C direct-hard"), ("shallow", "B shallow")):
        v = np.array(res[a])
        print(f"   {lbl:<12}{v.mean():>10.3f} ±{v.std():.3f}")
    print(f"   arm A final depth per seed: {depths}")

    def contrast(a, b):
        d = np.array(res[a]) - np.array(res[b])
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    print("\n   paired contrasts:")
    dp, sep = contrast("escalate", "direct")
    ds, ses = contrast("escalate", "shallow")
    print(f"     PRIMARY    A - C = {dp:+.3f} (2se {2*sep:.3f})  -> "
          f"{'significant' if abs(dp) > 2*sep else 'NOT significant'}")
    print(f"     secondary  A - B = {ds:+.3f} (2se {2*ses:.3f})  -> "
          f"{'significant' if abs(ds) > 2*ses else 'NOT significant'}")
    print("\n" + "-" * 76)
    if dp > 2 * sep:
        print("VERDICT: the curriculum is NECESSARY -- at equal compute, deepening the")
        print("task space beats training on the hard tasks directly. Task-space")
        print("escalation is a real engine in this domain.")
    else:
        print("VERDICT: FAILURE branch -- direct training on hard tasks matches or beats")
        print("the curriculum at equal compute. The engine claim does not survive here")
        print("either; what a curriculum buys is not continued improvement.")


if __name__ == "__main__":
    main()
