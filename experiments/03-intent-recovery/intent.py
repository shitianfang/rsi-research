"""Intent -> implementation -> budgeted recovery -> comparison.

A domain-general instantiation of the loop, deliberately NOT in a design domain
so that the truth is independently computable and the experiment can be checked.

    intent S        a qualitative structure, machine-authored
    implementation  a concrete artifact that realises S (many do)
    observation     probe the artifact within a BUDGET (number of calls)
    recovery S-hat  infer the structure back from what the probes returned
    comparison      dist(S, S-hat)

Here the artifact is a piecewise-constant program on [0,1]: breakpoints split
the domain into segments, each segment returns a value. The INTENT is the
qualitative structure -- how many segments and their rank order (e.g. "low,
high, medium") -- and many numerically different implementations realise the
same intent. They differ in how RECOVERABLE that intent is from a limited
number of calls, which is exactly the property under study.

Two verifier SHAPES are compared, both measured from the same probe data, so
the only difference is shape:

    scalar  v(A) = fidelity at the current budget          -- a POINT
    metric  v(A) = JND volume of correct recovery around A -- a VOLUME

Central prediction: optimising the point overfits the budget it was measured
at; optimising the volume transfers to budgets never seen.

Run:  ../../.venv/bin/python -u intent.py
"""

import numpy as np

K_SEG = 3                      # segments per program
NOISE = 0.06                   # observation noise on every probe


# ---------------------------------------------------------------------------
# 1. Intent and implementation
# ---------------------------------------------------------------------------

def random_intent(rng, k=K_SEG):
    """The intent: a rank order over segments. 'segment 2 is the highest,
    segment 0 the lowest, ...'. Purely qualitative, machine-authored."""
    return tuple(rng.permutation(k).tolist())


def random_impl(rng, intent):
    """A concrete program realising that intent: breakpoint positions and
    segment values whose rank order equals `intent`."""
    k = len(intent)
    b = np.sort(rng.uniform(0.05, 0.95, size=k - 1))
    # Levels are spread with a guaranteed gap, then jittered: a typical program
    # is recoverable at a loose budget and not at a tight one, so the measurement
    # has dynamic range in the region the loop actually works in.
    lv = np.linspace(0.10, 0.90, k) + rng.uniform(-0.08, 0.08, size=k)
    lv = np.sort(np.clip(lv, 0.0, 1.0))
    v = np.empty(k)
    for seg, rank in enumerate(intent):
        v[seg] = lv[rank]
    return dict(b=b, v=v, intent=intent)


def params(impl):
    return np.concatenate([impl["b"], impl["v"]])


def from_params(p, intent):
    k = len(intent)
    b = np.sort(np.clip(p[:k - 1], 0.02, 0.98))
    v = np.clip(p[k - 1:], 0.0, 1.0)
    return dict(b=b, v=v, intent=intent)


def perturb(impl, rng, scale=0.05, direction=None):
    p = params(impl)
    d = rng.normal(size=p.shape) if direction is None else direction
    d = d / (np.linalg.norm(d) + 1e-12)
    return from_params(p + d * scale, impl["intent"]), d


def evaluate(impl, x):
    idx = np.searchsorted(impl["b"], x)
    return impl["v"][idx]


# ---------------------------------------------------------------------------
# 2. Budgeted observation and recovery
# ---------------------------------------------------------------------------

def probe(impl, budget, rng):
    """Spend `budget` calls. Probes are placed by the prober, which does NOT
    know where the breakpoints are: uniform grid + jitter."""
    x = np.clip((np.arange(budget) + 0.5) / budget + rng.normal(scale=0.3 / budget, size=budget), 0, 1)
    y = evaluate(impl, x) + rng.normal(scale=NOISE, size=budget)
    return x, y


def recover(x, y, thresh=3.0):
    """Infer the structure from the samples alone.

    Change detection compares the mean of a window on each side of every
    candidate cut, with the threshold set at `thresh` standard errors of that
    difference. Averaging over a window is what makes a LARGER budget genuinely
    more informative -- a single-pair difference test does not improve with
    budget, so its false-positive count grows and recovery gets worse, which is
    a real bug this replaced. Thin segments (narrower than the window) are
    still missed, which is exactly the difficulty signal under study."""
    o = np.argsort(x)
    x, y = x[o], y[o]
    n = len(y)
    w = max(1, n // 8)
    if n < 2 * w + 1:
        return (0,)
    se = NOISE * np.sqrt(2.0 / w)
    score = np.array([abs(y[i - w:i].mean() - y[i:i + w].mean()) for i in range(w, n - w + 1)])
    cand = np.where(score > thresh * se)[0]
    cuts = []
    for c in cand[np.argsort(-score[cand])]:            # strongest first
        if all(abs(c - k) >= w for k in cuts):          # non-maximum suppression
            cuts.append(int(c))
    cuts = sorted(c + w for c in cuts)
    groups, start = [], 0
    for c in cuts + [n]:
        if c > start:
            groups.append(y[start:c])
        start = c
    groups = [g for g in groups if len(g)]
    if not groups:
        return (0,)
    means = np.array([g.mean() for g in groups])
    return tuple(np.argsort(np.argsort(means)).tolist())


def structure_match(intent, got):
    """Graded comparison of the intended structure against the recovered one:
    same number of segments AND same rank order = 1.0; partial credit via the
    longest common subsequence of the two rank sequences."""
    n, m = len(intent), len(got)
    if m == 0:
        return 0.0
    L = np.zeros((n + 1, m + 1))
    for i in range(n):
        for j in range(m):
            L[i + 1, j + 1] = L[i, j] + 1 if intent[i] == got[j] else max(L[i, j + 1], L[i + 1, j])
    lcs = L[n, m] / n
    return float(lcs * (1.0 if n == m else 0.6))        # penalty for wrong count


def fidelity(impl, budget, rng, reps=6):
    """THE measurement. No opinion enters: we wrote the intent, so the
    comparison is mechanical."""
    return float(np.mean([structure_match(impl["intent"], recover(*probe(impl, budget, rng)))
                          for _ in range(reps)]))


# ---------------------------------------------------------------------------
# 3. The metric: JND under budgeted recovery
# ---------------------------------------------------------------------------

def discriminable(a, b, budget, rng, reps=8):
    """Can a budgeted observer tell these two implementations apart? Compares
    the recovered structures -- a same/different judgment, never a quality one."""
    diff = 0
    for _ in range(reps):
        ra = recover(*probe(a, budget, rng))
        rb = recover(*probe(b, budget, rng))
        diff += int(ra != rb)
    return diff / reps


def jnd_radius(impl, direction, budget, rng, lo=0.005, hi=0.6, steps=7, target=0.5):
    """Bisect for the perturbation size at which the budgeted observer is right
    half the time. This is the MacAdam construction: t* is one unit in this
    direction, and it needs no ground truth about quality."""
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        d = discriminable(impl, perturb(impl, rng, scale=mid, direction=direction)[0], budget, rng, reps=6)
        if d < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def jnd_profile(impl, budget, rng, n_dirs=8):
    p = params(impl)
    dirs = rng.normal(size=(n_dirs, len(p)))
    return np.array([jnd_radius(impl, d, budget, rng) for d in dirs])


def robustness_volume(impl, budget, rng, n_dirs=6, scale=0.09):
    """The METRIC-shaped objective: how much room is there around this
    implementation in which the intent is still recovered correctly?
    A volume, not a point. Estimated as the fraction of perturbed neighbours
    that still recover the intent, geometric-mean weighted by fidelity."""
    p = params(impl)
    keep = []
    for _ in range(n_dirs):
        d = rng.normal(size=len(p))
        nb, _ = perturb(impl, rng, scale=scale, direction=d)
        keep.append(fidelity(nb, budget, rng, reps=3))
    return float(np.exp(np.mean(np.log(np.array(keep) + 1e-3))))


# ---------------------------------------------------------------------------
# 4. Budget calibration: the staircase (this IS the escalation rule)
# ---------------------------------------------------------------------------

def staircase(budget, pass_rate, target=0.5, lo=3, hi=40):
    """Move the budget so the population's pass rate sits at 50%, where a
    binary measurement carries the most information. Nothing external is
    consulted -- only how the current population is doing -- so difficulty
    rises exactly when the population gets good."""
    if pass_rate > target + 0.08:
        budget -= 1
    elif pass_rate < target - 0.08:
        budget += 1
    return int(np.clip(budget, lo, hi))


# ---------------------------------------------------------------------------
# 5. The loop
# ---------------------------------------------------------------------------

def run(rng, shape, escalate, T=60, K=6, budget0=14, tries=3):
    """shape in {'scalar','metric'}; escalate toggles the staircase."""
    intents = [random_intent(rng) for _ in range(K)]
    pop = [random_impl(rng, s) for s in intents]
    budget = budget0
    traj = []

    def value(impl):
        if shape == "scalar":
            return fidelity(impl, budget, rng, reps=5)          # a POINT
        return robustness_volume(impl, budget, rng)             # a VOLUME

    for t in range(T):
        i = int(rng.integers(0, K))
        cur = value(pop[i])
        for _ in range(tries):
            cand, _ = perturb(pop[i], rng, scale=float(rng.uniform(0.02, 0.14)))
            v = value(cand)
            if v > cur:
                pop[i], cur = cand, v
        if escalate:
            pr = float(np.mean([fidelity(s, budget, rng, reps=4) > 0.99 for s in pop]))
            budget = staircase(budget, pr)
        traj.append((t, budget, float(np.mean([value(s) for s in pop]))))
    return pop, budget, traj


def ladder_truth(pop, rng, ladder=(4, 6, 8, 10, 14, 20)):
    """Held-out evaluation: true fidelity across a budget ladder, including
    budgets an arm never trained at. This is the quantity no arm optimises."""
    return [float(np.mean([fidelity(s, b, rng, reps=8) for s in pop])) for b in ladder]


# ---------------------------------------------------------------------------

LADDER = (4, 6, 8, 10, 14, 20)
N_SEEDS = 8


def main():
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("INTENT -> IMPLEMENTATION -> BUDGETED RECOVERY -> COMPARISON")
    print("=" * 78)

    print("\n[E-0] Does the budget actually control difficulty? (20 random programs)")
    print(f"   {'budget':>7} {'mean fidelity':>15} {'sd':>7} {'fully recovered':>17}")
    for b in LADDER + (30,):
        f = np.array([fidelity(random_impl(np.random.default_rng(s), random_intent(np.random.default_rng(s))),
                               b, rng, reps=8) for s in range(20)])
        print(f"   {b:>7} {f.mean():>15.3f} {f.std():>7.3f} {(f > 0.99).mean():>17.2f}")
    print("   -> a loose budget recovers most programs (little signal left);")
    print("      a tight budget separates them. The limit IS the signal.")

    probe_impl = random_impl(rng, random_intent(rng))
    print("\n[E-A] Is the metric anisotropic? (if it were isotropic it would add nothing)")
    for b in (6, 12):
        prof = jnd_profile(imp, b, rng, n_dirs=8)
        print(f"   budget {b:>2}: JND per direction " + " ".join(f"{v:.3f}" for v in np.sort(prof)) +
              f"   ratio max/min = {prof.max()/max(prof.min(),1e-9):.1f}x")

    print("\n[E-B] Does the staircase find the 50% point on its own?")
    pop = [random_impl(rng, random_intent(rng)) for _ in range(6)]
    b = 20
    hist = []
    for _ in range(25):
        pr = float(np.mean([fidelity(s, b, rng, reps=4) > 0.99 for s in pop]))
        hist.append((b, pr))
        b = staircase(b, pr)
    print("   budget/pass-rate trace: " + "  ".join(f"{bb}:{pp:.2f}" for bb, pp in hist[::3]))
    print(f"   settled at budget {b}")

    print("\n[E-C] Four arms x %d seeds. Truth = fidelity across the whole budget ladder,"
          % N_SEEDS)
    print(f"      which NO arm optimises (each is only ever scored at its current budget).")
    print(f"   ladder {LADDER}")
    print(f"\n   {'arm':<17}{'ladder mean':>14}{'far-from-training':>20}{'final budget':>15}")
    summary = {}
    for shape in ("scalar", "metric"):
        for esc in (False, True):
            name = f"{shape}{'+escalate' if esc else '+fixed'}"
            lad, far, bf = [], [], []
            for seed in range(N_SEEDS):
                r = np.random.default_rng(100 + seed)
                pop, bfinal, traj = run(r, shape, esc, T=60)
                truth = np.array(ladder_truth(pop, r))
                lad.append(truth.mean())
                # budgets at least 2x away from where this arm ended up training
                far_idx = [i for i, b in enumerate(LADDER)
                           if b >= 2 * bfinal or b <= 0.5 * bfinal]
                far.append(truth[far_idx].mean() if far_idx else np.nan)
                bf.append(bfinal)
            summary[name] = (np.array(lad), np.array(far), np.array(bf))
            fa = np.array(far)[~np.isnan(far)]
            print(f"   {name:<17}{np.mean(lad):>8.3f} ±{np.std(lad):.3f}"
                  f"{np.mean(fa):>13.3f} ±{np.std(fa):.3f}"
                  f"{np.mean(bf):>10.1f} ±{np.std(bf):.1f}")

    print("\n   paired contrasts (same seeds, so the comparison is paired):")
    def contrast(a, b, idx=0):
        da = summary[a][idx] - summary[b][idx]
        se = da.std(ddof=1) / np.sqrt(len(da)) if len(da) > 1 else float("nan")
        return da.mean(), se
    for lbl, a, b in (("metric vs scalar, escalating ", "metric+escalate", "scalar+escalate"),
                      ("metric vs scalar, fixed      ", "metric+fixed", "scalar+fixed"),
                      ("escalating vs fixed, metric  ", "metric+escalate", "metric+fixed"),
                      ("escalating vs fixed, scalar  ", "scalar+escalate", "scalar+fixed")):
        d, se = contrast(a, b)
        star = "significant" if abs(d) > 2 * se else "NOT significant"
        print(f"     {lbl} Δladder = {d:+.3f} ± {se:.3f} (2se)  -> {star}")
    print("\n" + "-" * 78)
    print("read: 'far-from-training' averages only budgets at least 2x away from where")
    print("      that arm's staircase settled, so it cannot be earned by training there.")


if __name__ == "__main__":
    main()
