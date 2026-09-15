"""A metric defined without humans, and a loop whose difficulty is endogenous.

!! STATUS 2026-09-15: the matching construction below is FALSIFIED. Measured
!! match rates between two random designs are 0.01-0.05 (only 8-24% of pairs
!! ever match at all), so `fit_metric` is fitting an almost-constant target and
!! its low fit MSE is vacuous. The error is in the analogy: CIE colour matching
!! ADJUSTS one stimulus until it matches, it does not compare two unrelated
!! stimuli. The correct construction is the MacAdam one -- perturb a design in
!! many directions and find the threshold at which the resource-limited reader
!! can just discriminate it from itself; the resulting JND ellipsoid IS the
!! local metric tensor. Rewrite fit_metric/same_structure along those lines
!! before drawing any conclusion from this file. The spec/render/read/fidelity
!! half (sections 1-2) is sound and measured fine (3ms per evaluation).

Follow-up to experiments/01-verifier-as-metric, which established that a
metric-shaped verifier resists Goodhart better than a scalar one -- but which
was NOT RSI: its verifier was built from (simulated) human comparisons and then
frozen, so the loop could only climb to that frozen verifier's optimum.

This harness removes the human entirely and makes difficulty endogenous.

THE UNLOCK -- how a metric gets defined with no human in the loop:

    CIE 1931 never asked "is this colour good". It asked "do these two look the
    same", and recovered a space from thousands of MATCHING judgments. A
    matching judgment needs an observer that CONFUSES things -- it does not need
    a human. So:

        machine writes a spec (the intent)  ->  render  ->  a resource-limited
        reader tries to recover the spec  ->  recovery error

    Ground truth exists because WE authored the spec. The signal exceeds the
    generator's prior because whether a layout survives 3 fixations at 1/4
    resolution under clutter is an empirical fact about rendering and
    constrained perception, not something the generator knows -- the same way
    you cannot know whether code works without running it.

    The metric then falls out as confusability:

        d(A, B) = how often A and B are recovered as the SAME structure,
                  across the reader ensemble.

    That is the MacAdam-ellipse construction with an algorithmic observer.

THE RSI PART: the reader is adversarial. It searches the constraint space for
settings where the current champion designs fail, so the difficulty frontier
rises as the designers improve. Designer and reader are both machines; nothing
in the loop is elicited from anyone.

SCOPE HONESTY: this measures communicative fidelity and robustness, not beauty
and not personal taste. Human data, if ever added, would enter only as an ideal
point inside an already-built space -- O(dimensions), once, and optional.

Run:  ../../.venv/bin/python -u rsi.py
"""

import numpy as np

H, W = 48, 64                      # raster size; small on purpose (C4: instant)
ROLES = ("headline", "sub", "cta", "body", "logo")
N_ROLES = len(ROLES)


# ---------------------------------------------------------------------------
# 1. Spec = machine-authored ground truth (C3: never harvested from humans)
# ---------------------------------------------------------------------------

def random_spec(rng):
    """A poster as structured intent: which roles exist, where, how strong, and
    the reading order they are MEANT to be read in."""
    n = rng.integers(3, N_ROLES + 1)
    roles = rng.permutation(N_ROLES)[:n]
    el = []
    for r in roles:
        w = float(rng.uniform(0.18, 0.55))
        h = float(rng.uniform(0.08, 0.30))
        el.append(dict(role=int(r),
                       x=float(rng.uniform(0.02, 0.98 - w)),
                       y=float(rng.uniform(0.02, 0.98 - h)),
                       w=w, h=h,
                       weight=float(rng.uniform(0.25, 1.0)),
                       freq=float(rng.uniform(0.0, 1.0))))   # texture ~ "text density"
    # intended reading order = the order the author wants attention to go
    order = list(range(len(el)))
    rng.shuffle(order)
    return dict(el=el, order=order)


def mutate_spec(spec, rng, scale=1.0):
    s = dict(el=[dict(e) for e in spec["el"]], order=list(spec["order"]))
    e = s["el"][rng.integers(0, len(s["el"]))]
    for k, lo, hi in (("x", 0.0, 0.95), ("y", 0.0, 0.95), ("w", 0.10, 0.60),
                      ("h", 0.05, 0.35), ("weight", 0.1, 1.0), ("freq", 0.0, 1.0)):
        if rng.random() < 0.45:
            e[k] = float(np.clip(e[k] + rng.normal() * 0.09 * scale, lo, hi))
    e["x"] = float(np.clip(e["x"], 0.0, 1.0 - e["w"]))
    e["y"] = float(np.clip(e["y"], 0.0, 1.0 - e["h"]))
    return s


def render(spec):
    """Deterministic rasterisation. Two channels the reader can exploit:
    fill strength (weight) and high-frequency texture (freq ~ text density)."""
    img = np.zeros((H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for e in spec["el"]:
        x0, x1 = int(e["x"] * W), int(min(W, (e["x"] + e["w"]) * W))
        y0, y1 = int(e["y"] * H), int(min(H, (e["y"] + e["h"]) * H))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = np.full((y1 - y0, x1 - x0), e["weight"])
        if e["freq"] > 0.05:                      # texture = alternating rows
            rows = np.arange(y1 - y0)[:, None]
            patch = patch * (1 - e["freq"] * 0.55 * (rows % 2))
        img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], patch)
    return img


# ---------------------------------------------------------------------------
# 2. The reader: resource-limited, algorithmic, NO learned human prior (C1/C2)
# ---------------------------------------------------------------------------

CONSTRAINT_DIMS = ("scale", "blur", "fixations", "noise", "clutter")


def random_constraint(rng, hard=0.0):
    """A point in constraint space. `hard` in [0,1] slides every axis toward
    harsher. These are PHYSICAL limits (resolution, blur, glance count, noise,
    clutter) -- nothing here is fitted to human data."""
    def mix(easy, tough):
        return float(easy + (tough - easy) * np.clip(hard + rng.normal() * 0.12, 0, 1))
    return dict(scale=mix(1.0, 0.28), blur=mix(0.4, 2.6), fixations=int(round(mix(5, 2))),
                noise=mix(0.01, 0.30), clutter=mix(0.0, 0.55))


def _blur(img, sigma):
    if sigma <= 0.05:
        return img
    r = max(1, int(sigma * 2))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)


def degrade(img, c, rng):
    s = c["scale"]
    if s < 0.99:                                   # downsample then back up
        h2, w2 = max(6, int(H * s)), max(8, int(W * s))
        yi = (np.linspace(0, H - 1, h2)).astype(int)
        xi = (np.linspace(0, W - 1, w2)).astype(int)
        small = img[np.ix_(yi, xi)]
        yb = (np.linspace(0, h2 - 1, H)).astype(int)
        xb = (np.linspace(0, w2 - 1, W)).astype(int)
        img = small[np.ix_(yb, xb)]
    img = _blur(img, c["blur"])
    if c["clutter"] > 0.01:
        img = img + c["clutter"] * rng.random((H, W)) * (rng.random((H, W)) < 0.12)
    if c["noise"] > 0.005:
        img = img + rng.normal(scale=c["noise"], size=(H, W))
    return np.clip(img, 0, 1.6)


def read(img, c, rng):
    """Bottom-up saliency + a few fixations with inhibition of return.

    Returns the recovered structure: the sequence of fixated locations and a
    crude signature at each. This is the observer -- deliberately dumb, purely
    physical, and identical for every design."""
    d = degrade(img, c, rng)
    # local contrast energy: |d - blur(d)| -- standard bottom-up saliency
    sal = np.abs(d - _blur(d, 2.0))
    sal = _blur(sal, 1.2)
    seq = []
    taken = np.zeros_like(sal, dtype=bool)
    for _ in range(max(1, c["fixations"])):
        s = np.where(taken, -1.0, sal)
        idx = int(np.argmax(s))
        y, x = idx // W, idx % W
        if s.flat[idx] <= 0:
            break
        y0, y1 = max(0, y - 5), min(H, y + 6)
        x0, x1 = max(0, x - 7), min(W, x + 8)
        patch = d[y0:y1, x0:x1]
        seq.append((y / H, x / W,
                    float(patch.mean()),                                   # strength
                    float(np.abs(np.diff(patch, axis=0)).mean())))         # texture
        taken[max(0, y - 7):min(H, y + 8), max(0, x - 9):min(W, x + 10)] = True
    return seq


def recovered_order(spec, seq):
    """Map each fixation to the spec element it landed on; the resulting
    sequence is the reading order the design ACTUALLY induces."""
    out = []
    for (fy, fx, _, _) in seq:
        hit = -1
        for i, e in enumerate(spec["el"]):
            if e["x"] <= fx <= e["x"] + e["w"] and e["y"] <= fy <= e["y"] + e["h"]:
                hit = i
                break
        if hit >= 0 and (not out or out[-1] != hit):
            out.append(hit)
    return out


def fidelity(spec, c, rng, reps=2):
    """THE ORACLE (C2: a measurement of a process, never an opinion).
    How well does the rendered artifact induce the reading order it intends?"""
    img = render(spec)
    want = spec["order"]
    tot = 0.0
    for _ in range(reps):
        got = recovered_order(spec, read(img, c, rng))
        # longest common subsequence ratio against the intended order
        n, m = len(want), len(got)
        if m == 0:
            continue
        L = np.zeros((n + 1, m + 1))
        for i in range(n):
            for j in range(m):
                L[i + 1, j + 1] = L[i, j] + 1 if want[i] == got[j] else max(L[i, j + 1], L[i + 1, j])
        tot += L[n, m] / n
    return tot / reps


# ---------------------------------------------------------------------------
# 3. The metric, defined with no human: confusability under constrained reading
# ---------------------------------------------------------------------------

def signature(spec, readers, rng):
    """Behavioural signature: what this design does to each reader in the
    ensemble. Coordinates, not a score."""
    img = render(spec)
    out = []
    for c in readers:
        seq = read(img, c, rng)
        if not seq:
            out.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        a = np.array(seq)
        out.extend([len(seq) / 5.0, a[0, 0], a[0, 1], a[:, 2].mean(), a[:, 3].mean()])
    return np.array(out)


def same_structure(sa, sb, spec_a, spec_b, readers, rng):
    """The machine's MATCHING judgment: do these two designs get read the same
    way? This is the CIE colour-matching question with an algorithmic observer."""
    ia, ib = render(spec_a), render(spec_b)
    same = 0
    for c in readers:
        oa = [h for h in recovered_order(spec_a, read(ia, c, rng))]
        ob = [h for h in recovered_order(spec_b, read(ib, c, rng))]
        same += int(oa == ob)
    return same / len(readers)


def fit_metric(specs, readers, rng, k=4, iters=400, lr=0.25):
    """Ordinal embedding from machine matching judgments alone.

    Pairs judged 'same' are pulled together, 'different' pushed apart; the
    resulting space is a metric whose distances mean CONFUSABILITY. No human
    judgement of any kind enters here."""
    n = len(specs)
    S = np.stack([signature(s, readers, rng) for s in specs])
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    pairs = pairs[:min(len(pairs), 900)]
    tgt = np.array([same_structure(None, None, specs[i], specs[j], readers, rng)
                    for i, j in pairs])
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    Z = rng.normal(size=(n, k)) * 0.3
    cnt = np.maximum(np.bincount(np.concatenate([ii, jj]), minlength=n), 1)[:, None]
    for _ in range(iters):
        d = Z[ii] - Z[jj]
        dist2 = np.sum(d ** 2, axis=1)
        pred = np.exp(-dist2)                       # confusable <=> close
        r = (pred - tgt) * (-pred)
        g = (2 * r)[:, None] * d
        gz = np.zeros_like(Z)
        for col in range(k):
            gz[:, col] = (np.bincount(ii, weights=g[:, col], minlength=n)
                          - np.bincount(jj, weights=g[:, col], minlength=n))
        Z -= lr * gz / cnt
    # map signature -> embedding so unseen designs get coordinates
    G = S.T @ S + 1e-2 * np.eye(S.shape[1])
    B = np.linalg.solve(G, S.T @ Z)
    resid = float(np.mean((pred - tgt) ** 2))
    return dict(Z=Z, B=B, mu=None, k=k, fit_mse=resid, S=S)


def embed(metric, spec, readers, rng):
    return signature(spec, readers, rng) @ metric["B"]


# ---------------------------------------------------------------------------
# 4. No-ground-truth checks on the metric itself (C6)
# ---------------------------------------------------------------------------

def check_triangle(Z, rng, n=3000):
    t = rng.integers(0, len(Z), size=(n, 3))
    d = lambda a, b: np.linalg.norm(Z[a] - Z[b], axis=1)
    viol = d(t[:, 0], t[:, 2]) > d(t[:, 0], t[:, 1]) + d(t[:, 1], t[:, 2]) + 1e-9
    return float(viol.mean())


def check_split_alignment(specs, readers, rng, k=4):
    """Refit the metric on two disjoint halves of the READER ensemble. If the
    space is real, the two halves recover the same geometry (up to rotation):
    compare their pairwise-distance matrices. Needs no ground truth."""
    half = len(readers) // 2
    outs = []
    for R in (readers[:half], readers[half:]):
        m = fit_metric(specs, R, rng, k=k, iters=250)
        D = np.linalg.norm(m["Z"][:, None, :] - m["Z"][None, :, :], axis=2)
        outs.append(D[np.triu_indices(len(specs), 1)])
    a, b = outs
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.mean(a * b))


def intrinsic_dim(Z, thresh=0.95):
    Zc = Z - Z.mean(0)
    ev = np.linalg.svd(Zc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return int(np.searchsorted(np.cumsum(ev), thresh) + 1)


# ---------------------------------------------------------------------------
# 5. The loop: designers vs an adversarial reader (C7: endogenous difficulty)
# ---------------------------------------------------------------------------

def designer_step(spec, evaluate, rng, tries=3):
    best, bv = spec, evaluate(spec)
    for _ in range(tries):
        cand = mutate_spec(spec, rng)
        v = evaluate(cand)
        if v > bv:
            best, bv = cand, v
    return best, bv


def reader_escalate(champions, hard, rng, probe=6, target=0.55):
    """The adversary: sample harder constraint points and adopt the frontier
    where champions start to fail. Difficulty is set by the designers' own
    competence -- nobody configures a schedule."""
    for _ in range(probe):
        c = random_constraint(rng, hard=min(1.0, hard + 0.08))
        f = np.mean([fidelity(s, c, rng, reps=1) for s in champions])
        if f > target:                      # still too easy -> push harder
            hard = min(1.0, hard + 0.04)
        else:
            hard = max(0.0, hard - 0.01)
    return hard


def run_arm(rng, arm, T=40, K=6, n_readers=6, seed_specs=14):
    """arm in {'metric+escalate', 'metric+fixed', 'scalar+escalate', 'scalar+fixed'}"""
    use_metric = arm.startswith("metric")
    escalate = arm.endswith("escalate")
    hard = 0.0
    pop = [random_spec(rng) for _ in range(K)]
    readers = [random_constraint(rng, hard=hard) for _ in range(n_readers)]
    metric = None
    if use_metric:
        pool = [random_spec(rng) for _ in range(seed_specs)]
        metric = fit_metric(pool, readers, rng, k=4, iters=250)

    traj = []
    for t in range(T):
        if use_metric:
            # portfolio objective in the METRIC: cover the space, i.e. be far
            # from each other while each staying readable. Coordinates, not a score.
            Zp = np.stack([embed(metric, s, readers, rng) for s in pop])

            def ev(cand, i=None):
                z = embed(metric, cand, readers, rng)
                fid = np.mean([fidelity(cand, c, rng, reps=1) for c in readers])
                others = np.delete(Zp, i, axis=0) if i is not None else Zp
                spread = float(np.min(np.linalg.norm(others - z, axis=1))) if len(others) else 0.0
                return fid + 0.35 * spread
        else:
            def ev(cand, i=None):
                return float(np.mean([fidelity(cand, c, rng, reps=1) for c in readers]))

        i = int(rng.integers(0, K))
        pop[i], _ = designer_step(pop[i], lambda s: ev(s, i), rng)

        if escalate:
            hard = reader_escalate(pop, hard, rng)
            readers = [random_constraint(rng, hard=hard) for _ in range(n_readers)]
            if use_metric and t % 8 == 7:      # the improver improves: re-fit
                pool = [random_spec(rng) for _ in range(seed_specs)]
                metric = fit_metric(pool, readers, rng, k=4, iters=200)
        traj.append(hard)
    return pop, hard, traj, metric


def holdout_transfer(pop, rng, levels=(0.35, 0.6, 0.85), n=5):
    """P4: do the evolved designs survive constraint settings never trained on?"""
    out = []
    for lv in levels:
        cs = [random_constraint(rng, hard=lv) for _ in range(n)]
        out.append(float(np.mean([fidelity(s, c, rng, reps=1) for s in pop for c in cs])))
    return out


# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("A metric with no human in it: confusability under constrained reading")
    print("=" * 74)

    readers = [random_constraint(rng, hard=0.35) for _ in range(8)]
    pool = [random_spec(rng) for _ in range(16)]
    m = fit_metric(pool, readers, rng, k=4)
    print(f"\n[metric]  fitted from {len(pool)} designs x {len(readers)} machine readers")
    print(f"          matching-judgment fit MSE       {m['fit_mse']:.4f}")
    print(f"          intrinsic dimension (95% var)   {intrinsic_dim(m['Z'])}"
          f"   <- >1 means it is NOT reducible to a scalar")
    print(f"[checks]  triangle violation rate         {check_triangle(m['Z'], rng):.4f}")
    print(f"          reader-split geometry agreement {check_split_alignment(pool, readers, rng):.3f}"
          f"   <- refit on disjoint reader halves")

    print("\n" + "=" * 74)
    print("Endogenous difficulty: does the frontier keep rising, or plateau?")
    print("=" * 74)
    rows = []
    for arm in ("metric+escalate", "scalar+escalate", "metric+fixed", "scalar+fixed"):
        r2 = np.random.default_rng(7)
        pop, hard, traj, _ = run_arm(r2, arm, T=40)
        tr = holdout_transfer(pop, r2)
        rows.append((arm, hard, traj, tr))
        print(f"\n  {arm:<16} final frontier {hard:.3f}")
        print(f"  {'':<16} frontier trace  " +
              " ".join(f"{traj[i]:.2f}" for i in range(0, len(traj), max(1, len(traj) // 8))))
        print(f"  {'':<16} held-out fidelity @hard=0.35/0.60/0.85: " +
              "  ".join(f"{v:.3f}" for v in tr))

    print("\n" + "-" * 74)
    print("P1  escalating arms should keep moving; fixed arms cannot (frontier stays 0)")
    print("P2  intrinsic dim > 1 => the metric is not a disguised scalar")
    print("P3  held-out transfer separates discovery from overfitting to the frontier")


if __name__ == "__main__":
    main()
