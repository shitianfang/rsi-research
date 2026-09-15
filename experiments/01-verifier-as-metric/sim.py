"""Does verifier-of-verifier actually predict downstream harm?

A falsification harness for notes/directions/01-verifier-as-metric.md.

The thesis under test has three parts:

  C1  A scalar verifier drifts (Goodhart) on a heterogeneous population;
      a metric verifier does not, because it has no direction to game.
  C2  Ground-truth-FREE checks (held-out comparison likelihood, subpopulation
      stability, residual heterogeneity) predict a verifier's downstream harm.
  C3  Ground truth need scales with DIMENSIONS, not with samples.

None of this can be tested on real aesthetics: there is no ground truth there.
So we build a synthetic world whose truth we define, hide that truth from every
verifier, let each verifier drive an evolution loop, and score the results
against the hidden truth. This validates the MECHANISM, not any claim about
human taste -- and to keep it from being self-confirming, every experiment is
also run under MISSPECIFICATION, i.e. in worlds whose truth violates the
ideal-point metric assumption our own proposal is built on.

Run:  .venv/bin/python sim.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. The world: truth we define and no verifier ever sees
# ---------------------------------------------------------------------------

TRUTH_MODELS = ("ideal", "multipeak", "nonmetric")


class World:
    """Items and viewers in a d-dimensional latent space.

    `truth_model` selects how preference is generated:
      ideal      -- single-peaked weighted Euclidean (our proposal's assumption)
      multipeak  -- two ideal points per viewer, utility = max (violates single peak)
      nonmetric  -- weighted L0.5 distance (violates the metric assumption)
    """

    def __init__(self, rng, d=8, n_items=300, n_viewers=200, n_features=12,
                 truth_model="ideal", noise=0.6, n_clusters=3):
        assert truth_model in TRUTH_MODELS
        self.rng, self.d, self.truth_model, self.noise = rng, d, truth_model, noise

        # Items live at X. Viewers have ideal points P drawn from clusters, so
        # the population is genuinely heterogeneous -- a scalar cannot serve it.
        self.X = rng.normal(size=(n_items, d))
        centers = rng.normal(size=(n_clusters, d)) * 1.2
        which = rng.integers(0, n_clusters, size=n_viewers)
        self.P = centers[which] + rng.normal(size=(n_viewers, d)) * 0.45
        self.cluster = which
        self.P2 = centers[(which + 1) % n_clusters] + rng.normal(size=(n_viewers, d)) * 0.45
        # Per-viewer dimension weights: taste differs in WHICH axes matter.
        self.W = np.exp(rng.normal(size=(n_viewers, d)) * 0.5)

        # Verifiers observe items only through features, never through X.
        self.A = rng.normal(size=(d, n_features)) / np.sqrt(d)
        self.feat_noise = 0.05

    def features(self, X):
        return X @ self.A + self.rng.normal(size=(len(X), self.A.shape[1])) * self.feat_noise

    def utility(self, X, viewers):
        """True utility of items X for each viewer in `viewers`. Shape (V, N)."""
        P, W = self.P[viewers], self.W[viewers]
        diff = X[None, :, :] - P[:, None, :]
        if self.truth_model == "nonmetric":
            return -np.sum(W[:, None, :] * np.abs(diff) ** 0.5, axis=2)
        u1 = -np.sum(W[:, None, :] * diff ** 2, axis=2)
        if self.truth_model == "ideal":
            return u1
        d2 = X[None, :, :] - self.P2[viewers][:, None, :]
        return np.maximum(u1, -np.sum(W[:, None, :] * d2 ** 2, axis=2))

    def comparisons(self, n, item_idx=None):
        """Sample (viewer, a, b, label) triples. label=1 means viewer prefers a."""
        pool = np.arange(len(self.X)) if item_idx is None else item_idx
        v = self.rng.integers(0, len(self.P), size=n)
        a = pool[self.rng.integers(0, len(pool), size=n)]
        b = pool[self.rng.integers(0, len(pool), size=n)]
        keep = a != b
        v, a, b = v[keep], a[keep], b[keep]
        ua = self._u_pair(a, v)   # per-pair, avoids the full V x N matrix
        ub = self._u_pair(b, v)
        p = 1.0 / (1.0 + np.exp(-(ua - ub) / self.noise))
        y = (self.rng.random(len(p)) < p).astype(np.float64)
        return v, a, b, y

    def _u_pair(self, items, viewers):
        diff = self.X[items] - self.P[viewers]
        W = self.W[viewers]
        if self.truth_model == "nonmetric":
            return -np.sum(W * np.abs(diff) ** 0.5, axis=1)
        u1 = -np.sum(W * diff ** 2, axis=1)
        if self.truth_model == "ideal":
            return u1
        d2 = self.X[items] - self.P2[viewers]
        return np.maximum(u1, -np.sum(W * d2 ** 2, axis=1))

    def coverage(self, X_portfolio):
        """THE hidden objective: mean over viewers of their best item in the
        portfolio. A campaign wants to reach people, not be everyone's blandest
        option -- this is the quantity every arm is secretly judged on."""
        u = self.utility(X_portfolio, np.arange(len(self.P)))   # (V, K)
        return float(np.mean(np.max(u, axis=1)))


# ---------------------------------------------------------------------------
# 2. Verifiers under test. None of them can see World.X, .P, .W or .utility().
# ---------------------------------------------------------------------------

def _scatter(idx, vals, n):
    """Column-wise bincount scatter-add: same as np.add.at, ~50x faster."""
    return np.stack([np.bincount(idx, weights=vals[:, j], minlength=n)
                     for j in range(vals.shape[1])], axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ScalarVerifier:
    """One global score for everyone -- autoresearch's val_bpb, an LLM judge,
    a reward model. Fit by Bradley-Terry on comparisons, viewer identity
    discarded (it has nowhere to put it)."""

    name = "scalar"
    personal = False

    def __init__(self, n_feat_used=None, lr=0.6, iters=250):
        self.k, self.lr, self.iters = n_feat_used, lr, iters

    def _F(self, F):
        return F if self.k is None else F[:, :self.k]

    def fit(self, F, comps, world=None):
        v, a, b, y = comps
        F = self._F(F)
        self.w = np.zeros(F.shape[1])
        for _ in range(self.iters):
            z = (F[a] - F[b]) @ self.w
            g = (F[a] - F[b]).T @ (_sigmoid(z) - y) / len(y)
            self.w -= self.lr * g
        return self

    def score(self, F, viewer=None):
        return self._F(F) @ self.w

    def locate_all(self, comps, n_view):   # nothing to locate
        return


class RubricVerifier(ScalarVerifier):
    """A finite-dimensional rubric: the same scalar, but it can only see k of
    the feature axes. Models the L4 rubric family -- optimising it necessarily
    squeezes the dimensions it does not cover."""
    name = "rubric"


class MetricVerifier:
    """Our proposal. Learn an item embedding and per-viewer ideal points from
    pairwise comparisons only; score = -distance to that viewer's ideal point.
    A new item is placed via a linear map fitted from features to embedding;
    a new viewer is placed by locating their ideal point from a few comparisons.
    """

    name = "metric"
    personal = True

    def __init__(self, k=6, lr=0.08, iters=1200, ridge=1e-4):
        self.k, self.lr, self.iters, self.ridge = k, lr, iters, ridge

    def fit(self, F, comps, world=None):
        v, a, b, y = comps
        rng = np.random.default_rng(0)
        n_items, n_view = len(F), int(v.max()) + 1
        Z = rng.normal(size=(n_items, self.k)) * 0.5
        Q = rng.normal(size=(n_view, self.k)) * 0.5
        cz = np.maximum(np.bincount(a, minlength=n_items)
                        + np.bincount(b, minlength=n_items), 1)[:, None]
        cq = np.maximum(np.bincount(v, minlength=n_view), 1)[:, None]
        for _ in range(self.iters):
            da, db = Z[a] - Q[v], Z[b] - Q[v]
            z = np.sum(db ** 2, axis=1) - np.sum(da ** 2, axis=1)   # u_a - u_b
            r = (_sigmoid(z) - y)[:, None]
            gZa, gZb = r * (-2 * da), r * (2 * db)
            gQ = r * (2 * da - 2 * db)
            gz = _scatter(a, gZa, len(Z)) + _scatter(b, gZb, len(Z))
            gq = _scatter(v, gQ, len(Q))
            # normalise PER PARAMETER, not by total sample count: otherwise
            # each item's step is ~n_obs/N and the embedding never moves.
            Z -= self.lr * (gz / cz + self.ridge * Z)
            Q -= self.lr * (gq / cq + self.ridge * Q)
        self.Z, self.Q = Z, Q
        # features -> embedding, so unseen items get coordinates
        G = F.T @ F + 1e-2 * np.eye(F.shape[1])
        self.B = np.linalg.solve(G, F.T @ Z)
        return self

    def embed(self, F):
        return F @ self.B

    def score(self, F, viewer=None):
        z = self.embed(F)
        if viewer is None:                       # population-mean view
            return -np.mean(((z[:, None, :] - self.Q[None, :, :]) ** 2).sum(-1), axis=1)
        return -np.sum((z - self.Q[viewer]) ** 2, axis=1)

    def locate_all(self, comps, n_view, steps=300, lr=0.5):
        """Place EVERY viewer from their own comparisons, with the item
        embedding frozen. This is the O(k)-data-per-person step: the expensive
        space is already paid for and shared."""
        v, a, b, y = comps
        Q = np.zeros((n_view, self.k))
        cnt = np.maximum(np.bincount(v, minlength=n_view), 1)[:, None]
        for _ in range(steps):
            q = Q[v]
            da, db = self.Z[a] - q, self.Z[b] - q
            z = np.sum(db ** 2, axis=1) - np.sum(da ** 2, axis=1)
            r = (_sigmoid(z) - y)[:, None]
            Q -= lr * _scatter(v, r * (2 * da - 2 * db), n_view) / cnt
        self.Q = Q


# ---------------------------------------------------------------------------
# 3. The verifier-of-verifier: checks that never touch ground truth
# ---------------------------------------------------------------------------

def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / den) if den else 0.0


def check_heldout_loglik(verifier, F, comps):
    """Predictive log-likelihood on held-out COMPARISONS. Needs no quality
    labels -- only more of the same cheap observations."""
    v, a, b, y = comps
    if verifier.personal:
        za, zb = verifier.embed(F[a]), verifier.embed(F[b])
        q = verifier.Q[v]
        z = np.sum((zb - q) ** 2, axis=1) - np.sum((za - q) ** 2, axis=1)
    else:
        z = verifier.score(F[a]) - verifier.score(F[b])
    p = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def check_subpop_stability(make_verifier, F, comps, n_view):
    """Refit on two disjoint halves of the VIEWER population and compare the
    two fits' implied population ranking of items. A verifier forcing one order
    on a heterogeneous population is unstable here -- and this needs no truth."""
    v, a, b, y = comps
    half = v < n_view // 2
    out = []
    for m in (half, ~half):
        if m.sum() < 50:
            return 0.0
        ver = make_verifier().fit(F, (v[m], a[m], b[m], y[m]))
        out.append(ver.score(F))
    return _spearman(out[0], out[1])


def check_residual_heterogeneity(verifier, F, comps):
    """Per-viewer held-out accuracy spread. If a verifier fits some viewers well
    and others badly, its shape is wrong for the population. Returned negated so
    that HIGHER IS BETTER for every check (uniform sign convention)."""
    v, a, b, y = comps
    if verifier.personal:
        za, zb = verifier.embed(F[a]), verifier.embed(F[b])
        q = verifier.Q[v]
        z = np.sum((zb - q) ** 2, axis=1) - np.sum((za - q) ** 2, axis=1)
    else:
        z = verifier.score(F[a]) - verifier.score(F[b])
    correct = ((z > 0).astype(float) == y).astype(float)
    accs = [correct[v == u].mean() for u in np.unique(v) if (v == u).sum() >= 8]
    return -float(np.std(accs)) if len(accs) > 5 else 0.0


def check_cycle_rate(verifier, F, rng, n=2000):
    """Intransitive triples. Reported for completeness: it is VACUOUS for any
    verifier whose preference comes from a scalar or a Euclidean embedding --
    both are transitive by construction. Kept so the harness shows this."""
    n_items = len(F)
    t = rng.integers(0, n_items, size=(n, 3))
    s = verifier.score(F)
    sa, sb, sc = s[t[:, 0]], s[t[:, 1]], s[t[:, 2]]
    cyc = ((sa > sb) & (sb > sc) & (sc > sa)) | ((sb > sa) & (sc > sb) & (sa > sc))
    return -float(cyc.mean())


# ---------------------------------------------------------------------------
# 4. The evolution loop: an agent optimising against its verifier, not truth
# ---------------------------------------------------------------------------

RADIUS = 3.5   # the feasible space: items stay inside the data's own ball


def _clip(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X * np.minimum(1.0, RADIUS / np.maximum(n, 1e-9))


def evolve(world, verifier, rng, K=6, T=250, step=0.35, diversity=0.0):
    """Hill-climb a K-item portfolio using ONLY the verifier's opinion.
    `diversity` adds a feature-space spread bonus -- the fair steel-man for a
    scalar verifier, since a real engineer would not ship six identical items."""
    X = _clip(rng.normal(size=(K, world.d)))
    n_view = len(world.P)
    sample_v = rng.choice(n_view, size=min(80, n_view), replace=False)

    oracle = verifier is None

    def verifier_value(Xp):
        if oracle:
            return world.coverage(Xp)          # the ceiling: optimise truth itself
        F = Xp @ world.A                       # noiseless features at decision time
        if verifier.personal:
            z = verifier.embed(F)
            u = -((z[None, :, :] - verifier.Q[sample_v][:, None, :]) ** 2).sum(-1)
            val = float(np.mean(np.max(u, axis=1)))   # it can aim at subgroups
        else:
            val = float(np.mean(verifier.score(F)))   # it cannot; best it can do
        if diversity:
            d = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
            val += diversity * float(d[np.triu_indices(K, 1)].mean())
        return val

    cur = verifier_value(X)
    traj = []
    for t in range(T):
        i = rng.integers(0, K)
        cand = X.copy()
        cand[i] = _clip((cand[i] + rng.normal(size=world.d) * step)[None])[0]
        val = verifier_value(cand)
        if val > cur:
            X, cur = cand, val
        if t % 25 == 0 or t == T - 1:
            traj.append((t, cur, world.coverage(X)))
    return X, traj


# ---------------------------------------------------------------------------
# 5. Experiments
# ---------------------------------------------------------------------------

def build_arms():
    return {
        "scalar":        (lambda: ScalarVerifier(), 0.0),
        "scalar+div":    (lambda: ScalarVerifier(), 0.6),
        "rubric(k=4)":   (lambda: RubricVerifier(n_feat_used=4), 0.0),
        "metric(k=6)":   (lambda: MetricVerifier(k=6), 0.0),
    }


def assert_learned(world, F, comps, held, floor=0.55):
    """Harness self-check. Numbers read off an untrained verifier are noise;
    this makes that failure loud instead of silent."""
    out = {}
    for mk in (lambda: ScalarVerifier(), lambda: MetricVerifier(k=6)):
        ver = mk().fit(F, comps)
        v, a, b, y = held
        if ver.personal:
            za, zb, q = ver.embed(F[a]), ver.embed(F[b]), ver.Q[v]
            z = np.sum((zb - q) ** 2, axis=1) - np.sum((za - q) ** 2, axis=1)
        else:
            z = ver.score(F[a]) - ver.score(F[b])
        out[ver.name] = float((((z > 0).astype(float)) == y).mean())
    bad = [k for k, a in out.items() if a < floor]
    return out, bad


def setup(world, rng, n_comp=24000, n_locate=40):
    F = world.features(world.X)
    comps = world.comparisons(n_comp)
    held = world.comparisons(8000)
    return F, comps, held


def run_arm(world, make_ver, diversity, F, comps, rng, T=250):
    ver = make_ver().fit(F, comps)
    X, traj = evolve(world, ver, rng, T=T, diversity=diversity)
    return ver, X, traj


def experiment_1(seed=0, truth_model="ideal", T=200):
    """Goodhart gap: what the verifier claims vs what the truth says."""
    rng = np.random.default_rng(seed)
    world = World(rng, truth_model=truth_model)
    F, comps, held = setup(world, rng)
    rows = []
    for name, (mk, div) in build_arms().items():
        ver, X, traj = run_arm(world, mk, div, F, comps, rng, T=T)
        claimed = [c for _, c, _ in traj]
        true = [t for _, _, t in traj]
        claim_gain = claimed[-1] - claimed[0]
        true_gain = true[-1] - true[0]
        rows.append((name, world.coverage(X), claim_gain, true_gain,
                     _spearman(np.array(claimed), np.array(true))))
    return rows


def experiment_2(seed=0, truth_model="ideal", n_variants=22, T=120):
    """THE central test: do ground-truth-free checks predict true harm?

    Build a family of verifiers of deliberately varying quality, score each with
    the checks (no truth) and with final coverage (truth), then correlate.
    """
    rng = np.random.default_rng(seed)
    world = World(rng, truth_model=truth_model)
    F, comps, held = setup(world, rng)
    n_view = len(world.P)

    # The family must contain BOTH wrong-shape (scalar/rubric) and
    # right-shape-but-starved (metric) verifiers. If it only contains metric
    # variants the checks are never asked the question that matters:
    # "can you detect a verifier of the WRONG SHAPE without ground truth?"
    variants = []
    for k in (2, 4, 8, 12, None):
        variants.append((f"scalar/f{k}", lambda k=k: ScalarVerifier(n_feat_used=k), 0.0))
    for k in (2, 4, 6):
        variants.append((f"rubric/f{k}", lambda k=k: RubricVerifier(n_feat_used=k), 0.0))
    for k in (1, 2, 3, 4, 6, 8):
        variants.append((f"metric/k{k}", lambda k=k: MetricVerifier(k=k), 0.0))
    for frac in (0.02, 0.06, 0.2, 0.6):
        variants.append((f"metric/data{frac}", lambda: MetricVerifier(k=6), 0.0, frac))
        variants.append((f"scalar/data{frac}", lambda: ScalarVerifier(), 0.0, frac))
    variants = variants[:n_variants]

    out = []
    for spec in variants:
        name, mk, div = spec[0], spec[1], spec[2]
        frac = spec[3] if len(spec) > 3 else 1.0
        n = int(len(comps[0]) * frac)
        sub = tuple(c[:n] for c in comps)
        ver = mk().fit(F, sub)
        checks = {
            "loglik": check_heldout_loglik(ver, F, held),
            "stability": check_subpop_stability(mk, F, sub, n_view),
            "homogeneity": check_residual_heterogeneity(ver, F, held),
            "no_cycles": check_cycle_rate(ver, F, rng),
        }
        _, X, _ = (None, *evolve(world, ver, rng, T=T, diversity=div))
        out.append((name, checks, world.coverage(X)))
    return out


def experiment_3(truth_model="ideal", T=120, seeds=(0, 1, 2, 3)):
    """Anchor economics: how much per-viewer data does locating actually need?

    Averaged over seeds -- a single evolve run is a stochastic hill climb whose
    run-to-run spread is larger than the effect being measured. Reported with
    the spread so an effect smaller than it is not read as a result."""
    per_n = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        world = World(rng, truth_model=truth_model)
        F, comps, held = setup(world, rng)
        base = MetricVerifier(k=6).fit(F, comps)
        Z_keep, B_keep = base.Z.copy(), base.B.copy()
        ceiling = evolve(world, None, rng, T=T)[1][-1][2]
        for n_per in (2, 5, 10, 20, 40, 80):
            ver = MetricVerifier(k=6)
            ver.Z, ver.B = Z_keep.copy(), B_keep.copy()
            ver.Q = np.zeros((len(world.P), ver.k))
            ver.locate_all(world.comparisons(n_per * len(world.P)), len(world.P))
            X, _ = evolve(world, ver, rng, T=T)
            # normalised: 1.0 = oracle ceiling, 0.0 = random floor
            floor = world.coverage(_clip(rng.normal(size=(6, world.d))))
            frac = (world.coverage(X) - floor) / max(ceiling - floor, 1e-9)
            per_n.setdefault(n_per, []).append(frac)
    return [(n, float(np.mean(v)), float(np.std(v))) for n, v in sorted(per_n.items())]


# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=3, suppress=True)
    for tm in TRUTH_MODELS:
        tag = "ASSUMPTION HOLDS" if tm == "ideal" else "MISSPECIFIED"
        print(f"\n{'='*74}\nWORLD = {tm}   ({tag})\n{'='*74}")

        rng0 = np.random.default_rng(0); w0 = World(rng0, truth_model=tm)
        F0, c0, h0 = setup(w0, rng0)
        acc, bad = assert_learned(w0, F0, c0, h0)
        print("  self-check  held-out accuracy: " +
              "  ".join(f"{k}={a:.3f}" for k, a in acc.items()) +
              ("   !! BELOW CHANCE FLOOR: " + ",".join(bad) if bad else "   (ok)"))
        _, _, tr = (None, *evolve(w0, None, rng0, T=200))
        ceiling = tr[-1][2]
        floor = w0.coverage(_clip(rng0.normal(size=(6, w0.d))))
        print(f"  reference   oracle ceiling {ceiling:.2f}   random floor {floor:.2f}")

        print("\n[E1] Goodhart gap -- verifier's claim vs hidden truth")
        print(f"{'arm':<14}{'true coverage':>15}{'claimed gain':>14}{'true gain':>12}{'claim~truth':>13}")
        for name, cov, cg, tg, sp in experiment_1(truth_model=tm):
            print(f"{name:<14}{cov:>15.3f}{cg:>14.3f}{tg:>12.3f}{sp:>13.2f}")

        print("\n[E2] Do ground-truth-FREE checks predict true coverage?")
        res = experiment_2(truth_model=tm)
        cov = np.array([c for _, _, c in res])
        for key in ("loglik", "stability", "homogeneity", "no_cycles"):
            vals = np.array([ch[key] for _, ch, _ in res])
            spread = vals.max() - vals.min()
            note = "  <-- VACUOUS (no variance)" if spread < 1e-9 else ""
            print(f"  {key:<14} spearman(check, true coverage) = {_spearman(vals, cov):>6.2f}{note}")
        best = max(res, key=lambda r: r[2]); worst = min(res, key=lambda r: r[2])
        print(f"  best by truth : {best[0]:<16} coverage {best[2]:.3f}")
        print(f"  worst by truth: {worst[0]:<16} coverage {worst[2]:.3f}")

        print("\n[E3] Anchor economics -- fraction of the oracle-vs-random range recovered")
        print("     (mean +- sd over 4 seeds; an effect smaller than the sd is NOT a result)")
        for n_per, m, sd in experiment_3(truth_model=tm):
            bar = "#" * max(0, int(round(m * 30)))
            print(f"  {n_per:>3} comparisons/viewer   {m:>6.2f} +- {sd:.2f}  {bar}")


if __name__ == "__main__":
    main()
