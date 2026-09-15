"""E-D: when the ruler moves, does the metric SEE it? (pre-registered)

This tests the line's CORE claim -- the instrument claim -- which E-C never
touched: inject KNOWN drift into the observer and KNOWN capability change
into the artifacts, then ask whether the metric readout (JND radii) tells
the conditions apart where the scalar readout (mean fidelity) cannot.

Every readout is taken THROUGH the current, possibly-drifted observer --
a deployed instrument has nothing else. Simulation ground truth (which
condition was injected) is used only to score the identification afterward.

Conditions, applied per seed to the same base population:
    C0  null            nothing changes
    C1  hacked judge    observer ignores probes w.p. p and guesses the
                        intent distribution's mode ("always-pass" mechanism);
                        p is calibrated in the precheck so the score gain
                        matches C3 -- a hack disguised as normal progress
    C2  coarsened ruler observer's change-detection threshold doubled
    C3  true gain       artifacts hill-climbed at fixed budget, observer honest
    C4  both at once    C3 artifacts x C1 observer (the moving-ruler proper)

Readouts (fixed before running):
    scalar  mean fidelity of the population through the current observer
    metric  median log JND radius over population x random directions
            (bisection as in E-A; hi bound 1.2, censoring at the bound IS
            the "cannot distinguish anything" signal)

Pre-registered criteria (plan.html section 9):
    P1  dM(C1)-dM(C3) > 2se  AND  |ds(C1)-ds(C3)| < 2se   -> instrument claim holds
    fail: dM cannot separate C1 from C3                    -> claim dead, line void
    inconclusive: scalar separates C1/C3 on fresh seeds    -> toy uninformative
    S1  dM(C4)-dM(C3) > 2se   S2  dM(C2)-dM(C0) > 2se

Run:  ../../.venv/bin/python -u drift.py precheck   (tuning allowed here)
      ../../.venv/bin/python -u drift.py            (frozen cfg, fresh seeds)
"""

import sys
import numpy as np
from intent import (K_SEG, random_impl, perturb, probe, recover, structure_match)

CFG = dict(B=5, K=8, p_mode=0.5, p_lazy=0.7, thresh=6.0, n_dirs=6, rounds=25)
CHECK_SEEDS = range(4)          # precheck only; never reused
REG_SEEDS = range(200, 208)     # registered run

MODE_INTENT = tuple(range(K_SEG))


def skewed_intent(rng, p_mode):
    """Real task distributions carry priors a judge can exploit; a uniform
    intent world is an unrealistically hack-proof one."""
    if rng.random() < p_mode:
        return MODE_INTENT
    while True:
        c = tuple(rng.permutation(K_SEG).tolist())
        if c != MODE_INTENT:
            return c


# --- observers -------------------------------------------------------------

def honest(x, y):
    return recover(x, y)


def make_lazy(p, rng):
    def obs(x, y):
        if rng.random() < p:
            return MODE_INTENT
        return recover(x, y)
    return obs


def make_coarse(thresh):
    def obs(x, y):
        return recover(x, y, thresh=thresh)
    return obs


# --- readouts through an observer ------------------------------------------

def fid(impl, budget, rng, obs, reps=8):
    return float(np.mean([structure_match(impl["intent"], obs(*probe(impl, budget, rng)))
                          for _ in range(reps)]))


def pop_scalar(pop, budget, rng, obs):
    return float(np.mean([fid(s, budget, rng, obs) for s in pop]))


def discr(a, b, budget, rng, obs, reps=6):
    return np.mean([int(obs(*probe(a, budget, rng)) != obs(*probe(b, budget, rng)))
                    for _ in range(reps)])


def jnd(impl, direction, budget, rng, obs, lo=0.005, hi=1.2, steps=7, target=0.5, reps=6):
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        d = discr(impl, perturb(impl, rng, scale=mid, direction=direction)[0], budget, rng, obs,
                  reps=reps)
        if d < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def pop_metric(pop, budget, rng, obs, n_dirs, reps=6, agg="median"):
    """agg='median' is E-D's registered readout; E-D2 registers 'mean' because
    the hi-censored tail IS the signal and the median discards it."""
    radii = []
    for impl in pop:
        for _ in range(n_dirs):
            d = rng.normal(size=2 * K_SEG - 1)
            radii.append(jnd(impl, d, budget, rng, obs, reps=reps))
    logs = np.log(radii)
    return float(np.median(logs) if agg == "median" else np.mean(logs))


# --- capability injection ---------------------------------------------------

def improve(pop, budget, rng, rounds, tries=3):
    out = []
    for impl in pop:
        cur, cur_val = impl, fid(impl, budget, rng, honest, reps=6)
        for _ in range(rounds):
            for _ in range(tries):
                cand, _ = perturb(cur, rng, scale=float(rng.uniform(0.02, 0.14)))
                v = fid(cand, budget, rng, honest, reps=6)
                if v > cur_val:
                    cur, cur_val = cand, v
        out.append(cur)
    return out


# --- one seed, all conditions -----------------------------------------------

CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


def run_seed(seed, cfg):
    rng = np.random.default_rng([seed, 0])
    pop0 = [random_impl(rng, skewed_intent(rng, cfg["p_mode"])) for _ in range(cfg["K"])]
    brng = np.random.default_rng([seed, 1])
    s0 = pop_scalar(pop0, cfg["B"], brng, honest)
    m0 = pop_metric(pop0, cfg["B"], brng, honest, cfg["n_dirs"])
    pop_imp = improve(pop0, cfg["B"], np.random.default_rng([seed, 2]), cfg["rounds"])
    out = {"_s0": s0, "_m0": m0, "_pop0": pop0, "_pop_imp": pop_imp}
    for ci, cname in enumerate(CONDITIONS):
        crng = np.random.default_rng([seed, 10 + ci])
        if cname == "C0":
            pop, obs = pop0, honest
        elif cname == "C1":
            pop, obs = pop0, make_lazy(cfg["p_lazy"], crng)
        elif cname == "C2":
            pop, obs = pop0, make_coarse(cfg["thresh"])
        elif cname == "C3":
            pop, obs = pop_imp, honest
        else:
            pop, obs = pop_imp, make_lazy(cfg["p_lazy"], crng)
        # Progress monitor reads the production population; the ruler monitor
        # reads a FROZEN reference battery (pop0) -- metrology measures the
        # standard, not the product, else true robustness gains inflate JND
        # radii and drift re-conflates with capability (seen in precheck C3).
        # The reference needs no quality labels, so this stays anchor-free.
        s1 = pop_scalar(pop, cfg["B"], crng, obs)
        m1 = pop_metric(pop0, cfg["B"], crng, obs, cfg["n_dirs"])
        out[cname] = (s1 - s0, m1 - m0)
    return out


# --- E-D2: adversarially calibrated hack (final redesign, plan.html sec. 9) --

CFG2 = dict(B=5, K=8, p_mode=0.6, thresh=6.0, n_dirs=8, reps=8, rounds=8, calib_steps=9)
REG2_SEEDS = range(300, 312)


def calibrate_p(pop0, s0, target, budget, seed, steps):
    """The realistic threat model: the hacker tunes its strength so the score
    gain reproduces the progress actually observed on this seed. Calibration
    draws never overlap the readout draws."""
    lo, hi = 0.0, 1.0
    for i in range(steps):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng([seed, 40 + i])
        ds = pop_scalar(pop0, budget, rng, make_lazy(mid, rng)) - s0
        if ds < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_seed2(seed, cfg):
    rng = np.random.default_rng([seed, 0])
    pop0 = [random_impl(rng, skewed_intent(rng, cfg["p_mode"])) for _ in range(cfg["K"])]
    brng = np.random.default_rng([seed, 1])
    s0 = pop_scalar(pop0, cfg["B"], brng, honest)
    m0 = pop_metric(pop0, cfg["B"], brng, honest, cfg["n_dirs"], cfg["reps"], "mean")
    pop_imp = improve(pop0, cfg["B"], np.random.default_rng([seed, 2]), cfg["rounds"])
    ds3_cal = pop_scalar(pop_imp, cfg["B"], np.random.default_rng([seed, 3]), honest) - s0
    crng = np.random.default_rng([seed, 4])
    ds_ceiling = pop_scalar(pop0, cfg["B"], crng, make_lazy(1.0, crng)) - s0
    p = calibrate_p(pop0, s0, ds3_cal, cfg["B"], seed, cfg["calib_steps"])
    out = {"_s0": s0, "_m0": m0, "_pop0": pop0, "_pop_imp": pop_imp,
           "_p": p, "_reach": ds3_cal <= ds_ceiling}
    for ci, cname in enumerate(CONDITIONS):
        crng = np.random.default_rng([seed, 10 + ci])
        if cname == "C0":
            pop, obs = pop0, honest
        elif cname == "C1":
            pop, obs = pop0, make_lazy(p, crng)
        elif cname == "C2":
            pop, obs = pop0, make_coarse(cfg["thresh"])
        elif cname == "C3":
            pop, obs = pop_imp, honest
        else:
            pop, obs = pop_imp, make_lazy(p, crng)
        s1 = pop_scalar(pop, cfg["B"], crng, obs)
        m1 = pop_metric(pop0, cfg["B"], crng, obs, cfg["n_dirs"], cfg["reps"], "mean")
        out[cname] = (s1 - s0, m1 - m0)
    return out


def paired(results, ca, cb, idx):
    d = np.array([r[ca][idx] - r[cb][idx] for r in results])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def report(results, cfg, label):
    print(f"\n[{label}] seeds={len(results)}  cfg={cfg}")
    print(f"   {'cond':<6}{'d scalar':>12}{'d metric (log JND)':>22}")
    for c in CONDITIONS:
        ds = np.array([r[c][0] for r in results])
        dm = np.array([r[c][1] for r in results])
        print(f"   {c:<6}{ds.mean():>+9.3f} ±{ds.std():.3f}{dm.mean():>+15.3f} ±{dm.std():.3f}")
    for lbl, ca, cb, idx in (("P1a  dM(C1)-dM(C3)", "C1", "C3", 1),
                             ("P1b  ds(C1)-ds(C3)", "C1", "C3", 0),
                             ("S1   dM(C4)-dM(C3)", "C4", "C3", 1),
                             ("S2   dM(C2)-dM(C0)", "C2", "C0", 1)):
        d, se = paired(results, ca, cb, idx)
        sig = "significant" if abs(d) > 2 * se else "NOT significant"
        print(f"   {lbl} = {d:+.3f} (2se {2*se:.3f})  -> {sig}")


def precheck(results, cfg):
    print("\n[precheck] instrument-aliveness gates (tuning allowed, then freeze):")
    s0 = np.array([r["_s0"] for r in results])
    ok1 = 0.35 <= s0.mean() <= 0.80
    print(f"   PC1 honest scalar has dynamic range: mean s0 = {s0.mean():.3f}  -> {'OK' if ok1 else 'FAIL'}")
    d, se = paired(results, "C1", "C3", 0)
    ok2a = np.mean([r['C1'][0] for r in results]) > 0
    ok2b = abs(d) < 2 * se
    print(f"   PC2 hack inflates score: mean ds(C1) = {np.mean([r['C1'][0] for r in results]):+.3f} -> {'OK' if ok2a else 'FAIL'}")
    print(f"       hack magnitude matches C3: ds(C1)-ds(C3) = {d:+.3f} (2se {2*se:.3f}) -> {'OK' if ok2b else 'TUNE'}")
    gains = []
    for r in results:
        vrng = np.random.default_rng([hash(str(r['_s0'])) % 2**31, 99])
        f0 = np.mean([fid(s, cfg["B"], vrng, honest, reps=10) for s in r["_pop0"]])
        f1 = np.mean([fid(s, cfg["B"], vrng, honest, reps=10) for s in r["_pop_imp"]])
        gains.append(f1 - f0)
    g = np.array(gains)
    ok3 = g.mean() > 2 * g.std(ddof=1) / np.sqrt(len(g))
    print(f"   PC3 capability gain is real (held-out honest fid): +{g.mean():.3f} ± {g.std():.3f} -> {'OK' if ok3 else 'FAIL'}")
    m0 = np.array([r["_m0"] for r in results])
    ok4 = m0.mean() < np.log(0.6)
    print(f"   PC4 baseline JND measurable (median log < log 0.6 = {np.log(0.6):.2f}): {m0.mean():.3f} -> {'OK' if ok4 else 'FAIL'}")
    return ok1 and ok2a and ok2b and ok3 and ok4


def report2(results, cfg, label):
    report(results, cfg, label)
    ps = np.array([r["_p"] for r in results])
    print(f"   calibrated p per seed: " + " ".join(f"{p:.2f}" for p in ps))
    d, se = paired(results, "C1", "C3", 0)
    mc = abs(d) < 2 * se
    print(f"   MC  |ds(C1)-ds(C3)| = {d:+.3f} (2se {2*se:.3f})  -> {'holds' if mc else 'FAILS'}")
    return mc


def precheck2(results, cfg):
    print("\n[precheck E-D2] gates:")
    s0 = np.array([r["_s0"] for r in results])
    ok1 = 0.35 <= s0.mean() <= 0.80
    print(f"   PC1 dynamic range: mean s0 = {s0.mean():.3f} -> {'OK' if ok1 else 'FAIL'}")
    gains = []
    for r in results:
        vrng = np.random.default_rng([hash(str(r['_s0'])) % 2**31, 99])
        f0 = np.mean([fid(s, cfg["B"], vrng, honest, reps=10) for s in r["_pop0"]])
        f1 = np.mean([fid(s, cfg["B"], vrng, honest, reps=10) for s in r["_pop_imp"]])
        gains.append(f1 - f0)
    g = np.array(gains)
    ok3 = g.mean() > 2 * g.std(ddof=1) / np.sqrt(len(g))
    print(f"   PC3 capability real: +{g.mean():.3f} ± {g.std():.3f} -> {'OK' if ok3 else 'FAIL'}")
    m0 = np.array([r["_m0"] for r in results])
    ok4 = m0.mean() < np.log(0.6)
    print(f"   PC4 baseline measurable: mean log JND = {m0.mean():.3f} -> {'OK' if ok4 else 'FAIL'}")
    ok6 = all(r["_reach"] for r in results)
    print(f"   PC6 hack ceiling reaches target on every seed -> {'OK' if ok6 else 'FAIL'}")
    return ok1 and ok3 and ok4 and ok6


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "registered"
    print("=" * 78)
    print("E-D: DOES THE METRIC SEE THE RULER MOVE?")
    print("=" * 78)
    if mode == "ed2-precheck":
        results = [run_seed2(s, CFG2) for s in CHECK_SEEDS]
        mc = report2(results, CFG2, "E-D2 precheck seeds (not for verdicts)")
        ok = precheck2(results, CFG2)
        print(f"\n   E-D2 precheck {'PASSED' if ok and mc else 'NOT passed - tune and rerun'}")
        return
    if mode == "ed2":
        results = [run_seed2(s, CFG2) for s in REG2_SEEDS]
        mc = report2(results, CFG2, "E-D2 REGISTERED RUN, fresh seeds")
        dP, seP = paired(results, "C1", "C3", 1)
        print("\n" + "-" * 78)
        if not mc:
            print("VERDICT: INCONCLUSIVE (final) -- calibration failed on fresh seeds;")
            print("reported as-is, no further redesign (termination clause).")
        elif dP > 2 * seP:
            print("VERDICT: P1' HOLDS -- against a hack calibrated to be scalar-")
            print("indistinguishable from true progress, the metric still sees the")
            print("ruler move. The instrument claim stands.")
        else:
            print("VERDICT: FAILURE (final) -- MC held but the metric could not")
            print("separate the calibrated hack from true gain. The observability")
            print("claim is dead in this construction; no further redesign.")
        return
    if mode == "precheck":
        results = [run_seed(s, CFG) for s in CHECK_SEEDS]
        report(results, CFG, "precheck seeds (not for verdicts)")
        ok = precheck(results, CFG)
        print(f"\n   precheck {'PASSED - freeze cfg and run registered' if ok else 'NOT passed - tune and rerun'}")
        return
    results = [run_seed(s, CFG) for s in REG_SEEDS]
    report(results, CFG, "REGISTERED RUN, fresh seeds")
    dP1a, seP1a = paired(results, "C1", "C3", 1)
    dP1b, seP1b = paired(results, "C1", "C3", 0)
    print("\n" + "-" * 78)
    if dP1a > 2 * seP1a and abs(dP1b) < 2 * seP1b:
        print("VERDICT: P1 holds -- the metric separates the hacked judge from true")
        print("gain while the scalar cannot. The instrument claim stands.")
    elif dP1a <= 2 * seP1a:
        print("VERDICT: FAILURE branch -- the metric cannot separate C1 from C3.")
        print("The observability claim is dead; the line is void per the requirement.")
    else:
        print("VERDICT: INCONCLUSIVE branch -- the scalar separated C1 from C3 on")
        print("fresh seeds (calibration did not transfer); toy uninformative, redesign.")


if __name__ == "__main__":
    main()
