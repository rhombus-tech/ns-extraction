"""Mechanism test for the evolving-vs-frozen ratio q.  EXPLORATORY; criterion fixed BEFORE ring 2 was computed.

Hypothesis (from the ring-1 ablations in unsteady_ablation.out): the windowed decay rate of the orbit-varying scalar tracks the instantaneous closed-form core rate of the flow
(sqrt(|omega'| (h1+h2)), kappa-independent), so
        q(kappa) = rate_evolving / rate_frozen(age 2.5)  ~  core(tau_mid) / core(2.5),
with tau_mid the midpoint of the evolving measurement window for that kappa (core evaluated on the evolving flow's snapshots, interpolated linearly in time).
Criterion: |q_pred / q_meas - 1| <= 0.25 for each of the four cases (ring 1 in-sample, ring 2 out-of-sample); the outcome is reported either way.
Measured q and windows are those already recorded: ring 1 (stage1_results.json, 0.674 / 0.725), ring 2 (unsteady2_results_128.json).
"""
import json
import sys
import numpy as np
import analyze_stage1 as a
import orbits as ob
import time_dep_2d as td


def core_series(d):
    spd = td.speeds_of(d)
    out = []
    for i in range(len(d["times"])):
        lab, R, dr, dz = a.full_labels(d, i, spd)
        cm = ob.core_mode(lab, R, dr, dz, a.region_of(lab), 1e-3)
        out.append(cm["core_shear"] * sum(cm["hessian_eigenvalues"]))     # |omega'| (h1+h2); rate = sqrt(2 kappa * this)
    return np.array(out), d["times"]


def main():
    cases = []
    r1 = json.load(open("stage1_results.json"))
    for j, kap in enumerate((0.03, 0.01)):
        f, s = r1["F"][j], r1["S"][j]
        cases.append(("ring 1 (in-sample)", "stage1_full.npz", kap, s["rate"] / f["rate"], s["window"]))
    r2 = json.load(open("unsteady2_results_128.json"))
    for j, kap in enumerate((0.03, 0.01)):
        f, s = r2["frozen"][j], r2["evolving_128"][j]
        cases.append(("ring 2 (OUT-OF-SAMPLE)", "unsteady2_full_128.npz", kap, s["rate"] / f["rate"], s["window"]))
    cache = {}
    ok_all = True
    print("case | kappa | measured q | window | tau_mid | core(2.5) | core(tau_mid) | predicted q = sqrt(core ratio) | pred/meas")
    for name, fn, kap, q, win in cases:
        if fn not in cache:
            cache[fn] = core_series(np.load(fn))
        c, t = cache[fn]
        tm = 0.5 * (win[0] + win[1])
        c25, cm = np.interp(2.5, t, c), np.interp(tm, t, c)
        qp = float(np.sqrt(cm / c25))                    # rate ~ sqrt(|omega'| trace)
        ok = abs(qp / q - 1) <= 0.25
        ok_all &= ok
        print(f"{name:24s} | {kap:5.2f} | {q:.3f} | {tuple(win)} | {tm:5.2f} | {c25:6.2f} | {cm:6.2f} | {qp:.3f} | {qp / q:.3f} {'ok' if ok else 'OUTSIDE'}")
    print(f"\nall four within +-25%: {ok_all}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
