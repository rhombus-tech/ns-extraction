"""EXPLORATORY (in-sample, NOT preregistered): is the decay of the orbit-varying scalar in an EVOLVING ring explained by the instantaneous core-mode rate?

For the ring-1 run (stage1_full.npz, nu = 0.01) and its frozen counterpart (stage1_frozen.npz):
  s(t_i)   = local decay rate of ||delta c|| = -(least-squares slope of ln norm over 7 consecutive outputs, 3 time units)
  lam(t_i) = closed-form core-mode rate (orbits.core_mode) of the flow at that instant, at the same kappa
Reports s/lam over time, and the adiabatic prediction  r_ad(t) = exp(-int lam dt)  against the measured r(t) = norm(t)/norm(0) in a late window.
Nothing here is a test: it decides whether an adiabatic hypothesis is worth preregistering for a new flow.
"""
import sys
import numpy as np
import io_fields as io
import orbits as ob
import analyze_stage1 as a


def labels_full(d):
    n = len(d["times"])
    axes = []
    dz = float(d["dz"])
    for i in range(n):
        z = d["zc"][i] - a.Z_HALF + (np.arange(d["u_z"].shape[2]) + 0.5) * dz
        axes.append({"omega_phi": d["omega_phi"][i], "z": z})
    speeds = [io.ring_speed(axes[i - 1], d["times"][i - 1], axes[i], d["times"][i]) for i in range(1, n)]
    speeds = [speeds[0]] + speeds
    return [a.full_labels(d, i, speeds) for i in range(n)]


def series(d, labels, j, kappa):
    times = d["times"]
    norms, lam, wc = [], [], []
    for i in range(len(times)):
        lab, R, dr, dz = labels[i]
        region = a.region_of(lab)
        norms.append(a.dc_norm(d["slices"][i, j], lab, R, dr, dz, region))
        cm = ob.core_mode(lab, R, dr, dz, region, kappa)
        lam.append(cm["rate"] if cm else np.nan)
        wc.append(cm["core_frequency"] if cm else np.nan)
    return times, np.array(norms), np.array(lam), np.array(wc)


def local_rate(t, norms, i, half=3):
    lo, hi = max(0, i - half), min(len(t), i + half + 1)
    if hi - lo < 5 or np.any(norms[lo:hi] <= 0):
        return np.nan
    return -np.polyfit(t[lo:hi], np.log(norms[lo:hi]), 1)[0]


def report(name, d, labels):
    print(f"\n=== {name} ===", flush=True)
    for j, kap in enumerate(d["kappas"][:2]):  # the two kappa that are resolved (tail < 1e-6)
        t, norms, lam, wc = series(d, labels, j, float(kap))
        r = norms / norms[0]
        print(f"kappa = {kap:g}: t | r | local rate s | closed-form lam | s/lam | core freq", flush=True)
        for i in range(0, len(t), 2):
            s = local_rate(t, norms, i)
            print(f"   {t[i]:5.1f} | {r[i]:.3f} | {s:.3f} | {lam[i]:.3f} | {s / lam[i]:.2f} | {wc[i]:.2f}", flush=True)
        ok = np.isfinite(lam)
        ad = np.exp(-np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(t))]))
        late = np.where((r < 0.3) & ok)[0]
        if len(late):
            i0 = late[0]
            print(f"   adiabatic prediction of r(t)/r(t0) from t0 = {t[i0]:.1f}: at t = {t[-1]:.1f} predicted {ad[-1] / ad[i0]:.3f}, measured {r[-1] / r[i0]:.3f}", flush=True)


def main():
    d = np.load("stage1_full.npz")
    report("EVOLVING ring (nu = 0.01)", d, labels_full(d))
    df = np.load("stage1_frozen.npz")
    lab = a.frozen_labels(df)
    report("FROZEN snapshot", df, [lab] * len(df["times"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
