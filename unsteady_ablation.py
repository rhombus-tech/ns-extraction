"""Ablations with the validated 2D time-dependent propagator (time_dep_2d.py) on ring 1 (stage1_full.npz).

EXPLORATORY (not preregistered); convention fixed before running:
  A. FROZEN-AT-AGE: for tau in (0, 1, 2.5, 4, 6, 8) freeze the flow at the snapshot of age tau, place the standard blob, and measure the windowed rate (analyze_stage1 protocol).
     (tau = 2.5 must reproduce the 3D frozen result 0.2777 / 0.3878 as a check of the frozen mode.)
  B. EVOLVING-FROM-AGE: start the blob at snapshot tau0 and evolve with the true flow from then on (rate measured on time since tau0).
  C. Report the instantaneous core quantities (closed-form core rate, shear) at each snapshot.
Question answered: is the evolving-vs-frozen difference explained by the flow weakening with age (A, C), and how much by the early transient (B)?
"""
import sys
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import expm_multiply
import analyze_stage1 as a
import orbits as ob
import time_dep_2d as td
import true_operator as to

AGES = (0.0, 1.0, 2.5, 4.0, 6.0, 8.0)       # snapshot ages (output indices 0, 2, 5, 8, 12, 16)


def frozen_rate(d, spd, kappa, i_snap):
    times, fields, psi = td.frozen_series(d, kappa, i_snap, t_end=24.0, spd=spd)
    lab, R, dr, dz = a.full_labels(d, i_snap, spd)
    region = a.region_of(lab)
    norms = np.array([a.dc_norm(c, lab, R, dr, dz, region) for c in fields])
    rate, win = a.measured_rate(times, norms)
    return rate, win, norms / norms[0]


def evolving_from(d, spd, kappa, i0):
    """Blob placed at snapshot i0, then the true evolving flow; returns windowed rate on elapsed time and the window."""
    rho, dr, dz = d["rho"], float(d["dr"]), float(d["dz"])
    c = td.blob_initial(d)
    out = [c.copy()]
    n = len(d["times"])
    for i in range(i0, n - 1):
        U = spd[i + 1]
        psi = 0.5 * (td.psi_comoving(d, i, U) + td.psi_comoving(d, i + 1, U))
        L, _ = to.full_window_operator(psi, rho, dr, dz, kappa, p=1)
        c = expm_multiply(L * td.DT_OUT, c.ravel()).reshape(c.shape)
        s = (d["zc"][i + 1] - d["zc"][i]) - U * td.DT_OUT
        c = ndimage.shift(c, (0.0, -s / dz), order=3, mode="constant", cval=0.0)
        out.append(c.copy())
    norms = []
    for k, c in enumerate(out):
        lab, R, dr_, dz_ = a.full_labels(d, i0 + k, spd)
        norms.append(a.dc_norm(c, lab, R, dr_, dz_, a.region_of(lab)))
    norms = np.array(norms)
    t = np.arange(len(norms)) * td.DT_OUT
    rate, win = a.measured_rate(t, norms)
    return rate, win


def core_info(d, spd, i):
    lab, R, dr, dz = a.full_labels(d, i, spd)
    region = a.region_of(lab)
    return ob.core_mode(lab, R, dr, dz, region, 1e-3)


def main():
    d = np.load("stage1_full.npz")
    spd = td.speeds_of(d)
    print("C. instantaneous core quantities at kappa = 1e-3 (closed-form core rate, core shear |omega'|, core frequency, Hessian trace):")
    for age in AGES:
        i = int(round(age / td.DT_OUT))
        cm = core_info(d, spd, i)
        print(f"   age {age:4.1f}: rate {cm['rate']:.3f}  |omega'| {cm['core_shear']:.2f}  omega_c {cm['core_frequency']:.2f}  trace {sum(cm['hessian_eigenvalues']):.2f}", flush=True)
    for j in (0, 1):
        kap = float(d["kappas"][j])
        print(f"\n=== kappa = {kap:g}   (3D frozen at age 2.5: {[0.2777, 0.3878][j]};  3D evolving: {[0.1871, 0.2811][j]}) ===", flush=True)
        print("A. frozen-at-age windowed rate:")
        for age in AGES:
            i = int(round(age / td.DT_OUT))
            rate, win, r = frozen_rate(d, spd, kap, i)
            print(f"   age {age:4.1f}: rate {rate if rate is None else round(rate, 4)} window {win}", flush=True)
        print("B. evolving-from-age windowed rate (time measured from the start of the blob):")
        for age in (0.0, 2.5, 5.0):
            i0 = int(round(age / td.DT_OUT))
            rate, win = evolving_from(d, spd, kap, i0)
            print(f"   start age {age:4.1f}: rate {rate if rate is None else round(rate, 4)} window {win}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
