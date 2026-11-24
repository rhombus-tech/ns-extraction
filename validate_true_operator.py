"""Mechanical application of PREREGISTERED_TRUE_OPERATOR.md (V1: 2D solver vs 3D pseudo-spectral evolution; V2: eigenvalue convergence)."""
import json
import sys
import numpy as np
from scipy.sparse.linalg import expm_multiply
import analyze_stage1 as a
import orbits as ob
import rates
import true_operator as to
import compare_real_flow as cr

KAPPA_V1 = (1e-2, 3.3e-3)
KAPPA_V2 = 3.3e-3
T_END, N_OUT = 12.0, 25
Z_HALF = 2.0


def blob2d(R, Z, zc):
    return np.exp(-(((R - 1.0) ** 2 + (Z - zc - 0.25) ** 2) / (2 * 0.2**2)))


def delta_norms(frames, lab, R, dr, dz, region):
    out = []
    for c in frames:
        _, _, dc = ob.decompose(lab, R, dr, dz, c.reshape(lab.shape), region, 160)
        out.append(ob.weighted_l2(dc, R, dr, dz, region))
    return np.array(out)


def v1():
    psi_co, sign, rho, z, dr, dz = cr.real_flow()
    pf, rf, zf, drf, dzf = cr.refine(psi_co, rho, z, dr, dz, 2)
    lab = sign * pf
    R, Zg = np.meshgrid(rf, zf, indexing="ij")
    region = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    zc5 = float(np.load("stage1_frozen.npz")["zc"])
    c0 = blob2d(R, Zg, zc5).ravel()
    d3 = np.load("stage1_e1_frozen.npz")
    lab3, R3, dr3, dz3 = a.frozen_labels(d3)
    reg3 = a.region_of(lab3)
    res = {}
    for kap in KAPPA_V1:
        L, _ = to.full_window_operator(pf, rf, drf, dzf, kap)
        frames = expm_multiply(L, c0, start=0.0, stop=T_END, num=N_OUT, endpoint=True)
        n2 = delta_norms(frames, lab, R, drf, dzf, region)
        j = int(np.argmin(np.abs(d3["kappas"] - kap)))
        times = d3["times"]
        n3 = np.array([a.dc_norm(d3["slices"][i, j], lab3, R3, dr3, dz3, reg3) for i in range(N_OUT)])
        r2, r3 = n2 / n2[0], n3 / n3[0]
        assert np.allclose(times[:N_OUT], np.linspace(0, T_END, N_OUT))
        sel = r3 >= 0.05
        dev = np.abs(r2[sel] - r3[sel]) / r3[sel]
        res[kap] = {"max_dev": float(dev.max()), "n_points": int(sel.sum()), "r2": r2.tolist(), "r3": r3.tolist()}
        print(f"V1 kappa={kap:.1e}: {int(sel.sum())} comparison times, max relative deviation {dev.max():.3f}", flush=True)
        for t, x2, x3 in list(zip(times, r2, r3))[:N_OUT:3]:
            print(f"     t={t:4.1f} r_2D={x2:.3f} r_3D={x3:.3f}", flush=True)
    passed = all(v["max_dev"] <= 0.20 for v in res.values())
    return passed, res


def v2():
    psi_co, sign, rho, z, dr, dz = cr.real_flow()
    out = {}
    for k in (1.5, 2):
        pf, rf, zf, drf, dzf = cr.refine(psi_co, rho, z, dr, dz, k)
        lab = sign * pf
        R, _ = np.meshgrid(rf, zf, indexing="ij")
        region = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
        _, _, _, T, _ = rates.orbit_profiles(lab, R, drf, dzf, region, 1)
        w = 2 * np.pi / T
        L, _ = to.full_window_operator(pf, rf, drf, dzf, KAPPA_V2)
        rate, ev, frac = to.atmosphere_modes(L, region, (w.min(), w.max()), im_min=0.3 * w.min(), n_shifts=6)
        out[k] = rate
        print(f"V2 refinement {k}: grid {pf.shape}, slowest atmosphere rate {rate:.4f} (eig {ev}, localisation {frac})", flush=True)
    both = all(np.isfinite(v) for v in out.values())
    rel = abs(out[1.5] - out[2]) / out[2] if both else float("nan")
    return bool(both and rel <= 0.10), {"rates": out, "relative_difference": float(rel)}


def main():
    p1, r1 = v1()
    p2, r2 = v2()
    print("\n=== PREREGISTERED CRITERIA ===")
    print(f"V1 (2D solver vs 3D evolution): {'PASS' if p1 else 'FAIL'}  max deviations " +
          ", ".join(f"{k:.1e}: {v['max_dev']:.3f}" for k, v in r1.items()))
    print(f"V2 (eigenvalue converged):       {'PASS' if p2 else 'FAIL'}  {r2}")
    rule = ("1: ship as experimental rate tool with these validation numbers" if p1 and p2 else
            "2: NOT validated; do not ship as a rate tool" if not p1 else
            "3: ship only with an explicit resolution warning")
    print(f"Pre-declared interpretation {rule}")
    json.dump({"V1": r1, "V2": r2, "p1": p1, "p2": p2, "rule": rule}, open("true_operator_validation.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
