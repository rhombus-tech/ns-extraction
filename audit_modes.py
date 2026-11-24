"""Audit of the 'slowest atmosphere mode' claim: search the spectrum WITHOUT my localisation / |Im| filters and report where each
slow mode sits in psi, how localised it is, and its imaginary part. EXPLORATORY.
"""
import sys
import numpy as np
from scipy.sparse.linalg import eigs
import io_fields as io
import run_ring_case as rc
import scalar_experiment as se
import true_rates as tr
import true_operator as to
import orbits as ob
import rates

KAPPA = 2.5e-3
REFINE = 1.5


def main():
    ref = np.load("stage1_frozen.npz")
    speed, zc5 = float(ref["speed"]), float(ref["zc"])
    u5, _, _, _, h = rc.load(5)
    a5, _ = se.extraction(u5, h, zc5)
    psi, _ = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    pf, lab, rf, zf, R, Z, drf, dzf, region = tr._setup(psi_co, a5["omega_phi"], a5["rho"], a5["z"], a5["dr"], a5["dz"], REFINE)
    _, _, _, T, _ = rates.orbit_profiles(lab, R, drf, dzf, region, 1)
    w = 2 * np.pi / T
    print(f"kappa={KAPPA}, grid {pf.shape}, orbit frequency range [{w.min():.2f},{w.max():.2f}]", flush=True)
    L, _ = to.full_window_operator(pf, rf, drf, dzf, KAPPA)
    inside = region.ravel()
    labflat = lab.ravel()
    found = {}
    for s in np.linspace(0.0, 3.2, 14):
        for g in (0.03, 0.15):
            try:
                ev, vec = eigs(L, k=10, sigma=complex(-g, s), which="LM")
            except Exception as e:
                continue
            for lam, v in zip(ev, vec.T):
                key = (round(-lam.real, 4), round(abs(lam.imag), 3))
                p = np.abs(v) ** 2
                p /= p.sum()
                loc = float(p[inside].sum())
                psi_c = float((p[inside] * labflat[inside]).sum() / max(p[inside].sum(), 1e-30))
                found[key] = (loc, psi_c)
    print("\nslowest modes found (decay rate, |Im|, localisation in closed region, mean psi of the in-region part):")
    print("   psi runs from %.3f (core) to %.3f (edge of closed region)" % (lab[region].min(), lab[region].max()))
    for (rate, im), (loc, pc) in sorted(found.items())[:18]:
        print(f"  rate {rate:.4f}  |Im| {im:6.3f}  localisation {loc:.3f}  psi {pc:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
