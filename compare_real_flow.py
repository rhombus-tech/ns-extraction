"""True slowest orbit-varying eigenvalue of the full 2D operator on the REAL frozen ring flow vs the reduced estimator.

Frozen flow = t = 2.5 ring-frame velocity (as in stage 1). The true operator is solved on the whole (rho,z) window (real exterior
flow, absorbing edges); atmosphere modes are those with >= 90% of |v|^2 inside the closed-streamline region.
EXPLORATORY: run after the preregistered tests; its purpose is to find out whether the estimator or the test was at fault.
"""
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
import io_fields as io
import orbits as ob
import rates
import run_ring_case as rc
import scalar_experiment as se
import true_operator as to

REFINE = 2


def real_flow():
    ref = np.load("stage1_frozen.npz")
    speed, zc5 = float(ref["speed"]), float(ref["zc"])
    u5, _, t5, _, h = rc.load(5)
    a5, _ = se.extraction(u5, h, zc5)
    psi, _ = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    sign = -1.0 if psi_co[core] > 0 else 1.0
    return psi_co, sign, a5["rho"], a5["z"], a5["dr"], a5["dz"]


def refine(psi_co, rho, z, dr, dz, k):
    sp = RectBivariateSpline(rho, z, psi_co, kx=3, ky=3)
    rf = (np.arange(int(round(len(rho) * k))) + 0.5) * dr / k
    zf = z[0] - dz / 2 + (np.arange(int(round(len(z) * k))) + 0.5) * dz / k
    return sp(rf, zf), rf, zf, dr / k, dz / k


def main(kappas=(1e-2, 3.3e-3), k=REFINE):
    psi_co, sign, rho, z, dr, dz = real_flow()
    pf, rf, zf, drf, dzf = refine(psi_co, rho, z, dr, dz, k)
    lab = sign * pf
    R, _ = np.meshgrid(rf, zf, indexing="ij")
    region = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    cen, A, B, T, _ = rates.orbit_profiles(lab, R, drf, dzf, region, 1)
    w = 2 * np.pi / T
    print(f"grid {pf.shape} ({pf.size} cells), closed region {int(region.sum())} cells, orbit frequency range [{w.min():.2f},{w.max():.2f}]", flush=True)
    out = {}
    for kap in kappas:
        red = {bc: rates.reduced_rate(lab, R, drf, dzf, region, kap, 1, k_modes=(1,), bc=bc)["rate"]
               for bc in ("dirichlet", "neumann", "mixed")}
        L, _ = to.full_window_operator(pf, rf, drf, dzf, kap)
        true, ev, frac = to.atmosphere_modes(L, region, (w.min(), w.max()), im_min=0.3 * w.min(), n_shifts=6)
        out[kap] = (true, red)
        print(f"kappa={kap:.1e}: TRUE atmosphere rate {true:.4f} (eig {ev}, localisation {frac}); "
              f"reduced dirichlet {red['dirichlet']:.4f}, neumann {red['neumann']:.4f}, mixed {red['mixed']:.4f}", flush=True)
    print("\nreference, from the E1 time series (late-time local rates): kappa=1e-2: 0.07-0.12; kappa=3.3e-3: 0.11-0.21; "
          "registered-window rates: 0.388, 0.456")
    return 0


if __name__ == "__main__":
    sys.exit(main())
