"""SUPERSEDED: written for the earlier Cartesian-oscillator formula. With the orbit-averaged formula now in core_mode the ring-2 ratios are
1.09-1.12 (see verify_core_mode_v2.py and the guide's validation record); the 10% criterion below was for the old formula.

Out-of-sample check of orbits.core_mode on a DIFFERENT ring: nu = 0.005 (the first ring had nu = 0.01), frozen t = 2.5.

Convention fixed before running (no refitting allowed): core_mode is evaluated with the library defaults; the 2D operator rates are
slowest_mode(refine=1.5) at kappa in {2.5e-3, 5e-3}. A kappa counts only if a localised mode exists. PASS-level agreement = within 10%.
"""
import os
import sys
import numpy as np
import io_fields as io
import orbits as ob
import run_ring_case as rc
import scalar_experiment as se
import sim_ns3d
import true_rates as tr
import analyze_stage1 as a

NU2 = 0.005
OUT = "data_nu005"
KAPPAS = (2.5e-3, 5e-3)


def main():
    if not os.path.exists(f"{OUT}/ring_006.npz"):
        sim_ns3d.run("ring", 96, NU2, 3.0, outdir=OUT)
    snaps = {i: np.load(f"{OUT}/ring_{i:03d}.npz") for i in (4, 5, 6)}
    h = float(snaps[5]["h"])
    ex = {}
    for i, d in snaps.items():
        t = float(d["t"])
        ex[i] = (se.extraction(d["u"].astype(float), h, np.pi + 0.5 * t), t)
    (a4, _), t4 = ex[4]
    (a5, zc5), t5 = ex[5]
    (a6, _), t6 = ex[6]
    speed = io.ring_speed(a4, t4, a6, t6)
    psi, res = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    print(f"ring 2: nu={NU2}, speed {speed:.3f}, zc {zc5:.3f}, psi residual {res:.1e}", flush=True)

    core = np.unravel_index(np.argmax(np.abs(a5["omega_phi"])), a5["omega_phi"].shape)
    sign = -1.0 if psi_co[core] > 0 else 1.0
    lab = sign * psi_co
    R, _ = np.meshgrid(a5["rho"], a5["z"], indexing="ij")
    region = ob.closed_region(lab, np.ones_like(lab, dtype=bool), 0.02 * lab.min())
    lev, om = ob.orbit_frequency(lab, R, a5["dr"], a5["dz"], region)
    u, ws, sh, noise = ob.smooth_shear(lev, om)
    print(f"orbit frequency {om.max():.2f} -> {om.min():.2f}; shear zeros {ob.shear_zero_order(u, sh)}; SNR {np.abs(sh).max()/noise:.0f}", flush=True)

    worst = 0.0
    for kap in KAPPAS:
        cm = ob.core_mode(lab, R, a5["dr"], a5["dz"], region, kap)
        res_t = tr.slowest_mode(psi_co, a5["omega_phi"], a5["rho"], a5["z"], a5["dr"], a5["dz"], kap, refine=1.5)
        if res_t["rate"] is None:
            print(f"kappa={kap:.1e}: core_mode {cm['rate']:.4f}; 2D operator: no localised mode", flush=True)
            continue
        ratio = cm["rate"] / res_t["rate"]
        worst = max(worst, abs(ratio - 1))
        print(f"kappa={kap:.1e}: core_mode {cm['rate']:.4f} (Hessian {cm['hessian_eigenvalues'][0]:.2f},{cm['hessian_eigenvalues'][1]:.2f}, "
              f"shear {cm['core_shear']:.2f}) vs 2D operator {res_t['rate']:.4f} (loc {res_t['localisation']:.3f}); ratio {ratio:.3f}", flush=True)
    print(f"worst deviation {worst:.3f} -> {'within 10%' if worst <= 0.10 else 'OUTSIDE 10%'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
