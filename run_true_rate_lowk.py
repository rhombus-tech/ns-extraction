"""Larger-Pe true rates on the real frozen ring: lower kappa on finer grids, with a convergence check per kappa.

EXPLORATORY (not preregistered). Convention fixed before running: a kappa is reported as CONVERGED if the slowest localised mode at
refinement 2 and 2.5 agree within 10%; otherwise it is flagged UNCONVERGED and excluded from the exponent fit.
"""
import sys
import numpy as np
import io_fields as io
import run_ring_case as rc
import scalar_experiment as se
import true_rates as tr

KAPPAS = (1.25e-3, 1.7e-3, 2.5e-3)
REFINES = (2.0, 2.5)


def main():
    ref = np.load("stage1_frozen.npz")
    speed, zc5 = float(ref["speed"]), float(ref["zc"])
    u5, _, _, _, h = rc.load(5)
    a5, _ = se.extraction(u5, h, zc5)
    psi, _ = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    conv = []
    for kap in KAPPAS:
        rates = []
        for k in REFINES:
            r = tr.slowest_mode(psi_co, a5["omega_phi"], a5["rho"], a5["z"], a5["dr"], a5["dz"], kap, refine=k)
            rates.append(r["rate"])
            txt = "none" if r["rate"] is None else f"{r['rate']:.4f} (loc {r['localisation']:.3f})"
            print(f"kappa={kap:.2e} refine {k}: {r['cells']} cells, rate {txt}", flush=True)
        ok = all(x is not None for x in rates) and abs(rates[0] - rates[1]) / rates[1] <= 0.10
        conv.append((kap, rates[-1], ok))
        print(f"   -> {'CONVERGED' if ok else 'UNCONVERGED'}", flush=True)
    good = [(k, r) for k, r, ok in conv if ok]
    if len(good) >= 3:
        slope = np.polyfit(np.log([k for k, _ in good]), np.log([r for _, r in good]), 1)[0]
        print(f"\nexponent over {len(good)} converged kappa: {slope:.3f} (large-Pe theory, non-degenerate shear: 1/3)")
    else:
        print(f"\nonly {len(good)} converged kappa; no exponent reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
