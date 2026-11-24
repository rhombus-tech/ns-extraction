"""True (2D-operator) decay rates on the real frozen ring flow across kappa, plus the measured scaling exponent."""
import sys
import numpy as np
import io_fields as io
import run_ring_case as rc
import scalar_experiment as se
import true_rates as tr

KAPPAS = (2.5e-3, 3.3e-3, 5e-3, 7e-3, 1e-2, 2e-2)


def main():
    ref = np.load("stage1_frozen.npz")
    speed, zc5 = float(ref["speed"]), float(ref["zc"])
    u5, _, t5, _, h = rc.load(5)
    a5, _ = se.extraction(u5, h, zc5)
    psi, _ = io.stream_function(a5["u_rho"], a5["u_z"], a5["rho"], a5["dr"], a5["dz"])
    psi_co = io.to_comoving(psi, a5["rho"], speed)
    rows = []
    for kap in KAPPAS:
        res = tr.slowest_mode(psi_co, a5["omega_phi"], a5["rho"], a5["z"], a5["dr"], a5["dz"], kap, refine=1.5)
        rows.append((kap, res))
        if res["rate"] is None:
            print(f"kappa={kap:.1e}: {res['message']}", flush=True)
        else:
            print(f"kappa={kap:.1e}: rate {res['rate']:.4f}  eig {res['eigenvalue']:.3f}  localisation {res['localisation']:.3f}", flush=True)
    good = [(k, r["rate"]) for k, r in rows if r["rate"] is not None]
    if len(good) >= 3:
        lk = np.log([k for k, _ in good])
        lr = np.log([r for _, r in good])
        slope = np.polyfit(lk, lr, 1)[0]
        print(f"\nscaling over the {len(good)} kappa with a localised mode: d ln rate / d ln kappa = {slope:.3f} "
              f"(large-Pe theory for a non-degenerate shear: 1/3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
