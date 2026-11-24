"""Core-localised eigenvalue of the Oseen vortex operator vs the passive-scalar core-mode closed form, with convergence in N.

Eigenvectors are used to (i) drop the translation mode (overlap with G') and (ii) select modes localised near the core,
r < 4 x0 with x0 = (64 pi /(n alpha))^(1/4) the oscillator length (fraction of int |f|^2 r dr >= 0.9).
The GLOBAL infimum Sigma is reported only to show whether it is numerically converged (it is set by the weakly rotating outskirts).
"""
import sys
import numpy as np
from scipy.linalg import eig
import oseen_sigma as o


def analyse(n, alpha, N):
    M = o.build(n, alpha, N)
    lam, V = eig(M)
    h = o.R_MAX / (N + 1)
    r = h * np.arange(1, N + 1)
    G = np.exp(-r**2 / 4) / (4 * np.pi)
    f1 = -(r / 2) * G
    x0 = (64 * np.pi / (n * alpha)) ** 0.25
    out = []
    for k in range(len(lam)):
        v = V[:, k]
        w = np.abs(v) ** 2 * r
        w /= w.sum()
        trans = n == 1 and abs(np.vdot(f1, v)) / (np.linalg.norm(f1) * np.linalg.norm(v)) > 0.9
        out.append((-lam[k].real, lam[k].imag, float(w[r < 4 * x0].sum()), float((w * r).sum()), trans))
    out = [x for x in out if not x[4] and x[0] > 0]
    glob = min(out, key=lambda x: x[0])
    core = [x for x in out if x[2] >= 0.9]
    corem = min(core, key=lambda x: x[0]) if core else None
    return glob, corem, x0


def main():
    for n, alpha in ((2, 1e4), (1, 1e4)):
        print(f"\nn={n}, alpha={alpha:.0e}: convergence in N", flush=True)
        for N in (1000, 1400, 2000):
            g, c, x0 = analyse(n, alpha, N)
            pred = 2 * (n + 1) * np.cos(np.pi / 4) * np.sqrt(n * alpha / (64 * np.pi))
            ctxt = "no core-localised mode" if c is None else f"core mode Sigma={c[0]:.3f} (Im {c[1]:+.1f}, core weight {c[2]:.2f})  ratio to closed form {c[0]/pred:.3f}"
            print(f"  N={N}: global Sigma={g[0]:.3f} (Im {g[1]:+.1f}, mean r {g[3]:.2f}); {ctxt}; closed form {pred:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
