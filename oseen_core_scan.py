"""Core-localised eigenvalue of the Oseen-vortex operator across alpha: passive scalar (no nonlocal term) vs full linearised NS (nonlocal term).

Passive-scalar closed form in the same units (diffusion coefficient 1, core frequency curvature w2 = alpha/(64 pi)):
    Re lambda = 2 (n+1) cos(pi/4) sqrt(n w2).
Core mode = eigenmode with >= 90% of int |f|^2 r dr inside r < 4 x0, x0 = (64 pi/(n alpha))^(1/4); the translation mode is excluded.
"""
import sys
import numpy as np
from scipy.linalg import eig
import oseen_sigma as o

CASES = ((1e3, 1400), (3e3, 1600), (1e4, 1600), (3e4, 2000), (1e5, 2400), (3e5, 2800))


def core(n, alpha, N, nonlocal_term):
    M = o.build(n, alpha, N, nonlocal_term=nonlocal_term)
    lam, V = eig(M)
    h = o.R_MAX / (N + 1)
    r = h * np.arange(1, N + 1)
    G = np.exp(-r**2 / 4) / (4 * np.pi)
    f1 = -(r / 2) * G
    x0 = (64 * np.pi / (n * alpha)) ** 0.25
    best = None
    for k in range(len(lam)):
        v = V[:, k]
        w = np.abs(v) ** 2 * r
        w /= w.sum()
        if n == 1 and abs(np.vdot(f1, v)) / (np.linalg.norm(f1) * np.linalg.norm(v)) > 0.9:
            continue
        if w[r < 4 * x0].sum() >= 0.9 and -lam[k].real > 0 and (best is None or -lam[k].real < best[0]):
            best = (-lam[k].real, lam[k].imag)
    return best


def main():
    for n in (1, 2):
        print(f"\nn={n}", flush=True)
        res = {True: [], False: []}
        for alpha, N in CASES:
            pred = 2 * (n + 1) * np.cos(np.pi / 4) * np.sqrt(n * alpha / (64 * np.pi))
            row = f"  alpha={alpha:.0e} (N={N}): closed form {pred:8.2f}"
            for nl, name in ((False, "passive"), (True, "full NS")):
                b = core(n, alpha, N, nl)
                if b is None:
                    row += f" | {name}: no core mode"
                else:
                    res[nl].append((alpha, b[0]))
                    row += f" | {name}: {b[0]:8.2f} (ratio {b[0]/pred:.3f}, Im {b[1]:+.0f})"
            print(row, flush=True)
        for nl, name in ((False, "passive"), (True, "full NS")):
            if len(res[nl]) >= 3:
                a, s = zip(*res[nl])
                print(f"  n={n} {name}: exponent d ln Sigma / d ln alpha = {np.polyfit(np.log(a), np.log(s), 1)[0]:.3f} over {len(a)} values", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
