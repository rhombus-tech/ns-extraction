"""Measure the enhanced-dissipation exponent for the orbit-reduced model problem.

On closed streamlines, angle-action reduction gives, for angular mode k,
    H = -kappa d^2/dy^2 + i k w(y),   y in (0,1), Dirichlet,
with w(y) = (y - 1/2)^m  (shear w' has a zero of order m-1 at y=1/2; m=1 is the non-degenerate case w = y-ish).
The decay rate is the spectral abscissa  lambda(kappa) = min Re spec(H).

Candidates for the exponent p in lambda ~ kappa^p:
   BCZ / Pe^(2-beta) form :  p = m/(m+2)            (= kappa * Pe^(2-beta) with beta = 2(m+1)/(m+2))
   Theorem 2.6 form       :  p = 1/beta = (m+2)/(2(m+1))
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

K = 1.0


def operator(kappa, m, n):
    h = 1.0 / (n + 1)
    y = (np.arange(n) + 1) * h
    w = (y - 0.5) ** m if m > 1 else y
    main = 2 * kappa / h**2 + 1j * K * w
    off = -kappa / h**2 * np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csc"), y


def abscissa(kappa, m):
    layer = kappa ** (1.0 / (m + 2))
    n = int(min(max(60 / layer, 400), 60000))
    H, y = operator(kappa, m, n)
    guess = 0.7 * kappa ** (m / (m + 2.0))
    cands = []
    for sigma in (guess, guess * np.exp(1j * np.pi / 3), guess * np.exp(-1j * np.pi / 3), 3 * guess):
        try:
            ev = eigs(H, k=6, sigma=sigma, which="LM", return_eigenvectors=False)
            cands.extend(ev)
        except Exception:
            pass
    cands = np.array(cands)
    return float(np.min(cands.real)), n


def dense_check(kappa, m, n=1500):
    H, _ = operator(kappa, m, n)
    return float(np.min(np.linalg.eigvals(H.toarray()).real))


def main():
    ok = True
    kap = 1e-5
    a, n = abscissa(kap, 2)
    d = dense_check(kap, 2)
    good = abs(a - d) / d < 0.02
    ok &= good
    print(f"sanity (kappa=1e-5, m=2, N={n} sparse vs dense N=1500): {a:.5e} vs {d:.5e} {'PASS' if good else 'FAIL'}")

    kappas = np.logspace(-9, -5, 9)
    summary = []
    for m in (1, 2, 3):
        lam = np.array([abscissa(k, m)[0] for k in kappas])
        slope = np.polyfit(np.log(kappas[2:]), np.log(lam[2:]), 1)[0]
        p_bcz = m / (m + 2)
        beta = 2 * (m + 1) / (m + 2)
        p_thm = 1 / beta
        print(f"m={m}: beta={beta:.3f}; measured exponent p = {slope:.4f}   "
              f"BCZ kappa*Pe^(2-beta) -> {p_bcz:.4f};  Thm 2.6 kappa^(1/beta) -> {p_thm:.4f}")
        print("   kappa, lambda: " + "  ".join(f"{k:.0e}:{l:.2e}" for k, l in zip(kappas, lam)))
        summary.append((m, slope, p_bcz, p_thm))
        good = abs(slope - p_bcz) < 0.03
        ok &= good
        print(f"   matches BCZ form within 0.03: {'PASS' if good else 'FAIL'}; "
              f"matches Thm 2.6 form within 0.03: {abs(slope - p_thm) < 0.03}")
    print("ALL PASS (measured exponents agree with kappa^(m/(m+2)))" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
