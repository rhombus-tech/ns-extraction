"""Exact reduction of the PLANAR ELLIPTICAL Gaussian vortex core mode to coupled radial ODEs, and its numerical solution.

Flow: psi = -P exp(-q/2), q = x^2/a^2 + y^2/b^2.  Contours are similar ellipses; with x = a r cos(theta), y = b r sin(theta):
  * the advection is EXACTLY  omega(r) d/dtheta,  omega(r) = (P/(a b)) exp(-r^2/2) = -psi/(a b)   (theta is the exact angle variable);
  * the Laplacian in (r, theta) has coefficients s1 = (1/a^2 + 1/b^2)/2 and d = (1/a^2 - 1/b^2)/2:
        Lap c = (1/r) d_r( r (s1 + d cos2t) c_r ) - (d/r) d_r( sin2t c_t ) - (d/r) d_t( sin2t c_r ) + (1/r^2) d_t( (s1 - d cos2t) c_t ).
For c = sum_n f_n(r) e^{i n theta} (derived by hand; checked here against the d = 0 circular problem and against grid convergence):
  (Lap c)_n   gets  s1 (f_n'' + f_n'/r - n^2 f_n / r^2)
  (Lap c)_{n+2} gets (d/2) ( f_n'' - (2n+1) f_n'/r + n(n+2) f_n / r^2 )
  (Lap c)_{n-2} gets (d/2) ( f_n'' + (2n-1) f_n'/r + n(n-2) f_n / r^2 )
Eigenproblem (c ~ e^{-lambda t}):  lambda f_n = i n omega(r) f_n - kappa (Lap f)_n ,  f_n(0) = f_n(R) = 0.
Slowest core mode: n = 1, coupled to n = -1, +3, ...  (odd n only).
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

OMC = 2.0        # core angular frequency omega_c
AB = 0.09        # a*b (fixed), so omega_c = P/(a b) = 2 with P = 0.18


def coefficients(e):
    """e = b/a ellipticity; a b fixed.  Returns (s1, d, closed-form trace factor)."""
    a2, b2 = AB / e, AB * e                  # a^2, b^2
    s1 = 0.5 * (1 / a2 + 1 / b2)
    d = 0.5 * (1 / a2 - 1 / b2)
    return s1, d


def closed_form(kappa, e):
    """sqrt(2 kappa |omega'| (h1+h2)) with omega' = 1/(a b), h1+h2 = P (1/a^2 + 1/b^2):  = sqrt(4 kappa omega_c s1)."""
    s1, _ = coefficients(e)
    return float(np.sqrt(4 * kappa * OMC * s1))


def build(kappa, e, nmodes=8, N=2400, R=5.0):
    s1, d = coefficients(e)
    h = R / N
    r = h * np.arange(1, N)
    M = N - 1
    ns = [2 * j - (nmodes - 1) for j in range(nmodes)]   # odd, symmetric: e.g. -7..7
    om = OMC * np.exp(-r**2 / 2)

    D2 = sp.diags([np.ones(M - 1), -2 * np.ones(M), np.ones(M - 1)], [-1, 0, 1]) / h**2
    D1 = sp.diags([-np.ones(M - 1), np.ones(M - 1)], [-1, 1]) / (2 * h)
    Ri = sp.diags(1 / r)
    Ri2 = sp.diags(1 / r**2)
    I = sp.identity(M)
    blocks = {}
    for ia, n in enumerate(ns):
        diag = (-kappa * s1 * (D2 + Ri @ D1 - n * n * Ri2)).tocsr().astype(complex) + (1j * n * sp.diags(om)).tocsr()
        blocks[(ia, ia)] = diag
        if ia + 1 < nmodes:      # f_n -> mode n+2
            blocks[(ia + 1, ia)] = -kappa * 0.5 * d * (D2 - (2 * n + 1) * Ri @ D1 + n * (n + 2) * Ri2)
        if ia - 1 >= 0:         # f_n -> mode n-2
            blocks[(ia - 1, ia)] = -kappa * 0.5 * d * (D2 + (2 * n - 1) * Ri @ D1 + n * (n - 2) * Ri2)
    grid = [[blocks.get((i, j)) for j in range(nmodes)] for i in range(nmodes)]
    return sp.bmat(grid, format="csc", dtype=complex), ns, r


def core_rate(kappa, e, nmodes=8, N=2400, R=5.0):
    A, ns, r = build(kappa, e, nmodes, N, R)
    cf = closed_form(kappa, e)
    sigma = 1j * OMC + cf * (1 - 1j)                   # leading-order prediction (Re = Im = rate, mode n=1)
    ev, vec = eigs(A, k=6, sigma=sigma, which="LM")
    j = int(np.argmin(np.abs(ev - sigma)))
    lam = ev[j]
    v = vec[:, j]
    M = len(r)
    w = np.array([np.sum(np.abs(v[i * M:(i + 1) * M]) ** 2) for i in range(len(ns))])
    w /= w.sum()
    return float(lam.real), complex(lam), dict(zip(ns, np.round(w, 6)))


def main():
    print("circular check (e = 1): multimode solver vs the independent radial result 'Re mu = sqrt(8 k alpha) - 6 k gamma/alpha' (derive_finite_size.py)")
    for kappa in (3e-4,):
        lam, _, wts = core_rate(kappa, 1.0)
        s1, _ = coefficients(1.0)
        pred = np.sqrt(4 * kappa * OMC * s1) - 1.5 * kappa * s1
        print(f"  kappa={kappa:g}: Re lambda = {lam:.6f}, circular theory {pred:.6f}, ratio {lam / pred:.5f}; mode weights {wts}")
    print("\nconvergence for e = 2, kappa = 3e-4:")
    for nm, N in ((6, 2400), (8, 2400), (10, 2400), (8, 3600), (8, 4800)):
        lam, _, _ = core_rate(3e-4, 2.0, nm, N)
        print(f"  modes {nm:2d}, N {N}: Re lambda = {lam:.7f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
