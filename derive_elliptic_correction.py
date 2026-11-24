"""Finite-size correction of the vortex-core mode for a core with SIMILAR ELLIPTICAL contours (planar elliptical Gaussian family).

Result:   Re lambda = sqrt(8 kappa s1 alpha) - 6 kappa s1 gamma / alpha + O(kappa^(3/2)),
with s1 = (1/a^2 + 1/b^2)/2 and omega(r) = omega_c - alpha r^2 + gamma r^4 the orbit angular frequency in the ELLIPTIC radius r (x = a r cos t, y = b r sin t).
For the Gaussian vortex alpha = omega_c/2, gamma = omega_c/8, so  Re lambda = sqrt(4 kappa omega_c s1) - (3/2) kappa s1.

Derivation (see elliptic_core_multimode.py for the exact reduction to coupled radial ODEs):
  * the n = 1 diagonal block is exactly the circular problem with kappa -> kappa s1, giving the circular result of derive_finite_size.py;
  * the anisotropy d = (1/a^2 - 1/b^2)/2 couples n = 1 to n = 3 and n = -1 only (n -> n +- 2);
  * second-order shift:  (kappa d / 2)^2 [ I_3 / (-2 i omega_c) + I_{-1} / (+2 i omega_c) ]  with I_3 = I_{-1} = 24 beta^2  (computed below), so the two channels CANCEL
    at leading order and the coupling first enters at relative order sqrt(kappa), i.e. O(kappa^(3/2)).
This script (1) recomputes the matrix elements symbolically, (2) checks the formula against the exact multimode solution for ellipticity 1..3 and kappa 3e-5..1e-3.
"""
import sys
import numpy as np
import sympy as sp
import elliptic_core_multimode as m


def symbolic_cancellation():
    r = sp.symbols("r", positive=True)
    b = sp.symbols("b", positive=True)           # beta (continued to complex values afterwards)
    f0 = r * sp.exp(-b * r**2)
    D = lambda f, k=1: sp.diff(f, r, k)
    B3 = lambda f: D(f, 2) - 3 * D(f) / r + 3 * f / r**2
    A3 = lambda g: D(g, 2) + 5 * D(g) / r + 3 * g / r**2
    Bm = lambda f: D(f, 2) + D(f) / r - f / r**2
    Am = Bm
    w = lambda expr: sp.integrate(sp.simplify(r * f0 * expr), (r, 0, sp.oo))
    N = w(f0)
    I3 = sp.simplify(w(A3(B3(f0))) / N)
    Im = sp.simplify(w(Am(Bm(f0))) / N)
    return I3, Im


def main():
    I3, Im = symbolic_cancellation()
    print(f"matrix elements: I_3 = {I3}, I_-1 = {Im}  ->  equal: {sp.simplify(I3 - Im) == 0}; denominators -2i w_c and +2i w_c: second-order sum = "
          f"(kappa d/2)^2 * ({I3}) * (-1/(2 i w_c) + 1/(2 i w_c)) = 0")
    print("\nellipticity | kappa | exact Re lambda | formula sqrt(4 k w s1) - 1.5 k s1 | exact/formula - 1")
    worst = 0.0
    for e in (1.0, 1.25, 1.5, 2.0, 3.0):
        s1, _ = m.coefficients(e)
        for k in (1e-3, 3e-4, 1e-4, 3e-5):
            lam, _, _ = m.core_rate(k, e, nmodes=6, N=2400)
            f = np.sqrt(4 * k * m.OMC * s1) - 1.5 * k * s1
            lead = m.closed_form(k, e)
            worst = max(worst, abs(lam / f - 1))
            print(f"  e={e:4.2f}  k={k:.0e}  {lam:.6f}  {f:.6f}  {lam / f - 1:+.2e}   (leading term alone: {lam / lead - 1:+.3f})", flush=True)
    print(f"\nmax |exact/formula - 1| over all 20 cases = {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
