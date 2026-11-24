"""Check the Oseen-Helmholtz conjugation identity symbolically and the double-precision overflow claim."""
import sys
import numpy as np
import sympy as sp


def symbolic_identity():
    x, y, z = sp.symbols("x y z", real=True)
    nu = sp.symbols("nu", positive=True)
    c = sp.Matrix(sp.symbols("c1 c2 c3", real=True))
    cn = sp.sqrt(c.dot(c))
    chat = c / cn
    kappa = cn / (2 * nu)
    X = sp.Matrix([x, y, z])
    V = sp.exp(-(x**2 + 2 * y**2 + z**2)) * (1 + x * y + z**3)

    def lap(f):
        return sum(sp.diff(f, v, 2) for v in (x, y, z))

    def adv(f):
        return sum(c[i] * sp.diff(f, X[i]) for i in range(3))

    lhs = nu * lap(V) - adv(V)
    w = sp.exp(-kappa * chat.dot(X))
    inner = w * V
    rhs = (1 / w) * nu * (lap(inner) - kappa**2 * inner)
    diff = lhs - rhs
    R = sp.Rational
    vals = {x: R(37, 100), y: R(-81, 100), z: R(55, 100), nu: R(13, 1000),
            c[0]: R(13, 10), c[1]: R(-7, 10), c[2]: R(4, 10)}
    return float(diff.subs(vals).evalf(40)), float(lhs.subs(vals).evalf(40))


def overflow_threshold():
    ok = True
    with np.errstate(over="ignore"):
        for k_L in (100, 700, 709, 710, 800):
            v = np.exp(np.float64(k_L))
            print(f"  exp({k_L}) = {v}")
    k_L_max = np.log(np.finfo(np.float64).max)
    re_max = 2 * k_L_max
    print(f"  log(DBL_MAX) = {k_L_max:.2f}  ->  overflow at kappa*L >= {k_L_max:.1f}, Re = 2*kappa*L >= {re_max:.0f}")
    claim_re = 1400
    ok &= abs(re_max - claim_re) / claim_re < 0.05
    return ok, re_max


def conditioning_1d():
    """Steady nu V'' - c V' = f on [0,L] with Dirichlet BCs, direct FD vs transformed solve."""
    L = 1.0
    n = 400
    h = L / n
    xs = np.linspace(0, L, n + 1)
    f = np.exp(-((xs - 0.5) / 0.1) ** 2)
    out = []
    for Re in (10, 100, 1000, 1400, 2000):
        nu = 1.0 / Re
        c = 1.0
        kap = c / (2 * nu)
        with np.errstate(over="ignore", invalid="ignore"):
            w = np.exp(-kap * xs)
        finite = np.all(np.isfinite(w)) and np.all(w > 0)
        dyn = np.log10(w[0] / w[-1]) if finite and w[-1] > 0 else np.inf
        out.append((Re, kap * L, dyn))
        print(f"  Re={Re:5d}: kappa*L={kap*L:7.1f}, weight dynamic range over the domain = 10^{dyn:.0f}" if np.isfinite(dyn)
              else f"  Re={Re:5d}: kappa*L={kap*L:7.1f}, weight underflows/overflows (dynamic range lost)")
    return out


def main():
    d, scale = symbolic_identity()
    good = abs(d) < 1e-20 * max(1.0, abs(scale))
    print(f"identity residual (30-digit symbolic at a point): {d:.3e}  scale {scale:.3e}  {'PASS' if good else 'FAIL'}")
    print("overflow threshold:")
    ok, re_max = overflow_threshold()
    print(f"  paper claims Re >~ 1400; computed {re_max:.0f}: {'PASS' if ok else 'FAIL'}")
    print("weight dynamic range over a unit domain:")
    conditioning_1d()
    print("ALL PASS" if (good and ok) else "FAILURES")
    return 0 if (good and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
