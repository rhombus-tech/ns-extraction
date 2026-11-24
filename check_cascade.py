"""Check Theorem (cascade spectral structure): Jacobi matrix, zero diagonal, off-diagonal alpha."""
import numpy as np

ALPHA = 1 / np.sqrt(3)


def spectral_radius(n, alpha=ALPHA, offdiag=None):
    off = np.full(n - 1, alpha) if offdiag is None else offdiag
    m = np.diag(off, 1) + np.diag(off, -1)
    return np.max(np.abs(np.linalg.eigvalsh(m)))


def main():
    ok = True
    checks = [
        (3, np.sqrt(2) * ALPHA, "rho_3 = sqrt2*alpha"),
        (5, np.sqrt(3) * ALPHA, "rho_5 = sqrt3*alpha (=1)"),
    ]
    for n, expected, label in checks:
        got = spectral_radius(n)
        good = abs(got - expected) < 1e-12
        ok &= good
        print(f"N={n}: {got:.12f} expected {expected:.12f} [{label}] {'PASS' if good else 'FAIL'}")

    for n in (6, 10, 50, 500):
        got = spectral_radius(n)
        bound = 2 * ALPHA * np.cos(np.pi / (n + 1))
        good = abs(got - bound) < 1e-10 and got > 1
        ok &= good
        print(f"N={n}: {got:.9f} (2a cos(pi/(N+1)) = {bound:.9f}) >1: {got > 1} {'PASS' if good else 'FAIL'}")

    limit = 2 * ALPHA
    print(f"N->inf limit 2/sqrt3 = {limit:.6f}; N=5000 gives {spectral_radius(5000):.6f}")

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(2, 40))
        off = rng.uniform(-ALPHA, ALPHA, n - 1)
        r = spectral_radius(n, offdiag=off)
        worst = max(worst, r / (2 * ALPHA * np.cos(np.pi / (n + 1))))
    good = worst <= 1 + 1e-12
    ok &= good
    print(f"random |M_k,k+1|<=alpha: max ratio to bound = {worst:.6f} {'PASS' if good else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
