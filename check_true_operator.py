"""Validate true_operator.py, then compare the reduced estimator (rates.py) with the true eigenvalue on circular-orbit flows."""
import sys
import numpy as np
import true_operator as to
import rates

S_A, S_B, Q0 = 0.1, 0.7, 0.16


def hfun(q, m):
    return q + q * q / 2 if m == 1 else q + (q - Q0) ** 3 + Q0**3


def circular(m, n):
    dr = 2.0 / n
    dz = 2.0 / n
    rho = (np.arange(n) + 0.5) * dr
    z = -1 + (np.arange(n) + 0.5) * dz
    R, Z = np.meshgrid(rho, z, indexing="ij")
    s2 = (R - 1) ** 2 + Z**2
    return hfun(s2, m), rho, R, dr, dz, (s2 > S_A**2) & (s2 < S_B**2)


def main():
    ok = True
    import scipy.sparse as sp

    # pure diffusion, no flow, rectangular Dirichlet block: lowest eigenvalue vs separable 1D solves
    n = 120
    dr0 = 1.0 / n
    dz0 = 1.0 / n
    rho0 = 1.0 + (np.arange(n) + 0.5) * dr0
    z0 = (np.arange(n) + 0.5) * dz0
    reg = np.zeros((n + 2, n + 2), dtype=bool)
    reg[1:-1, 1:-1] = True
    rho_p = np.concatenate([[rho0[0] - dr0], rho0, [rho0[-1] + dr0]])
    L = to.build_operator(np.zeros((n + 2, n + 2)), rho_p, dr0, dz0, reg, 1.0, "dirichlet")
    from scipy.sparse.linalg import eigs
    lam = -eigs(L, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0].real
    # separable: z-part pi^2 ; rho-part: smallest eigenvalue of -(f'' + f'/rho) on [1,2] with f=0 at ends
    h = dr0
    r = np.arange(1, n + 1) * h + 1.0
    D = np.zeros((n - 1, n - 1))
    rr = r[:-1]
    for k in range(n - 1):
        D[k, k] = 2 / h**2
        if k > 0:
            D[k, k - 1] = -1 / h**2 + 1 / (2 * h * rr[k])
        if k < n - 2:
            D[k, k + 1] = -1 / h**2 - 1 / (2 * h * rr[k])
    lam_r = np.min(np.linalg.eigvals(D).real)
    expect = lam_r + np.pi**2
    good = abs(lam - expect) / expect < 0.03
    ok &= good
    print(f"pure-diffusion Dirichlet block: lowest eigenvalue {lam:.4f} vs separable {expect:.4f} {'PASS' if good else 'FAIL'}")

    # full-window operator, compactly supported flow, kappa = 0: advection antisymmetric -> purely imaginary spectrum
    n2 = 90
    dr2, dz2 = 2.0 / n2, 2.0 / n2
    rho2 = (np.arange(n2) + 0.5) * dr2
    z2 = -1 + (np.arange(n2) + 0.5) * dz2
    R2, Z2 = np.meshgrid(rho2, z2, indexing="ij")
    psi2 = -np.exp(-(((R2 - 1) ** 2) + Z2**2) / 0.08)
    Lf, _ = to.full_window_operator(psi2, rho2, dr2, dz2, 0.0)
    V2 = (R2 * dr2 * dz2).ravel()
    M2 = (sp.diags(V2) @ Lf).tocsr()
    asym2 = abs(M2 + M2.T).max() / max(abs(M2).max(), 1e-30)
    ev = np.linalg.eigvals(Lf.toarray())
    good = asym2 < 1e-3 and np.max(ev.real) < 1e-3 * np.max(np.abs(ev.imag)) + 1e-9
    ok &= good
    print(f"full window, kappa=0, compact flow: ||M+M^T||/||M|| = {asym2:.1e}, max Re(eig) = {np.max(ev.real):.1e} "
          f"(max |Im| {np.max(np.abs(ev.imag)):.2f}) {'PASS' if good else 'FAIL'}")
    Lk, _ = to.full_window_operator(psi2, rho2, dr2, dz2, 1e-3)
    evk = np.linalg.eigvals(Lk.toarray())
    good = np.max(evk.real) <= 1e-9
    ok &= good
    print(f"full window, kappa=1e-3: no growing mode, max Re(eig) = {np.max(evk.real):.2e} {'PASS' if good else 'FAIL'}")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
