"""Spectral lower bound Sigma(alpha) of the linearised Navier-Stokes operator around the Lamb-Oseen vortex, computed directly.

Operator (Li-Wei-Zhang, arXiv:1701.06269, eqs 1.5-1.8; self-similar variables, tau = log t):
    d_tau w = (L - alpha Lambda) w ,   L = Delta + (xi/2).grad + 1 ,   Lambda w = v_G.grad w + u.grad G ,  u = K_BS * w,
    G = e^{-|xi|^2/4}/(4 pi),  v_G = (xi_perp/(2 pi |xi|^2)) (1 - e^{-|xi|^2/4}).
For angular mode n, w = f(r) e^{i n theta}:
    L_n f   = f'' + f'/r - n^2 f/r^2 + (r/2) f' + f
    Lambda f = i n Omega(r) f + (i n / 2) G(r) psi_n(r),   Omega = (1 - e^{-r^2/4})/(2 pi r^2),   psi_n'' + psi_n'/r - n^2 psi_n/r^2 = f
(u = grad^perp psi gives u_r = -i n psi_n / r, and G' = -(r/2) G).  Exact check used: the translation mode f = G' satisfies
Lambda f = 0, so (L - alpha Lambda) f = -f/2 for every alpha.  Sigma_n(alpha) = min over eigenvalues of  -Re(lambda)  (excluding the translation mode for n = 1).
Passive-scalar core-mode prediction in the same units (kappa = 1, Omega ~ w0 - w2 r^2, w2 = alpha/(64 pi)):
    Re lambda = 2 (n+1) cos(pi/4) sqrt(n w2).
"""
import sys
import numpy as np
from scipy.linalg import eigvals

R_MAX = 16.0


def build(n, alpha, N, nonlocal_term=True):
    h = R_MAX / (N + 1)
    r = h * np.arange(1, N + 1)
    Om = np.where(r < 1e-3, 1 / (8 * np.pi), (1 - np.exp(-r**2 / 4)) / (2 * np.pi * r**2))
    G = np.exp(-r**2 / 4) / (4 * np.pi)
    I = np.eye(N)
    d2 = (np.diag(np.full(N - 1, 1.0), 1) + np.diag(np.full(N - 1, 1.0), -1) - 2 * I) / h**2
    d1 = (np.diag(np.full(N - 1, 1.0), 1) - np.diag(np.full(N - 1, 1.0), -1)) / (2 * h)
    lap = d2 + np.diag(1 / r) @ d1 - np.diag(n**2 / r**2)
    Lmat = lap + np.diag(r / 2) @ d1 + I
    # Poisson operator with psi(0) = 0 and the decaying condition psi' = -n psi / r at the outer end
    P = lap.copy()
    # ghost psi_{N+1} = psi_{N-1} - 2 h n psi_N / r_N: its d2 part adds +1/h^2 to psi_{N-1} and -2n/(h r_N) to psi_N;
    # its d1 part psi_{N+1}/(2 h r_N) adds +1/(2 h r_N) to psi_{N-1} and -n/r_N^2 to psi_N
    P[N - 1, N - 2] += 1 / h**2 + 1 / (2 * h * r[-1])
    P[N - 1, N - 1] += -2 * n / (h * r[-1]) - n / r[-1] ** 2
    Pinv = np.linalg.inv(P)
    # -alpha*Lambda with Lambda f = i n Omega f + (i n / 2) G psi_n   (u = grad^perp psi, so u_r = -i n psi_n / r)
    M = Lmat - 1j * alpha * n * np.diag(Om)
    if nonlocal_term:  # the vorticity-coupling term u.grad G; dropping it gives a PASSIVE SCALAR advected by the Oseen flow
        M = M - 0.5j * alpha * n * np.diag(G) @ Pinv
    return M


def sigma(n, alpha, N):
    lam = eigvals(build(n, alpha, N))
    if n == 1:
        lam = lam[~((np.abs(lam.real + 0.5) < 1e-3) & (np.abs(lam.imag) < 1e-3))]  # translation mode, in ker Lambda
    k = np.argmax(lam.real)
    return -lam[k].real, lam[k].imag


def main():
    ok = True
    print("alpha = 0 check (exact spectrum of L on angular mode n is -n/2, -n/2-1, ...):")
    for n in (1, 2, 3):
        lam = np.sort(eigvals(build(n, 0.0, 1200)).real)[::-1][:3]
        good = np.allclose(lam, [-n / 2, -n / 2 - 1, -n / 2 - 2], atol=2e-3)
        ok &= good
        print(f"  n={n}: {np.round(lam, 4)} expected {[-n/2, -n/2-1, -n/2-2]} {'PASS' if good else 'FAIL'}")
    print("\nSigma_n(alpha) from the full operator (nonlocal term included) vs passive-scalar core-mode closed form:")
    rows = []
    for n in (1, 2, 3):
        for alpha, N in ((1e3, 1400), (1e4, 1600), (1e5, 1800)):
            s, im = sigma(n, alpha, N)
            w2 = alpha / (64 * np.pi)
            pred = 2 * (n + 1) * np.cos(np.pi / 4) * np.sqrt(n * w2)
            rows.append((n, alpha, s, pred))
            print(f"  n={n} alpha={alpha:.0e}: Sigma = {s:9.3f} (Im {im:+10.2f})  closed form {pred:9.3f}  ratio {s/pred:.3f}", flush=True)
    for n in (1, 2, 3):
        s = [r[2] for r in rows if r[0] == n]
        a = [r[1] for r in rows if r[0] == n]
        print(f"  n={n}: exponent of Sigma in alpha over 1e3..1e5: {np.polyfit(np.log(a), np.log(s), 1)[0]:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
