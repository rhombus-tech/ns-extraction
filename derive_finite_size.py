"""Finite-size correction to the vortex-core mode for CIRCULAR vortices (planar, angular velocity Omega(r)), derived by first-order perturbation theory and checked
against an independent radial eigenvalue solver.

Model: c_t + Omega(r) c_theta = kappa Laplacian c, mode c = f(r) exp(i theta) exp(-lambda t), lambda = i Omega_c + mu,
    mu f = -kappa (f'' + f'/r - f/r^2) + i (Omega_c - Omega(r)) f ,   f(0) = f(R) = 0,
    Omega(r) = Omega_c - alpha r^2 + gamma r^4 + ...
Leading: f0 = r exp(-beta r^2), beta^2 = -i alpha / (4 kappa), mu0 = 8 kappa beta, Re mu0 = sqrt(8 kappa alpha)   (= the closed form sqrt(2 kappa |omega'| (h1 + h2))).
First order in the r^4 term (bilinear form, the operator is complex-symmetric):  <r^4> = 6 i kappa / alpha, delta mu = i gamma <r^4> = -6 kappa gamma / alpha  (real).
So   Re lambda = sqrt(8 kappa alpha) - 6 kappa gamma / alpha + O(kappa^(3/2)).
Convention fixed before running: three profiles x four kappa; agreement is reported as the ratio of the solver's Re(mu) to the leading closed form and to the corrected one; no threshold tuned.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

OMC, B = 2.0, 0.3
PROFILES = {
    "Gaussian        Omega = Oc exp(-r^2/2b^2)": (lambda r: OMC * np.exp(-r**2 / (2 * B**2)), OMC / (2 * B**2), OMC / (8 * B**4)),
    "Lamb-Oseen-like Omega = Oc (1-exp(-r^2/b^2))/(r^2/b^2)": (lambda r: OMC * (1 - np.exp(-(r / B) ** 2)) / (r / B) ** 2, OMC / (2 * B**2), OMC / (6 * B**4)),
    "Algebraic       Omega = Oc / (1+r^2/b^2)^2": (lambda r: OMC / (1 + (r / B) ** 2) ** 2, 2 * OMC / B**2, 3 * OMC / B**4),
}


def radial_mu(Omega, kappa, mu_guess, R=1.6, n=3200):
    h = R / n
    r = h * np.arange(1, n)                      # unknowns f_1 .. f_{n-1}, f_0 = f_n = 0
    om = Omega(r)
    main = 2 * kappa / h**2 + kappa / r**2 + 1j * (OMC - om)
    up = -kappa / h**2 - kappa / (2 * h * r)    # coefficient of f_{j+1}
    lo = -kappa / h**2 + kappa / (2 * h * r)    # coefficient of f_{j-1}
    M = sp.diags([lo[1:], main, up[:-1]], [-1, 0, 1], format="csc", dtype=complex)
    ev = eigs(M, k=4, sigma=mu_guess, which="LM", return_eigenvectors=False)
    return ev[np.argmin(np.abs(ev - mu_guess))]


def main():
    print(f"{'profile':56s} kappa      solver Re(mu)  leading   corrected   solver/leading  solver/corrected")
    worst_lead, worst_corr = 0.0, 0.0
    for name, (Om, alpha, gamma) in PROFILES.items():
        for kappa in (1e-3, 3e-4, 1e-4, 3e-5):
            lead = np.sqrt(8 * kappa * alpha)
            corr = lead - 6 * kappa * gamma / alpha
            guess = 4 * np.sqrt(kappa * alpha) * np.exp(-1j * np.pi / 4) * 0.98
            mu = radial_mu(Om, kappa, guess)
            print(f"{name:56s} {kappa:.0e}   {mu.real:.5f}      {lead:.5f}   {corr:.5f}     {mu.real / lead:.4f}          {mu.real / corr:.4f}", flush=True)
            worst_lead = max(worst_lead, abs(mu.real / lead - 1))
            worst_corr = max(worst_corr, abs(mu.real / corr - 1))
    print(f"\nmax |ratio-1|: leading form {worst_lead:.3f}, corrected form {worst_corr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
