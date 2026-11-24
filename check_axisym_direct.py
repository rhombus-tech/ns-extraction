"""Direct axisymmetric test of the enhanced-dissipation exponent with the 5D operator.

Flow: psi = h(s^2), s^2 = (rho-1)^2 + z^2 (circular orbits around the ring core at rho=1, z=0).
  poloidal advection in polar coords (s,theta): d_t + v d_theta, v = 2 h'(s^2)/rho, rho = 1 + s cos(theta);
  orbit angular frequency w_orb(psi) = 2 h'(s^2) exactly (mean of rho over theta is 1).
Evolved quantity: the orbit-varying part dOmega = (I-P) Omega, with the operator (I-P) L (I-P), where
  L = kappa*Delta_5 - u_p.grad and P is the orbit average (the orbit-constant part decays only at the
  molecular rate and is excluded; the source S = Delta_5 g(psi) is kept out of the homogeneous operator),
  Delta_5 = d_ss + (1/s) d_s + (1/s^2) d_thth + (3/rho)(cos th d_s - (sin th / s) d_th).
Annulus s in [s_a, s_b], Dirichlet walls. Time stepping: Crank-Nicolson, sparse LU.

m=1: h' = 1 + q          (shear nonzero everywhere)       expect rate ~ kappa^(1/3)
m=2: h' = 1 + 3 (q-q0)^2 (simple shear zero at q0)        expect rate ~ kappa^(1/2)
(Theorem 2.6 as written would give kappa^(3/4), kappa^(2/3).)
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

S_A, S_B, Q0 = 0.1, 0.7, 0.16


def hprime(q, m):
    return 1 + q if m == 1 else 1 + 3 * (q - Q0) ** 2


def build(kappa, m, ns, nth):
    h = (S_B - S_A) / (ns + 1)
    dth = 2 * np.pi / nth
    rows, cols, vals = [], [], []

    def idx(i, j):
        return i * nth + (j % nth)

    for i in range(ns):
        s = S_A + (i + 1) * h
        for j in range(nth):
            th = j * dth
            rho = 1 + s * np.cos(th)
            v = 2 * hprime(s * s, m) / rho
            a_s = 1 / s + 3 * np.cos(th) / rho
            b_th = -3 * kappa * np.sin(th) / (s * rho) - v
            r = idx(i, j)
            c_ss = kappa / h**2
            c_thth = kappa / (s * s * dth**2)
            diag = -2 * c_ss - 2 * c_thth
            ent = {r: diag}
            if i + 1 < ns:
                ent[idx(i + 1, j)] = ent.get(idx(i + 1, j), 0) + c_ss + kappa * a_s / (2 * h)
            if i - 1 >= 0:
                ent[idx(i - 1, j)] = ent.get(idx(i - 1, j), 0) + c_ss - kappa * a_s / (2 * h)
            ent[idx(i, j + 1)] = ent.get(idx(i, j + 1), 0) + c_thth + b_th / (2 * dth)
            ent[idx(i, j - 1)] = ent.get(idx(i, j - 1), 0) + c_thth - b_th / (2 * dth)
            for c, val in ent.items():
                rows.append(r)
                cols.append(c)
                vals.append(val)
    n = ns * nth
    L = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    s_arr = S_A + (np.arange(ns) + 1) * h
    th_arr = np.arange(nth) * dth
    SS, TT = np.meshgrid(s_arr, th_arr, indexing="ij")
    W = (SS * (1 + SS * np.cos(TT))).ravel()
    return L, SS, TT, W


def orbit_projector(SS, TT):
    """P f = orbit average on each circle s=const with the invariant measure rho d(theta) (dt = rho dtheta / 2h')."""
    ns, nth = SS.shape
    rho = 1 + SS * np.cos(TT)
    blocks = [sp.csr_matrix(np.tile(rho[i] / rho[i].sum(), (nth, 1))) for i in range(ns)]
    return sp.block_diag(blocks, format="csc")


def decay_rate(kappa, m, ns=120, nth=48, dt=0.05, tmax=2500.0, stop_ratio=1e-9):
    L, SS, TT, W = build(kappa, m, ns, nth)
    n = L.shape[0]
    I = sp.identity(n, format="csc")
    Q = (I - orbit_projector(SS, TT)).tocsc()
    L = (Q @ L @ Q).tocsc()
    lu = splu((I - 0.5 * dt * L).tocsc())
    rhs_mat = (I + 0.5 * dt * L).tocsr()
    f = np.exp(-(((SS - 0.4) / 0.2) ** 2)) * np.cos(TT)
    u = Q @ f.ravel()
    n0 = np.sqrt(np.sum(W * u * u))
    ts, ns_ = [0.0], [1.0]
    steps = int(tmax / dt)
    for k in range(1, steps + 1):
        u = lu.solve(rhs_mat @ u)
        if k % 10 == 0:
            ts.append(k * dt)
            ns_.append(np.sqrt(np.sum(W * u * u)) / n0)
            if ns_[-1] < stop_ratio:
                break
    ts, ns_ = np.array(ts), np.array(ns_)
    t_end = ts[-1]
    sel = (ts > 0.6 * t_end) & (ns_ > 0)
    slope = np.polyfit(ts[sel], np.log(ns_[sel]), 1)[0]
    return -slope, t_end, ns_[-1]


def main():
    ok = True
    cases = (
        (1, 1 / 3, 3 / 4, (3e-5, 1e-5, 3e-6, 1e-6), dict(ns=240, nth=32, tmax=3000.0)),
        (2, 1 / 2, 2 / 3, (1e-3, 3e-4, 1e-4, 3e-5), dict()),
    )
    for m, p_bcz, p_thm, kappas, kw in cases:
        rates = []
        for kap in kappas:
            lam, t_end, last = decay_rate(kap, m, **kw)
            rates.append(lam)
            print(f"  m={m} kappa={kap:.0e}: late-time rate = {lam:.4e}  (run to t={t_end:.0f}, norm ratio {last:.1e})", flush=True)
        local = [np.log(rates[i] / rates[i + 1]) / np.log(kappas[i] / kappas[i + 1]) for i in range(len(rates) - 1)]
        slope = np.polyfit(np.log(kappas), np.log(rates), 1)[0]
        closer_to_bcz = abs(slope - p_bcz) < abs(slope - p_thm)
        if m == 2:
            good = abs(slope - p_bcz) < 0.07 and closer_to_bcz
        else:
            # m=1 is pre-asymptotic at these kappa (layer width vs annulus): require a falling local exponent
            good = all(local[i + 1] < local[i] for i in range(len(local) - 1)) and local[-1] < 0.42 and closer_to_bcz
        ok &= good
        print(f"m={m}: overall exponent {slope:.3f}, local exponents {[round(x, 3) for x in local]}; "
              f"BCZ kappa^(m/(m+2)) = {p_bcz:.3f}; Thm 2.6 as written kappa^(1/beta) = {p_thm:.3f}; "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    lam_c, _, _ = decay_rate(1e-4, 2, ns=240, nth=64, dt=0.05)
    lam_f, _, _ = decay_rate(1e-4, 2)
    conv = abs(lam_c - lam_f) / lam_f
    print(f"resolution check (m=2, kappa=1e-4): fine {lam_c:.4e} vs base {lam_f:.4e}, rel diff {conv:.1%} "
          f"{'PASS' if conv < 0.1 else 'FAIL'}")
    ok &= conv < 0.1
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
