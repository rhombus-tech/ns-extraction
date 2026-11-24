"""Mechanical test of the pipeline's scalar-decay prediction on MEASURED data (velocity + dye) of a vortex with closed streamlines.

    python3 measured_vortex_test.py data.npz          # real data
    python3 measured_vortex_test.py --selftest        # synthetic data: must PASS with 5% noise, must FAIL when the velocity used is 40% too strong

data.npz keys: t (nt,), x (nx,), y (ny,) uniform grids, ux, uy (nt, nx, ny) velocity, c (nt, nx, ny) dye concentration, kappa (scalar, molecular diffusivity of the dye, same units),
               optional: translating (0/1; default 0 = vortex stationary in the lab frame).
Planar flow is mapped onto the axisymmetric solver with a very large radius rho0 = 1000 * (domain size) (weight rho^p -> uniform), so all validated machinery is reused.

PREREGISTERED CRITERIA (fixed here, before any real data is analysed; thresholds are conventions chosen from the 0.6% agreement of the 2D propagator with 3D evolving runs
and from a noise allowance, NOT tuned to data):
  G  applicability gate: |div u| rms / |grad u| rms <= 0.10 (planar incompressibility) AND a closed-streamline region of >= 200 cells exists; otherwise NOT APPLICABLE (no verdict).
  M1 measured r(t) = ||dc(t)|| / ||dc(0)|| (orbit-varying part in the closed region); r must reach <= 0.3 above the noise floor (floor = median of the smallest 10% of r), else INCONCLUSIVE.
  P  prediction: advance the MEASURED initial dye field with the 2D operator driven by the MEASURED velocity (interval-mean stream function), compute r_pred(t) the same way.
  C1 windowed rate (analyze_stage1 protocol: r in [0.03, 0.3], >= 4 points): pred/meas in [0.8, 1.25].
  C2 max |ln(r_pred / r_meas)| <= 0.25 over times with r_meas >= max(0.1, 3 * floor).
  PASS iff C1 and C2; FAIL if the data are usable (M1 satisfied) and either fails; otherwise INCONCLUSIVE.
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import expm_multiply
import io_fields as io
import orbits as ob
import true_operator as to
import analyze_stage1 as a

RHO_BIG = 1000.0


def planar_grid(x, y):
    dx, dy = float(np.diff(x).mean()), float(np.diff(y).mean())
    L = max(x[-1] - x[0], y[-1] - y[0])
    rho = RHO_BIG * L + (x - x.mean())             # large radius, uniform spacing dx
    return rho, dx, dy


def stream_planar(ux, uy, rho, dx, dy):
    """psi with u_rho = -psi_z/rho, u_z = psi_rho/rho  <->  u_x = -psi_y/rho0, u_y = psi_x/rho0 (rho0 ~ constant)."""
    psi, res = io.stream_function(ux, uy, rho, dx, dy)
    return psi, res


def gate(ux, uy, dx, dy, lab):
    div = np.gradient(ux, dx, axis=0) + np.gradient(uy, dy, axis=1)
    grad = np.sqrt(np.mean(np.gradient(ux, dx, axis=0) ** 2 + np.gradient(ux, dy, axis=1) ** 2 + np.gradient(uy, dx, axis=0) ** 2 + np.gradient(uy, dy, axis=1) ** 2))
    ratio = float(np.sqrt(np.mean(div**2)) / max(grad, 1e-300))
    region = a.region_of(lab)
    return ratio, int(region.sum()), region


def label(psi):
    """Orient psi so its minimum is the vortex core and remove the far-field level (median of the border)."""
    border = np.concatenate([psi[0], psi[-1], psi[:, 0], psi[:, -1]])
    p = psi - np.median(border)
    return p if abs(p.min()) >= abs(p.max()) else -p


def norms_of(c_frames, lab, R, dr, dz, region):
    return np.array([a.dc_norm(c, lab, R, dr, dz, region) for c in c_frames])


def predict(c0, psi_frames, times, rho, dx, dy, kappa):
    c = c0.copy()
    out = [c.copy()]
    for i in range(len(times) - 1):
        psi = 0.5 * (psi_frames[i] + psi_frames[i + 1])
        L, _ = to.full_window_operator(psi, rho, dx, dy, kappa, p=1)
        c = expm_multiply(L * (times[i + 1] - times[i]), c.ravel()).reshape(c.shape)
        out.append(c.copy())
    return out


def run(data, velocity_scale=1.0, verbose=True):
    t, x, y = data["t"], data["x"], data["y"]
    # fixed preprocessing (added after the synthetic self-test showed that grid-scale white noise alone fails the divergence gate, before any real data was seen):
    # Gaussian smoothing of 1.5 grid cells for the velocity and 1 cell for the dye, applied identically to measurement and prediction input.
    ux = gaussian_filter(data["ux"] * velocity_scale, (0, 1.5, 1.5))
    uy = gaussian_filter(data["uy"] * velocity_scale, (0, 1.5, 1.5))
    c = gaussian_filter(data["c"], (0, 1.0, 1.0))
    kappa = float(data["kappa"])
    rho, dx, dy = planar_grid(x, y)
    psi_frames = [stream_planar(ux[i], uy[i], rho, dx, dy)[0] for i in range(len(t))]
    lab0 = label(psi_frames[0])
    ratio, ncell, region = gate(ux[0], uy[0], dx, dy, lab0)
    if verbose:
        print(f"gate: |div u|/|grad u| = {ratio:.3f} (limit 0.10), closed region {ncell} cells (min 200)")
    if ratio > 0.10 or ncell < 200:
        print("RESULT: NOT APPLICABLE (gate failed); no verdict")
        return "NOT APPLICABLE"
    R, _ = np.meshgrid(rho, y, indexing="ij")
    n_meas = norms_of(c, lab0, R, dx, dy, region)
    r_meas = n_meas / n_meas[0]
    floor = float(np.median(np.sort(r_meas)[: max(1, len(r_meas) // 10)]))
    c_pred = predict(c[0], psi_frames, t, rho, dx, dy, kappa)
    n_pred = norms_of(c_pred, lab0, R, dx, dy, region)
    r_pred = n_pred / n_pred[0]
    rate_m, win_m = a.measured_rate(t, n_meas)
    rate_p, win_p = a.measured_rate(t, n_pred)
    usable = (r_meas.min() <= 0.3) and rate_m is not None
    if verbose:
        print(f"M1: r_meas reaches {r_meas.min():.3f}, noise floor {floor:.3f}; windowed rate measured {rate_m}, predicted {rate_p}")
    if not usable:
        print("RESULT: INCONCLUSIVE (measured decay does not reach the registered window)")
        return "INCONCLUSIVE"
    c1 = 0.8 <= rate_p / rate_m <= 1.25 if rate_p else False
    sel = r_meas >= max(0.1, 3 * floor)
    dev = float(np.max(np.abs(np.log(np.maximum(r_pred[sel], 1e-9) / r_meas[sel])))) if sel.any() else np.inf
    c2 = dev <= 0.25
    if verbose:
        print(f"C1 pred/meas windowed rate = {rate_p / rate_m:.3f} -> {'ok' if c1 else 'FAIL'};  C2 max|ln(r_pred/r_meas)| = {dev:.3f} (limit 0.25) -> {'ok' if c2 else 'FAIL'}")
    verdict = "PASS" if (c1 and c2) else "FAIL"
    print(f"RESULT: {verdict}")
    return verdict


# ---------------------------------------------------------------------------------------------------------- synthetic self-test
def synthetic(noise=0.05, n=96, T=14.0, nt=29, seed=0, kappa=1e-3):
    rng = np.random.default_rng(seed)
    L = 1.0
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P, A = 0.18, 0.3
    psi_pl = -P * np.exp(-(X**2 + Y**2) / (2 * A**2))                 # planar psi; u_x = -psi_y, u_y = psi_x  (ccw or cw irrelevant)
    ux, uy = -np.gradient(psi_pl, y, axis=1), np.gradient(psi_pl, x, axis=0)
    c0 = np.exp(-(((X - 0.18) ** 2 + Y**2) / (2 * 0.07**2)))
    data = dict(t=np.linspace(0, T, nt), x=x, y=y, kappa=kappa)
    rho, dx, dy = planar_grid(x, y)
    psi = stream_planar(ux, uy, rho, dx, dy)[0]
    c = predict(c0, [psi] * nt, data["t"], rho, dx, dy, kappa)
    c = np.array(c)
    scale_u = np.sqrt(np.mean(ux**2 + uy**2))
    data["ux"] = np.repeat(ux[None], nt, 0) + noise * scale_u * rng.standard_normal((nt, n, n))
    data["uy"] = np.repeat(uy[None], nt, 0) + noise * scale_u * rng.standard_normal((nt, n, n))
    data["c"] = c + noise * np.sqrt(np.mean(c[0] ** 2)) * 0.2 * rng.standard_normal(c.shape)
    return data


def selftest():
    print("SELF-TEST 1: synthetic planar Gaussian vortex, 5% noise, CORRECT model -> expect PASS")
    d = synthetic(noise=0.05)
    v1 = run(d)
    print("\nSELF-TEST 2: same data, but the velocity used by the predictor is 40% too strong -> expect FAIL")
    v2 = run(d, velocity_scale=1.4)
    ok = (v1 == "PASS") and (v2 == "FAIL")
    print(f"\nself-test {'OK' if ok else 'PROBLEM'}: test1 {v1}, test2 {v2}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    d = dict(np.load(sys.argv[1]))
    run(d)
