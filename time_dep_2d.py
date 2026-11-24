"""Time-dependent 2D axisymmetric propagator for a passive scalar in an EVOLVING ring, driven by the stored snapshots of the 3D run.

For each output interval [t_i, t_{i+1}] (0.5 time units) the scalar is advanced with the 2D axisymmetric advection-diffusion operator (rho^-1 div(rho grad), p = 1; true_operator.full_window_operator)
built from the interval-mean stream function in the frame translating with the ring (speed U_i from consecutive ring centroids), then re-centred on the ring (the frame speed differs slightly
from the centroid motion).  Validation: the result must reproduce the 3D evolving scalar's orbit-varying norm r(t) and windowed rate (analyze_stage1 protocol).
Also provides frozen-at-snapshot propagation (the same operator for all time) for ablations.
"""
import sys
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import expm_multiply
import io_fields as io
import true_operator as to
import analyze_stage1 as a

DT_OUT = 0.5
SIGMA = 0.2


def speeds_of(d):
    n = len(d["times"])
    dz = float(d["dz"])
    axes = [{"omega_phi": d["omega_phi"][i], "z": d["zc"][i] - a.Z_HALF + (np.arange(d["u_z"].shape[2]) + 0.5) * dz} for i in range(n)]
    sp_ = [io.ring_speed(axes[i - 1], d["times"][i - 1], axes[i], d["times"][i]) for i in range(1, n)]
    return [sp_[0]] + sp_


def psi_comoving(d, i, speed):
    psi, _ = io.stream_function(d["u_rho"][i], d["u_z"][i], d["rho"], float(d["dr"]), float(d["dz"]))
    return io.to_comoving(psi, d["rho"], speed)


def blob_initial(d):
    """The scalar blob of scalar_experiment.blob in ring-centred (rho, z') coordinates: centre rho = 1, z' = +0.25, sigma 0.2."""
    rho = d["rho"]
    dz = float(d["dz"])
    nz = d["u_z"].shape[2]
    zp = -a.Z_HALF + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(rho, zp, indexing="ij")
    return np.exp(-(((R - 1.0) ** 2 + (Z - 0.25) ** 2) / (2 * SIGMA**2)))


def evolve(d, kappa, c0, n_int=None, spd=None):
    """Advance c0 through the evolving flow; returns the list of fields at every output time."""
    rho, dr, dz = d["rho"], float(d["dr"]), float(d["dz"])
    spd = speeds_of(d) if spd is None else spd
    n = len(d["times"]) if n_int is None else n_int + 1
    c = c0.copy()
    out = [c.copy()]
    for i in range(n - 1):
        U = spd[i + 1]
        psi = 0.5 * (psi_comoving(d, i, U) + psi_comoving(d, i + 1, U))
        L, _ = to.full_window_operator(psi, rho, dr, dz, kappa, p=1)
        c = expm_multiply(L * DT_OUT, c.ravel()).reshape(c.shape)
        s = (d["zc"][i + 1] - d["zc"][i]) - U * DT_OUT           # frame speed minus centroid motion
        c = ndimage.shift(c, (0.0, -s / dz), order=3, mode="constant", cval=0.0)
        out.append(c.copy())
    return out


def frozen_series(d, kappa, i_snap, t_end=24.0, spd=None):
    """Frozen flow = snapshot i_snap in its comoving frame, blob initial condition, for the same output times as the 3D frozen run."""
    rho, dr, dz = d["rho"], float(d["dr"]), float(d["dz"])
    spd = speeds_of(d) if spd is None else spd
    psi = psi_comoving(d, i_snap, spd[max(i_snap, 1)])
    L, _ = to.full_window_operator(psi, rho, dr, dz, kappa, p=1)
    num = int(round(t_end / DT_OUT)) + 1
    frames = expm_multiply(L, blob_initial(d).ravel(), start=0.0, stop=t_end, num=num, endpoint=True)
    return np.linspace(0.0, t_end, num), [f.reshape(blob_initial(d).shape) for f in frames], psi


def norms_evolving(d, fields, spd):
    out = []
    for i, c in enumerate(fields):
        lab, R, dr, dz = a.full_labels(d, i, spd)
        out.append(a.dc_norm(c, lab, R, dr, dz, a.region_of(lab)))
    return np.array(out)


def lab_from_psi(psi, rho):
    core = np.unravel_index(np.argmin(np.where(np.ones_like(psi, bool), psi, np.inf)), psi.shape)
    return psi


def main():
    d = np.load("stage1_full.npz")
    spd = speeds_of(d)
    c0_an = blob_initial(d)
    print(f"initial blob: analytic vs stored slice, max abs diff {np.abs(c0_an - d['slices'][0, 0]).max():.2e} (stored max {d['slices'][0,0].max():.3f})")
    times = d["times"]
    for j in (0, 1):
        kap = float(d["kappas"][j])
        fields = evolve(d, kap, d["slices"][0, j], spd=spd)
        n2 = norms_evolving(d, fields, spd)
        n3 = np.array([a.dc_norm(d["slices"][i, j], *a.full_labels(d, i, spd)[:1], *a.full_labels(d, i, spd)[1:], a.region_of(a.full_labels(d, i, spd)[0])) for i in range(len(times))])
        r2, r3 = n2 / n2[0], n3 / n3[0]
        rate2, w2 = a.measured_rate(times, n2)
        rate3, w3 = a.measured_rate(times, n3)
        print(f"\nkappa={kap:g}:  3D rate {rate3:.4f} window {w3} | 2D rate {rate2:.4f} window {w2} | ratio 2D/3D {rate2 / rate3:.3f}")
        print("   t   : " + " ".join(f"{t:5.1f}" for t in times[::2]))
        print("   r_3D: " + " ".join(f"{x:5.3f}" for x in r3[::2]))
        print("   r_2D: " + " ".join(f"{x:5.3f}" for x in r2[::2]))
        print(f"   max |r_2D - r_3D| = {np.max(np.abs(r2 - r3)):.3f}; relative field error at t=5: {np.linalg.norm(fields[10] - d['slices'][10, j]) / np.linalg.norm(d['slices'][10, j]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
