"""First analysis of MEASURED 3D velocity data: a vortex ring (Cambridge repository 1810/262984, Ri2d4_Data, CC BY 4.0), frame 450.

The data are not mine and no quantity here was tuned to them. What is and is not tested:
  - tested: whether the extraction/pipeline honesty metrics behave sensibly on measured, bounded, anisotropically sampled data;
            ring axis found from the data vs the dataset authors' own script (xbar, ybar = 0.07525, 0.068);
  - NOT tested: any decay rate (there is no tracer measurement here; the 'Scalar' variables are checked below and are not a tracer).
Ring speed: the dataset gives no frame interval, so the speed is ESTIMATED as the enstrophy-weighted mean axial velocity at the vortex core
(a vortex core moves with the ring) -- an assumption, stated, not a measurement.
"""
import sys
import numpy as np
from scipy.io import netcdf_file
import fd3d
import io_fields as io

NC = "external/Ri2d4_Data/VisitC_enst_0450.nc"


def load():
    f = netcdf_file(NC, "r", mmap=False)
    x, y, z = [np.array(f.variables[k].data, float) for k in "xyz"]
    u3 = np.stack([np.array(f.variables[k].data, float).transpose(2, 1, 0) for k in "uvw"])  # (3, x, y, z)
    sc = {k: np.array(f.variables[k].data, float).transpose(2, 1, 0) for k in ("Scalar", "Scalar2", "Scalar3")}
    h = (float(np.diff(x).mean()), float(np.diff(y).mean()), float(np.diff(z).mean()))
    return x, y, z, u3, sc, h


def main():
    x, y, z, u3, sc, h = load()
    w3 = fd3d.curl(u3, h, False)
    en = np.sum(w3 * w3, axis=0)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = [float((en * a).sum() / en.sum()) for a in (X, Y, Z)]
    print(f"ring axis from enstrophy centroid: ({cx:.5f}, {cy:.5f}) m; authors' script uses (0.07525, 0.068); ring z = {cz:.4f} m")
    names = ("wx", "wy", "wz")
    for k, a in sc.items():
        cors = [abs(np.corrcoef(a.ravel(), w3[i].ravel())[0, 1]) for i in range(3)]
        print(f"  {k}: |corr| with my vorticity components {dict(zip(names, np.round(cors, 3)))}")
    a = io.sample_axisymmetric(u3, h, (cx, cy), rho_max=0.045, z_half=0.025, nrho=90, nz=100, nphi=24, w=w3, z_center=cz, mode="nearest")
    print(f"\nextraction about the ring axis: nonaxisym = {a['nonaxisym']:.3f}, swirl = {a['swirl']:.3f}")
    R, _ = np.meshgrid(a["rho"], a["z"], indexing="ij")
    wphi = a["omega_phi"]
    core = np.unravel_index(np.argmax(np.abs(wphi)), wphi.shape)
    wt = np.where(np.abs(wphi) > 0.5 * np.abs(wphi).max(), np.abs(wphi), 0.0)
    U = float((wt * a["u_z"]).sum() / wt.sum())
    print(f"core of azimuthal vorticity at rho = {a['rho'][core[0]]*1e3:.1f} mm, |omega_phi| = {abs(wphi[core]):.1f} 1/s; estimated ring speed U = {U*1e3:.1f} mm/s (assumption: core moves with the ring)")
    psi, res = io.stream_function(a["u_rho"], a["u_z"], a["rho"], a["dr"], a["dz"])
    print(f"stream-function consistency residual = {res:.3f}")
    ok, reasons = io.applicability(a, res)
    if not ok:
        print("\nAPPLICABILITY GATE: FAILED -- the axisymmetric pipeline does not apply to this flow, so no shear, core or rate numbers are reported.")
        for r_ in reasons:
            print("   -", r_)
        return 0
    pco = io.to_comoving(psi, a["rho"], U)
    lab = (-1.0 if pco[core] > 0 else 1.0) * pco
    d = __import__("diagnose").diagnose_axisymmetric(lab, R, a["dr"], a["dz"], omega_phi=wphi, level=0.02 * lab.min(), kappa=1e-6)
    print(f"closed-streamline region: {d['closed_cells']} cells; ||dOmega||/||Omega|| = {d['rel_residual']:.3f}")
    print(f"orbit-frequency shear: confidence {d['shear_confidence']} (SNR {d['shear_snr']:.0f}), zeros {d['zeros']}")
    cm = d.get("core_mode")
    if isinstance(cm, dict):
        print(f"core: Hessian {cm['hessian_eigenvalues'][0]:.3g}, {cm['hessian_eigenvalues'][1]:.3g} (ellipticity {cm['ellipticity']:.2f}); core shear {cm['core_shear']:.3g}; core frequency {cm['core_frequency']:.3g} 1/s")
        print(f"closed-form core-mode rate for a diffusivity of water-like nu = 1e-6 m^2/s: {cm['rate']:.3e} 1/s (a PREDICTION about this flow; not validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
