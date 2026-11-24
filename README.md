# ns-extraction

Mixing-rate tools extracted from the Navier–Stokes regularity program.

Given a velocity field (CFD, PIV, or analytic), these tools compute:
- Enstrophy balance (production, dissipation, unsteadiness, boundary flux, forcing)
- Orbit decomposition and shear profile for flows with closed streamlines
- Enhanced dissipation rate and decay curve via the 2D advection–diffusion operator
- Consistency checks for solvers and learned models

## Quick start

```
python3 diagnose.py demo
```

For your own data:

```python
import io_fields, diagnose

psi, rho, dr, dz = io_fields.sample_axisymmetric(your_data)
result = diagnose.diagnose_axisymmetric(psi, rho, dr, dz)
```

For measured velocity + dye data:

```
python3 measured_vortex_test.py data.npz
python3 measured_vortex_test.py --selftest
```

See the [Practitioner Guide](https://doi.org/10.5281/zenodo.24498505) for which tool to use for which problem.

## Requirements

Python 3.9+, NumPy, SciPy.

## Simulation data

The preregistered tests require simulation data files (~430 MB) not included in this repository.
Download them from the [Zenodo archive](https://doi.org/10.5281/zenodo.24498505) and unpack into `data/` and `data_nu005/`.

## Structure

| File | Purpose |
|------|---------|
| `diagnose.py` | Main entry point: orbit decomposition, shear, rate exponent |
| `io_fields.py` | Data adapters (npz, HDF5, CSV, VTK), stream function, co-moving frame |
| `orbits.py` | Orbit averaging and decomposition |
| `true_operator.py` | 2D advection–diffusion operator |
| `true_rates.py` | Decay rate and curve from the 2D operator |
| `fd3d.py` | 3D finite-difference enstrophy diagnostics |
| `spectral.py` | Spectral 3D diagnostics |
| `measured_vortex_test.py` | Preregistered test on measured velocity + dye |
| `sim_ns3d.py` | Pseudo-spectral Navier–Stokes solver (generates test data) |
| `time_dep_2d.py` | Time-dependent 2D propagator for evolving flows |
| `mv2003_test/` | Published-data comparison with Meunier & Villermaux (2003) |
| `PREREGISTERED_*.md` | Preregistered criteria (hashed before any comparison) |

## Published-data comparison

The `mv2003_test/` directory contains a preregistered comparison of the 2D operator
against measured dye data from Meunier & Villermaux, "How vortices mix", J. Fluid Mech. 476 (2003).
Three of four criteria pass; the late-time slope criterion fails by 0.011 past its tolerance.
See `mv2003_test/RESULTS_MV2003.md` for the full verdict and `mv2003_test/AUDIT_MV2003.md` for the post-hoc audit.

## Citation

T. Zisckind, "The Millennium Extraction: mixing-rate tools from the Navier–Stokes regularity program", 2026.

## License

MIT
