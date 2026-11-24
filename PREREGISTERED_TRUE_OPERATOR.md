# Preregistration: is the 2D ground-truth operator a trustworthy rate tool?

Written before running validate_true_operator.py. Hash in PREREGISTERED_TRUE_OPERATOR.sha256. Do not edit after hashing.

Context: compare_real_flow.py (exploratory) found that the true slowest atmosphere eigenmode of the 2D finite-volume operator on the real
frozen ring flow decays at 0.381 (kappa = 3.3e-3) while the orbit-reduced estimator gave 0.13-0.27. Before shipping the 2D solver as the
way to obtain a decay rate, test that it is a faithful model of the independent 3D pseudo-spectral scalar simulation (stage1_e1_frozen.npz)
and that its eigenvalue is converged.

## V1: the 2D solver reproduces the independent 3D scalar evolution
Same frozen ring-frame flow, same initial blob. 2D: `true_operator.full_window_operator` on the (240 x 480) refined grid,
time evolution by `scipy.sparse.linalg.expm_multiply`, outputs t = 0, 0.5, ..., 12. 3D: stage1_e1_frozen.npz (128^3 pseudo-spectral).
For each, r(t) = ||delta_c(t)|| / ||delta_c(0)|| with delta_c the orbit-varying part in the closed region (same decomposition, same region criterion).
For kappa in {1e-2, 3.3e-3}: let T = the outputs in [0, 12] where r_3D(t) >= 0.05. Deviation = max over T of |r_2D(t) - r_3D(t)| / r_3D(t).
**PASS** iff the deviation is <= 0.20 for BOTH kappa. Otherwise FAIL.

## V2: eigenvalue converged in resolution
At kappa = 3.3e-3, slowest atmosphere eigenvalue (|Im| >= 0.3 * min orbit frequency, >= 90% of |v|^2 inside the closed region, 6 shifts)
at refinement 1.5 (180 x 360) and 2 (240 x 480). **PASS** iff the two rates differ by <= 10% (relative to the finer one) and both exist.

## Interpretation (declared in advance)
1. V1 PASS and V2 PASS: the 2D solver is shipped as an experimental rate tool for frozen closed-streamline flows, documented with exactly these validation numbers, and the guide says what it is and is not (one flow, one Reynolds number, frozen flow).
2. V1 FAIL: the 2D solver is not validated; it is not shipped as a rate tool and the guide reports that neither route produced a trustworthy rate.
3. V1 PASS, V2 FAIL: shipped only with an explicit resolution warning requiring a convergence check by the user, reported as such.

## Not claimed
Anything beyond the frozen t = 2.5 ring flow at the stated kappa; large Peclet number; the evolving flow; other flows; experiments.
