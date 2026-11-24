# Amendment to the true-operator preregistration (written after the fact; honest record)

`true_operator.py` was registered (hash in PREREGISTERED_TRUE_OPERATOR.sha256) before the validation V1/V2 were run. Afterwards I modified it:
`build_operator` and `full_window_operator` gained a parameter `p` (default 1), changing the diffusive face conductance from rho_f (len/dist) to
rho_f^p rho_i^(1-p) (len/dist), in order to test the five-dimensional weight (p = 3). For p = 1 the expression is algebraically identical to the registered one.

This is a change to a registered file, so the original hash record now correctly reports `true_operator.py: FAILED`. That failure is expected and is NOT hidden or overwritten.

Verification that the p = 1 behaviour is unchanged:
1. `check_true_operator.py` (pure diffusion vs separable, antisymmetry, no growing mode) passes with the modified file.
2. The registered validation `validate_true_operator.py` (whose own hash still verifies, i.e. unchanged) was rerun unchanged against the modified file:
   V1 max deviations 0.003 and 0.008 (identical to the original), V2 rates 0.37976504883241263 and 0.38090342515054787 vs the original
   0.3797650488324121 and 0.3809034251505483 (differences ~1e-15). Both criteria PASS again. Outputs: true_operator_validation_ORIGINAL.out and _rerun.out.
3. The p = 1 Gaussian-vortex rates in check_p3.py (0.1565/0.1575) equal the previously recorded 0.1575.

## Amendment 2 (2026-11-23, found by auditing my own code)

Defect: the documentation says "reflecting axis", but `full_window_operator` padded the axis side by linear extrapolation, so for a window touching rho = 0
(first cell centre dr/2) the face value of psi at the axis was O(dr^2) rather than 0 and a small advective flux leaked through the axis.
Measured on a compact flow with psi ~ rho^2 at the axis (kappa = 0, constant field): max |L 1| in the axis row 1.8 (interior 9e-4); integrated leak -3.0e-4.
Fix: if the window touches the axis (|rho[0] - dr/2| < 1e-9 dr) the ghost row is set to -psi[0], so the axis face value is exactly 0 (precondition: psi = 0 on the axis, true for
stream functions integrated from the axis). After the fix: axis row 1.2e-3 (the interior level), integrated leak -4.9e-9.
Also: `orbit_varying_rate` silently swallowed eigs failures; it now prints them (as `atmosphere_modes` does).

Effect on the registered results (ring windows DO include the axis): the registered validation rerun unchanged, outputs in true_operator_validation_rerun2_axisfix.out
(after the p amendment: true_operator_validation_rerun1_pfix.out): V1 deviations unchanged (0.003, 0.008); V2 rates 0.37976515 and 0.38090363 vs original 0.37976505 and 0.38090343
(changes ~3e-7 relative). Both criteria PASS. check_true_operator.py and check_p3.py pass. The Gaussian-vortex runs (window rho in [0.2, 1.8]) never touch the axis and are unaffected by construction.

The new hash of the modified file is in PREREGISTERED_TRUE_OPERATOR_AMENDMENT.sha256 (regenerated after Amendment 2).
