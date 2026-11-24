# ABANDONED, NEVER HASHED OR RUN. Superseded by a ground-truth eigenvalue comparison (compare_real_flow.py) which showed the estimator itself under-predicts; retesting it would not have been informative.
# Stage 1 extension E2 (drafted, not registered): late-time exponential rate on a new frozen flow

Written after E1 FAILED. The E1 diagnosis (post hoc) was that the registered measurement window (r in [0.03, 0.3]) sat on the
fast shear-driven transient, whereas the estimate predicts the ASYMPTOTIC slowest rate. E2 re-tests the estimate with a measurement
that targets the asymptotic exponential decay, on a flow whose late-time behaviour has NOT been inspected (out-of-sample).
Hash in PREREGISTERED_STAGE1_E2.sha256. Nothing below may be changed after hashing.

## Flow (new, not previously analysed)
Frozen ring-frame velocity from the t = 4.0 snapshot (index 8) of the existing ring run (data/ring_008.npz), in the frame moving with
the ring (speed U from centroid shift between snapshots 7 and 9; ring centroid z_c from snapshot 8), spectrally upsampled 96^3 -> 128^3.

## Scalar (same procedure as E1, new kappa set)
K = {2e-2, 1e-2, 5e-3, 3.3e-3}. Same initial blob (centred rho = 1.0, z = z_c + 0.25, sigma = 0.2), pseudo-spectral RK4, dt = 0.02,
output every 0.5, run to t = 30. Same (rho,z) sampling, closed region, orbit decomposition, norm.
Explicit-RK4 stability at 128^3 requires kappa * 3 * (128/3)^2 * dt < 2.78, i.e. kappa <= 0.0255; all K satisfy it.

## Measurement of the asymptotic rate (fixed in advance)
r(t) = ||delta_c(t)|| / ||delta_c(0)||. Local rate s(t_i) = -(least-squares slope of ln r over the 7 consecutive outputs centred at t_i)
(3 time units), defined where all r > 0. t_peak = the t_i maximising s(t_i) among t_i with r(t_i) >= 0.03.
Window = [t_peak + 3, min(t_peak + 9, last output time with r >= 0.01)]. The asymptotic rate is defined ONLY IF the window holds
>= 8 outputs, r >= 0.01 throughout, and the linear fit of ln r on t over the window has R^2 >= 0.9; the rate is minus its slope.
Otherwise the kappa is **excluded (rate undefined)**.

## Resolution rule (same)
A kappa is valid iff, at the last output, the variance fraction in |k| > 0.75*(n/3), n = 128, is < 1e-6, AND the rate is defined.

## Prediction (same procedure as Stage 1/E1)
`rates.reduced_rate(psi, rho, dr, dz, region, kappa, p=1, k_modes=(1,), bc in {dirichlet, neumann})` with the same psi and region as
the measurement (frozen t = 4.0 ring frame); point prediction = geometric mean of the two bc values.

## Criteria
- **F1''**: PASS if >= 2 valid kappa and every measured/predicted ratio is in [0.5, 2]; FAIL if any valid ratio is outside; else INCONCLUSIVE.
- **F2''**: if >= 3 valid kappa: PASS iff |slope_measured - slope_predicted| <= 0.15 (slopes of ln rate vs ln kappa); else INCONCLUSIVE.

## Pre-declared interpretation
1. F1'' PASS (F2'' PASS or INCONCLUSIVE): the estimate is reinstated in the guide as "within a factor 2 of the asymptotic late-time rate on frozen ring flows at Pe ~ 100-300", still labelled experimental; the E1 failure is attributed to the measurement window and the guide says so, with both results shown.
2. F1'' FAIL: the estimate has failed twice and stays removed; rates.py is kept only as a documented negative result.
3. F1'' INCONCLUSIVE: the estimate stays withdrawn, the E1 diagnosis stays a hypothesis, and the guide says so.
4. F2'' FAIL (any F1'' outcome other than FAIL): the estimate's kappa-dependence is not supported; it stays withdrawn.

## Not claimed
Nothing about large Peclet number, the evolving flow, other flows or Reynolds numbers, vorticity (p = 3), or experiments.
