# Stage 1 extension E1 (preregistered before running): frozen flow at 128^3

Written after the Stage 1 result (F1 PASS, S1 PASS, F2 INCONCLUSIVE with only two resolved kappa). Not part of the original
preregistration; it is a follow-up whose criteria are fixed here before it is run. Hash in PREREGISTERED_STAGE1_E1.sha256.

## Purpose
Make the kappa-scaling criterion decidable by resolving more diffusivities. At 96^3 the two smaller kappa failed the resolution rule.

## Setup (identical to Stage 1 frozen flow except resolution)
Same frozen ring-frame velocity (t = 2.5 snapshot, speed U and centroid z_c from stage1_frozen.npz), spectrally upsampled
from 96^3 to 128^3 (zero padding; same physical flow). Same kappa set K = {3e-2, 1e-2, 3.3e-3, 1.1e-3}, same initial blob,
same pseudo-spectral RK4, dt = 0.02, output every 0.5 up to t = 24, same (rho,z) sampling grid, same closed region and
orbit decomposition, same measured-rate window (r in [0.03, 0.3], >= 4 points), same prediction procedure
(`rates.reduced_rate`, p = 1, k = 1, Dirichlet and Neumann, geometric mean) using the same psi.
Resolution rule: a kappa is valid iff at the last output the variance fraction in |k| > 0.75 * (n/3) is < 1e-6, with n = 128.

## Criteria (same definitions as Stage 1)
- **F1'**: PASS if >= 2 valid kappa and every valid measured/predicted ratio is in [0.5, 2]; FAIL if any valid ratio is outside; else INCONCLUSIVE.
- **F2'**: if >= 3 valid kappa: PASS iff |slope_measured - slope_predicted| <= 0.15 (slopes of ln rate vs ln kappa over valid kappa); else INCONCLUSIVE.

## Pre-declared interpretation
1. F2' FAIL: the estimate's kappa-dependence is not supported on the real ring flow; the guide says the estimate gives only an order-of-magnitude figure at Pe ~ 100-300 and that its scaling with kappa is not reproduced.
2. F1' FAIL (with any F2' outcome): the rate estimate is falsified on the frozen real flow and is removed from the guide.
3. F1' PASS and F2' PASS: the scaling is supported on the frozen ring flow (one flow, one Reynolds number).
4. Anything INCONCLUSIVE is reported as inconclusive. The kappa = 3e-2 and 1e-2 results at 128^3 are reported next to their 96^3 values as a resolution-sensitivity check; they do not replace the Stage 1 verdicts.

## Not claimed
Nothing about large Peclet number beyond the smallest valid kappa, the evolving flow, other flows, or experiments.
