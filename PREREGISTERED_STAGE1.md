# Stage 1 preregistration: does the orbit-reduced rate estimate predict passive-scalar homogenisation in a real vortex-ring flow?

Written 2026-11-23, BEFORE any scalar run. Hash of this file recorded in PREREGISTERED_STAGE1.sha256. Do not edit after hashing.

## Question
`rates.reduced_rate` estimates the decay rate of the orbit-varying part of an advected quantity from the streamline
geometry. Does that estimate predict how fast a passive scalar actually homogenises along streamlines in a real
(simulated) vortex-ring flow, (F) with the velocity frozen and (S) with the full evolving flow?

## Flow (fixed in advance)
Viscous vortex ring from `sim_ns3d.py ring 96 0.01 T`: periodic 96^3, nu = 0.01, Gamma = 3, R0 = 1, core a = 0.35.
Frozen flow (F): the t = 2.5 snapshot velocity of the existing run, in the frame moving with the ring (u - U z_hat, U = ring speed
at that snapshot from centroid shift), fixed in time. Evolving flow (S): the same simulation continued to t = 10.

## Scalar (fixed in advance)
Passive scalar c with diffusivity kappa in K = {3e-2, 1e-2, 3.3e-3, 1.1e-3} (Schmidt numbers 1/3, 1, 3, 9).
Initial condition (axisymmetric about the ring axis, orbit-varying): c0 = exp(-((rho-1.0)^2 + (z-z_c-0.25)^2) / (2*0.2^2)),
z_c = ring centroid at the start. Pseudo-spectral, 2/3 dealiasing, RK4, dt = 0.02, output every 0.5 time units.

## Measurement (fixed in advance)
At each output time, sample c on the (rho,z) half-plane (phi-average), compute the stream function in the ring frame
(ring frame at that time for S, the fixed t = 2.5 frame for F), closed region = connected set {psi < 0.02 psi_min} around the core.
delta_c = c - orbit average (orbit average with weight rho d rho dz per psi-shell, 160 bins). Norm = rho-weighted L2 over
the closed region. Let r(t) = ||delta_c(t)|| / ||delta_c(t0)||, t0 = first output.
**Measured rate** = -slope of ln ||delta_c|| versus t, least squares over output times with r in [0.03, 0.3].
A run is **excluded** (rate undefined) if fewer than 4 output times fall in that window.

## Resolution rule (fixed in advance)
A (kappa, flow) run is **valid** only if, at the last output time, the fraction of scalar variance in modes with
|k| > 0.75 * (n/3) is below 1e-6. Invalid runs are reported but do not count for or against any criterion.

## Prediction (fixed in advance)
`rates.reduced_rate(psi, rho, dr, dz, region, kappa, p=1, k_modes=(1,), bc=...)` for bc in {dirichlet, neumann}, using the
same psi and region as the measurement: for F at the fixed t = 2.5 frame; for S at the snapshot nearest the midpoint of the
fit window. Point prediction = geometric mean of the two bc values.

## Criteria
- **F1 (frozen, rate).** PASS if >= 2 valid kappa and for every valid kappa measured / predicted lies in [0.5, 2]. FAIL if any valid kappa lies outside. Otherwise INCONCLUSIVE.
- **F2 (frozen, scaling).** If >= 3 valid kappa: PASS if |slope_measured - slope_predicted| <= 0.15, slopes = d ln(rate) / d ln(kappa) by least squares over valid kappa. Else INCONCLUSIVE.
- **S1 (full flow, rate).** Same as F1 on the evolving flow.

## Pre-declared interpretation
1. F1 FAIL: the rate estimate is falsified on the real ring flow; it is removed from the guide and the paper's text is changed to say so.
2. F1 PASS and S1 FAIL: valid for frozen / quasi-steady flow only; the guide states that limit and the discrepancy is attributed to flow evolution (not tested further here).
3. F1 PASS and S1 PASS: supported on this flow family only (one Reynolds number, one geometry). Not a general validation.
4. Any INCONCLUSIVE is reported as inconclusive, not as a pass.

## Not claimed by this test
No claim about turbulent flows, other Reynolds numbers, wall-bounded or swirling flows, the vorticity (p = 3) version, the
exponent m/(m+2) on real data, or any experiment.
