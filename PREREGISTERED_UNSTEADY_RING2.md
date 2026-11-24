# Preregistration: does a frozen-flow decay rate overstate an evolving flow's decay on a SECOND ring? (written before any run; hashed with unsteady_ring2.py)

## Background (what is already known, disclosed)
Ring 1 (nu = 0.01, 96^3): measured windowed decay rate of the orbit-varying passive scalar was 0.2777 (frozen) vs 0.1871 (evolving) at kappa = 3e-2, and 0.3878 vs 0.2811 at kappa = 1e-2,
i.e. evolving/frozen q = 0.674 and 0.725 (Stage 1; analyze_stage1.py). The guide and paper say this on one flow. This test is OUT OF SAMPLE for ring 2 (nu = 0.005), whose scalar decay has never been computed.
A prior in-sample exploration (explore_adiabatic.py, not preregistered) found that the instantaneous closed-form core rate does NOT explain the evolving decay (its adiabatic integral predicted r = 0.001, measured 0.17),
so no mechanism is being tested here; only the empirical statement that q < 1.

## Flows
Ring 2: sim_ns3d ring, N = 96, nu = 0.005, snapshots data_nu005/ring_000..006.npz (already generated; same initial condition as ring 1).
- FROZEN: velocity of snapshot 5 (t = 2.5) in the ring frame, speed from centroid shift between snapshots 4 and 6, exactly as scalar_experiment.run_frozen. Run to t = 24.
- EVOLVING: scalar carried by the evolving ring from t = 0 (NS3D N = 96, nu = 0.005, RK4 dt = 0.02), exactly as scalar_experiment.run_full. Run to t = 10.
Scalar: same blob as Stage 1 (rho = 1, z = z_c + 0.25 (frozen) / pi + 0.25 (evolving), sigma = 0.2); kappa in {3e-2, 1e-2} (the two resolved ones in Stage 1); outputs every 0.5.

## Measurement (identical to Stage 1, analyze_stage1.py; no discretion)
r(t) = ||delta c(t)|| / ||delta c(0)|| in the closed region (psi < 0.02 psi_min, orbit decomposition 160 bins). Rate = -slope of ln ||delta c|| over outputs with 0.03 <= r <= 0.3, requiring >= 4 outputs.
A (flow, kappa) is VALID iff the scalar spectral tail fraction at the last output is < 1e-6 and the rate is defined. q(kappa) = evolving rate / frozen rate.
The flow is RESOLVED iff the velocity spectral tail (energy fraction at |k| > 0.75 n/3) is < 1e-6 for the frozen snapshot and at every evolving output.

## Criterion U1 (the only one)
- PASS: flow resolved, >= 2 valid kappa, and EVERY valid q lies in [0.5, 0.9].
- FAIL: flow resolved, >= 2 valid kappa, and ANY valid q outside [0.5, 0.9].
- INCONCLUSIVE: otherwise (fewer than 2 valid kappa, or flow not resolved).
The band [0.5, 0.9] is centred on the ring-1 values (0.674, 0.725) with generous width; it is chosen with ring 1 in view and cannot be changed after the run.

## Pre-declared interpretation
1. PASS: "a frozen-flow rate overstates this evolving ring's decay by 10-100%" holds on two rings (still one geometry, two Reynolds numbers); the guide says so and still advises against a frozen rate for an evolving flow.
2. FAIL: the ring-1 ratio does not generalise; the guide reports both rings and says the ratio is flow-specific.
3. INCONCLUSIVE: the guide keeps the single-ring statement and says it is untested elsewhere.

## Not claimed
Anything about the mechanism (the adiabatic closed-form explanation already failed in-sample); other geometries or Reynolds numbers; the slowest-mode rate (this is the windowed transient decay of ||delta c||); experiments.

## Code
unsteady_ring2.py (frozen / full / analyze), reusing analyze_stage1.py, scalar_experiment.py, io_fields.py, orbits.py, sim_ns3d.py unchanged. The analysis function was checked to reproduce the ring-1 q values (0.674, 0.725) before registration.
Hash: PREREGISTERED_UNSTEADY_RING2.sha256 (this file, unsteady_ring2.py, analyze_stage1.py, scalar_experiment.py, io_fields.py, orbits.py, sim_ns3d.py).
