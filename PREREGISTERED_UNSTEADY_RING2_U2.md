# Preregistration U2: evolving ring 2 at 128^3 (written after the 96^3 result was known; hashed with unsteady_ring2_128.py before the run)

## Why this exists, and what is NOT preregistered in the strong sense
PREREGISTERED_UNSTEADY_RING2.md (U1) returned INCONCLUSIVE: the 96^3 evolving flow failed the resolution check (spectral tail 1.6e-6 vs limit 1e-6). The 96^3 descriptive values were q = 0.697 (kappa 3e-2) and 0.473 (kappa 1e-2).
U2 reruns ONLY the evolving side at 128^3. Because those 96^3 values are known, a band like U1's [0.5, 0.9] would be tuned; U2 therefore tests DIRECTION ONLY, which is the weaker claim the guide makes
("a frozen-flow rate overstates the decay of an evolving flow"). The 128^3 numbers are a genuine new measurement but the criterion was chosen with the 96^3 outcome in view, and this is stated here and in the guide.

## Run
Evolving ring 2: NS3D N = 128, nu = 0.005, RK4, dt = 0.01 (needed: explicit-RK4 scalar stability kappa*3*(N/3)^2*dt < 2.78 gives 1.64 for kappa = 3e-2), outputs every 0.5, to t = 10; same blob, same kappa {3e-2, 1e-2}, same extraction and measurement as U1.
Frozen side: the existing unsteady2_frozen.npz (96^3 velocity snapshot t = 2.5; velocity tail 4.0e-7 < 1e-6, resolved; both scalar kappa valid). It is not rerun. The 96^3 evolving file is kept as unsteady2_full_96.npz.
Rate measurement, validity (scalar tail < 1e-6 at the last output, rate defined) and velocity-resolution rule (tail < 1e-6 at every evolving output and the frozen snapshot): identical to U1.

## Criterion U2a (the only one)
PASS: flow resolved, >= 2 valid kappa, and every valid q = evolving(128^3) / frozen < 0.95.
FAIL: flow resolved, >= 2 valid kappa, any valid q >= 0.95. INCONCLUSIVE otherwise.
Reported, not criteria: the q values themselves; whether they lie in U1's band [0.5, 0.9]; the resolution sensitivity |rate(128^3) - rate(96^3)| / rate(128^3).

## Pre-declared interpretation
1. PASS: the guide says "frozen overstates the evolving decay on two resolved rings" (direction), gives the q values, and states the size is not predictable if the q values differ across rings by more than 20%.
2. FAIL: the guide says the direction did not generalise.
3. INCONCLUSIVE: the guide keeps the existing text.

## Not claimed
Mechanism; other geometries or Reynolds numbers beyond nu = 0.01 and 0.005; the slowest-mode rate; experiments.

Hash: PREREGISTERED_UNSTEADY_RING2_U2.sha256 (this file and unsteady_ring2_128.py; the files hashed in PREREGISTERED_UNSTEADY_RING2.sha256 remain unchanged).
