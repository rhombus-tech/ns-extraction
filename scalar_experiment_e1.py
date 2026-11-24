"""Stage 1 extension E1 (PREREGISTERED_STAGE1_E1.md): frozen-flow scalar run at 128^3. Writes stage1_e1_frozen.npz."""
import time
import numpy as np
import io_fields as io
import run_ring_case as rc
import scalar_experiment as se
from spectral import upsample

N = 128


def main(t_end=24.0):
    ref = np.load("stage1_frozen.npz")
    speed, zc5 = float(ref["speed"]), float(ref["zc"])
    u5, _, t5, _, h96 = rc.load(5)
    uf = u5.copy()
    uf[2] -= speed
    uf = upsample(uf, N)
    h = 2 * np.pi / N
    print(f"E1: upsampled frozen flow to {N}^3, U={speed:.4f}, zc={zc5:.4f}", flush=True)

    x = np.arange(N) * h
    sc = se.Scalar(N)
    chs = [np.fft.rfftn(se.blob(x, zc5)) * sc.dealias for _ in se.KAPPAS]
    steps = int(round(t_end / se.DT))
    times, slices, tails = [], [], []
    t0 = time.time()
    for s in range(steps + 1):
        if s % se.OUT_EVERY == 0:
            row, trow = [], []
            for ch in chs:
                c = np.fft.irfftn(ch, s=(N,) * 3)
                row.append(io.sample_axisym_scalar(c, h, se.CENTER, se.RHO_MAX, se.Z_HALF, z_center=zc5, **se.SL))
                trow.append(sc.tail_fraction(c))
            times.append(s * se.DT)
            slices.append(row)
            tails.append(trow)
            if (s // se.OUT_EVERY) % 6 == 0:
                print(f"  t={s*se.DT:5.1f} wall {time.time()-t0:5.0f}s tails {['%.1e' % v for v in trow]}", flush=True)
        if s == steps:
            break
        new = []
        for ch, kap in zip(chs, se.KAPPAS):
            k1 = sc.rhs(uf, ch, kap)
            k2 = sc.rhs(uf, ch + 0.5 * se.DT * k1, kap)
            k3 = sc.rhs(uf, ch + 0.5 * se.DT * k2, kap)
            k4 = sc.rhs(uf, ch + se.DT * k3, kap)
            new.append(ch + se.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        chs = new
    np.savez_compressed("stage1_e1_frozen.npz", times=np.array(times), kappas=np.array(se.KAPPAS), slices=np.array(slices),
                        tails=np.array(tails), psi=ref["psi"], rho=ref["rho"], z=ref["z"], dr=ref["dr"], dz=ref["dz"],
                        speed=speed, zc=zc5)
    print("saved stage1_e1_frozen.npz", flush=True)


if __name__ == "__main__":
    main()
