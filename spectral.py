"""Spectral helpers on a periodic cubic box [0,L)^3: curl, Biot-Savart, strain, vorticity direction.

Default L = 2*pi (integer wavenumbers).  Pass L explicitly for other box sizes.
"""
import numpy as np


class Grid:
    def __init__(self, n, L=2 * np.pi):
        self.n = n
        self.L = L
        h = L / n
        self.x = np.arange(n) * h
        k1 = np.fft.fftfreq(n, d=h) * (2 * np.pi)
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, k1, indexing="ij")
        self.k = np.stack([self.kx, self.ky, self.kz])
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = np.where(self.k2 == 0, 1.0, self.k2)
        self.dV = h ** 3

    def fft(self, f):
        return np.fft.fftn(f, axes=(-3, -2, -1))

    def ifft(self, fh):
        return np.fft.ifftn(fh, axes=(-3, -2, -1)).real

    def integrate(self, f):
        return float(np.sum(f) * self.dV)

    def curl(self, a):
        ah = self.fft(a)
        k = self.k
        c = np.stack([
            1j * (k[1] * ah[2] - k[2] * ah[1]),
            1j * (k[2] * ah[0] - k[0] * ah[2]),
            1j * (k[0] * ah[1] - k[1] * ah[0]),
        ])
        return self.ifft(c)

    def velocity_from_vorticity(self, w):
        wh = self.fft(w)
        k = self.k
        cr = np.stack([
            k[1] * wh[2] - k[2] * wh[1],
            k[2] * wh[0] - k[0] * wh[2],
            k[0] * wh[1] - k[1] * wh[0],
        ])
        uh = 1j * cr / self.k2_safe
        uh[:, self.k2 == 0] = 0
        return self.ifft(uh)

    def grad_tensor(self, u):
        """G[i,j] = d_j u_i."""
        uh = self.fft(u)
        g = np.empty((3, 3) + u.shape[1:])
        for i in range(3):
            for j in range(3):
                g[i, j] = self.ifft(1j * self.k[j] * uh[i])
        return g

    def strain_from_vorticity(self, w):
        g = self.grad_tensor(self.velocity_from_vorticity(w))
        return 0.5 * (g + np.swapaxes(g, 0, 1))

    def divergence(self, u):
        uh = self.fft(u)
        return self.ifft(1j * (self.k[0] * uh[0] + self.k[1] * uh[1] + self.k[2] * uh[2]))

    def grad_scalar(self, f):
        fh = self.fft(f)
        return np.stack([self.ifft(1j * self.k[i] * fh) for i in range(3)])


def upsample(a, m):
    """Spectral (zero-padding) upsampling of the last three axes from n^3 to m^3; Nyquist modes dropped."""
    n = a.shape[-1]
    ah = np.fft.fftn(a, axes=(-3, -2, -1))
    dst = np.ix_(*[np.r_[0:n // 2, m - n // 2 + 1:m] for _ in range(3)])
    src = np.ix_(*[np.r_[0:n // 2, n // 2 + 1:n] for _ in range(3)])
    out = np.zeros(a.shape[:-3] + (m, m, m), dtype=complex)
    for c in np.ndindex(a.shape[:-3]):
        o = np.zeros((m, m, m), dtype=complex)
        o[dst] = ah[c][src]
        out[c] = o
    return np.fft.ifftn(out, axes=(-3, -2, -1)).real * (m / n) ** 3


def random_vorticity(grid, rng, kmax=2.0):
    """Random real divergence-free vorticity: curl of a band-limited real vector potential."""
    a = rng.standard_normal((3,) + (grid.n,) * 3)
    ah = grid.fft(a)
    ah[:, np.sqrt(grid.k2) > kmax] = 0
    ah[:, grid.k2 == 0] = 0
    a = grid.ifft(ah)
    return grid.curl(a)
