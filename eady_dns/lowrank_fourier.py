"""
Low-rank random functions in 2D/3D.

These functions sample 2D and 3D Gaussian random functions with 
unit periodicity on arbitrary direct-product grids.

The functions are reproducibly constructed using random low-rank
components with uniformly distributed Fourier amplitudes up to
a specified cutoff frequency.

"""


import numpy as np
from scipy.linalg.blas import daxpy


def rand_fourier_series_1d(x, kmax, density=1, rand=None):
    """
    Parameters
    ----------
    x : array-like
        Arry of positions (arbitrary dimension)
    kmax : int
        Maximum wavenumber (integer form)
    density : float (default: 1)
        Fraction of modes up to cutoff to populate
    rand : np.random.RandomState or None (default: None)
        Random state, needed for determinism
        
    Returns
    -------
    f : array-like
        Values of the random function F(x)
    """ 
    if rand is None:
        rand = np.random.RandomState()
    flags = rand.rand(kmax+1)
    n = 0
    f = np.zeros_like(x)
    kx = np.zeros_like(x)
    g = np.zeros_like(x)
    for k, flag in enumerate(flags):
        if flag < density:
            a, b = rand.randn(2)
            np.multiply((2*np.pi*k), x, out=kx)
            np.cos(kx, out=g)
            daxpy(g, f, a=a)
            np.sin(kx, out=g)
            daxpy(g, f, a=b)
            n += 1
    f /= n**0.5
    return f


def rand_fourier_series_2d(x, y, kmax, density=1, rand=None):
    if rand is None:
        rand = np.random.RandomState()
    n = 0
    f = 0
    for kx in range(kmax+1):
        Cx = np.cos(kx * x)
        Sx = np.sin(kx * x)
        for ky in range(kmax+1):
            if kx**2 + ky**2 <= kmax**2:
                if rand.rand() < density:
                    Cy = np.cos(ky * y)
                    Sy = np.sin(ky * y)
                    a, b, c, d = rand.randn(4)
                    f += a*Cx*Cy + b*Cx*Sy + c*Sx*Cy + d*Sx*Sy
                    n += 1
    f /= n**0.5
    return f


def rand_fourier_series_3d(x, y, z, kmax, density=1, rand=None):
    if rand is None:
        rand = np.random.RandomState()
    n = 0
    f = 0
    for kx in range(kmax+1):
        Cx = np.cos(kx * x)
        Sx = np.sin(kx * x)
        for ky in range(kmax+1):
            if kx**2 + ky**2 <= kmax**2:
                Cy = np.cos(ky * y)
                Sy = np.sin(ky * y)
                for kz in range(kmax+1):
                    if kx**2 + ky**2 + kz**2 <= kmax**2:
                        if rand.rand() < density:
                            a = rand.randn(8)
                            Cz = np.cos(kz * z)
                            Sz = np.sin(kz * z)
                            f += (a[0]*Cx*Cy*Cz + 
                                  a[1]*Cx*Cy*Sz + 
                                  a[2]*Cx*Sy*Cz + 
                                  a[3]*Sx*Cy*Cz +
                                  a[4]*Cx*Sy*Sz + 
                                  a[5]*Sx*Sy*Cz + 
                                  a[6]*Sx*Cy*Sz + 
                                  a[7]*Sx*Sy*Sz)
                            n += 1
    f /= n**0.5
    return f


def rand_fourier_series_2d_lowrank(x, y, kmax, density=1, rank=1, rand=None):
    """
    Parameters
    ----------
    x, y : array-like
        Arry of positions (column and row vectors)
    kmax : int
        Maximum wavenumber (integer form)
    density : float (default: 1)
        Fraction of modes up to cutoff to populate
    rank : int (default: 1)
        Rank of the random funciton
    rand : np.random.RandomState or None (default: None)
        Random state, needed for determinism
        
    Returns
    -------
    f : array-like
        Values of the random function F(x,y)
    """ 
    if rand is None:
        rand = np.random.RandomState()
    f = 0
    for r in range(rank):
        fx = rand_fourier_series_1d(x, kmax, density, rand)
        fy = rand_fourier_series_1d(y, kmax, density, rand)
        f += fx * fy
    f /= rank**0.5
    return f


def rand_fourier_series_3d_lowrank(x, y, z, kmax, density=1, rank=1, rand=None):
    """
    Parameters
    ----------
    x, y, z : array-like
        Arry of positions (column and row vectors)
    kmax : int
        Maximum wavenumber (integer form)
    density : float (default: 1)
        Fraction of modes up to cutoff to populate
    rank : int (default: 1)
        Rank of the random funciton
    rand : np.random.RandomState or None (default: None)
        Random state, needed for determinism
        
    Returns
    -------
    f : array-like
        Values of the random function F(x,y,z)
    """ 
    if rand is None:
        rand = np.random.RandomState()
    f = 0
    for r in range(rank):
        fx = rand_fourier_series_1d(x, kmax, density, rand)
        fy = rand_fourier_series_1d(y, kmax, density, rand)
        fz = rand_fourier_series_1d(z, kmax, density, rand)
        f += fx * fy * fz
    f /= rank**0.5
    return f

