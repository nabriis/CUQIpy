"""NumPy backend implementation for CUQIpy array operations."""

import numpy as np
import scipy.linalg as splinalg
from scipy.fftpack import idst

# Module reference
_backend = np

def get_backend():
    """Get the NumPy backend module."""
    return _backend

def to_numpy(arr):
    """Convert array to NumPy array (identity for NumPy backend)."""
    return np.asarray(arr)

def pad(array, pad_width, mode='constant', **kwargs):
    """NumPy pad function."""
    return np.pad(array, pad_width, mode=mode, **kwargs)

# Additional NumPy-specific functions that might not exist in other backends
def sinc(x):
    """NumPy sinc function."""
    return np.sinc(x)

def fix(x):
    """NumPy fix function."""
    return np.fix(x)

def piecewise(x, condlist, funclist, *args, **kwargs):
    """NumPy piecewise function."""
    return np.piecewise(x, condlist, funclist, *args, **kwargs)

def percentile(a, q, axis=None, **kwargs):
    """NumPy percentile function."""
    return np.percentile(a, q, axis=axis, **kwargs)

def median(a, axis=None, **kwargs):
    """NumPy median function."""
    return np.median(a, axis=axis, **kwargs)

def gradient(f, *args, **kwargs):
    """NumPy gradient function."""
    return np.gradient(f, *args, **kwargs)

def trapz(y, x=None, dx=1.0, axis=-1):
    """NumPy trapz function."""
    return np.trapz(y, x=x, dx=dx, axis=axis)

def interp(x, xp, fp, left=None, right=None, period=None):
    """NumPy interp function."""
    return np.interp(x, xp, fp, left=left, right=right, period=period)

def histogram(a, bins=10, range=None, weights=None, density=False):
    """NumPy histogram function."""
    return np.histogram(a, bins=bins, range=range, weights=weights, density=density)

def digitize(x, bins, right=False):
    """NumPy digitize function."""
    return np.digitize(x, bins, right=right)

def bincount(x, weights=None, minlength=0):
    """NumPy bincount function."""
    return np.bincount(x, weights=weights, minlength=minlength)

def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """NumPy unique function."""
    return np.unique(ar, return_index=return_index, return_inverse=return_inverse, 
                     return_counts=return_counts, axis=axis)

def setdiff1d(ar1, ar2, assume_unique=False):
    """NumPy setdiff1d function."""
    return np.setdiff1d(ar1, ar2, assume_unique=assume_unique)

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """NumPy intersect1d function."""
    return np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)

def union1d(ar1, ar2):
    """NumPy union1d function."""
    return np.union1d(ar1, ar2)

def in1d(ar1, ar2, assume_unique=False, invert=False):
    """NumPy in1d function."""
    return np.in1d(ar1, ar2, assume_unique=assume_unique, invert=invert)

def isin(element, test_elements, assume_unique=False, invert=False):
    """NumPy isin function."""
    return np.isin(element, test_elements, assume_unique=assume_unique, invert=invert)

def ediff1d(ary, to_end=None, to_begin=None):
    """NumPy ediff1d function."""
    return np.ediff1d(ary, to_end=to_end, to_begin=to_begin)

def idst_function(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """Inverse discrete sine transform using scipy."""
    return idst(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)

# Expose scipy linalg functions
def solve(a, b):
    """Solve linear system using scipy."""
    return splinalg.solve(a, b)

def inv(a):
    """Matrix inverse using scipy."""
    return splinalg.inv(a)

def det(a):
    """Matrix determinant using scipy."""
    return splinalg.det(a)

def logdet(a):
    """Log determinant using scipy."""
    return splinalg.logdet(a)

def slogdet(a):
    """Sign and log determinant using scipy."""
    return splinalg.slogdet(a)

def cholesky(a):
    """Cholesky decomposition using scipy."""
    return splinalg.cholesky(a)

def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """Cholesky factorization using scipy."""
    return splinalg.cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)

def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve using Cholesky factorization."""
    return splinalg.cho_solve(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)