"""PyTorch backend implementation for CUQIpy array operations."""

import warnings

try:
    import torch
    _backend = torch
    PYTORCH_AVAILABLE = True
except ImportError:
    _backend = None
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorch backend will not be functional.")

def get_backend():
    """Get the PyTorch backend module."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    return _backend

def to_numpy(arr):
    """Convert PyTorch tensor to NumPy array."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    # If it's not a tensor, assume it's already numpy-compatible
    import numpy as np
    return np.asarray(arr)

def pad(array, pad_width, mode='constant', **kwargs):
    """PyTorch pad function with limited mode support."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    
    if mode != 'constant':
        raise NotImplementedError(f"PyTorch backend only supports 'constant' padding mode, got '{mode}'")
    
    # Convert pad_width to PyTorch format
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * array.ndim
    elif len(pad_width) == 2 and isinstance(pad_width[0], int):
        pad_width = [pad_width] * array.ndim
    
    # PyTorch pad expects padding in reverse dimension order and flattened
    torch_pad = []
    for pad in reversed(pad_width):
        if isinstance(pad, int):
            torch_pad.extend([pad, pad])
        else:
            torch_pad.extend([pad[0], pad[1]])
    
    constant_value = kwargs.get('constant_values', 0)
    return torch.nn.functional.pad(array, torch_pad, mode='constant', value=constant_value)

# Functions that don't have direct PyTorch equivalents
def sinc(x):
    """Sinc function - not available in PyTorch."""
    raise NotImplementedError("sinc function is not available in PyTorch backend")

def fix(x):
    """Fix function - not available in PyTorch."""
    raise NotImplementedError("fix function is not available in PyTorch backend")

def piecewise(x, condlist, funclist, *args, **kwargs):
    """Piecewise function - not available in PyTorch."""
    raise NotImplementedError("piecewise function is not available in PyTorch backend")

def percentile(a, q, axis=None, **kwargs):
    """Percentile function using PyTorch quantile."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.quantile(a, q/100.0, dim=axis, **kwargs)

def median(a, axis=None, **kwargs):
    """Median function."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    if axis is None:
        return torch.median(a)
    else:
        return torch.median(a, dim=axis).values

def gradient(f, *args, **kwargs):
    """Gradient function - not available in PyTorch."""
    raise NotImplementedError("gradient function is not available in PyTorch backend")

def trapz(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal integration - not available in PyTorch."""
    raise NotImplementedError("trapz function is not available in PyTorch backend")

def interp(x, xp, fp, left=None, right=None, period=None):
    """1D interpolation - not available in PyTorch."""
    raise NotImplementedError("interp function is not available in PyTorch backend")

def histogram(a, bins=10, range=None, weights=None, density=False):
    """Histogram - limited support in PyTorch."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    if weights is not None or density:
        raise NotImplementedError("PyTorch histogram doesn't support weights or density")
    return torch.histogram(a, bins=bins, range=range)

def digitize(x, bins, right=False):
    """Digitize - not available in PyTorch."""
    raise NotImplementedError("digitize function is not available in PyTorch backend")

def bincount(x, weights=None, minlength=0):
    """Bincount function."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.bincount(x, weights=weights, minlength=minlength)

def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """Unique function with limited support."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    if axis is not None:
        raise NotImplementedError("PyTorch unique doesn't support axis parameter")
    return torch.unique(ar, return_inverse=return_inverse, return_counts=return_counts)

def setdiff1d(ar1, ar2, assume_unique=False):
    """Set difference - not available in PyTorch."""
    raise NotImplementedError("setdiff1d function is not available in PyTorch backend")

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """Set intersection - not available in PyTorch."""
    raise NotImplementedError("intersect1d function is not available in PyTorch backend")

def union1d(ar1, ar2):
    """Set union - not available in PyTorch."""
    raise NotImplementedError("union1d function is not available in PyTorch backend")

def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Test membership - not available in PyTorch."""
    raise NotImplementedError("in1d function is not available in PyTorch backend")

def isin(element, test_elements, assume_unique=False, invert=False):
    """Test membership - limited support in PyTorch."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.isin(element, test_elements)

def ediff1d(ary, to_end=None, to_begin=None):
    """Differences between consecutive elements - not available in PyTorch."""
    raise NotImplementedError("ediff1d function is not available in PyTorch backend")

def idst_function(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """Inverse discrete sine transform - not available in PyTorch."""
    raise NotImplementedError("idst function is not available in PyTorch backend")

# Linear algebra functions
def solve(a, b):
    """Solve linear system."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.linalg.solve(a, b)

def inv(a):
    """Matrix inverse."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.linalg.inv(a)

def det(a):
    """Matrix determinant."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.linalg.det(a)

def logdet(a):
    """Log determinant - not directly available."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.log(torch.abs(torch.linalg.det(a)))

def slogdet(a):
    """Sign and log determinant."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.linalg.slogdet(a)

def cholesky(a):
    """Cholesky decomposition."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    return torch.linalg.cholesky(a)

def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """Cholesky factorization - not available in PyTorch."""
    raise NotImplementedError("cho_factor is not available in PyTorch backend")

def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve using Cholesky factorization - not available in PyTorch."""
    raise NotImplementedError("cho_solve is not available in PyTorch backend")