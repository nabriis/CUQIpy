"""
CUQIpy Array Backend Module

This module provides a backend-agnostic interface for array operations.
It allows switching between different array backends like NumPy, PyTorch, etc.
through environment variables or programmatic configuration.

Usage:
    import cuqi.array as xp
    
    # Use array operations
    x = xp.array([1, 2, 3])
    y = xp.zeros((3, 3))
    z = xp.dot(x, y)

Backend Selection:
    Set the CUQI_ARRAY_BACKEND environment variable or use set_backend():
    - "numpy" (default): Use NumPy backend
    - "pytorch"/"torch": Use PyTorch backend (GPU + gradients)
    
    Example:
        export CUQI_ARRAY_BACKEND=pytorch
        # or
        xp.set_backend("pytorch")
"""

import os
import warnings
from ._array import CUQIarray

# Backend selection mechanism
_BACKEND_NAME = os.getenv("CUQI_ARRAY_BACKEND", "numpy").lower()
_backend_module = None
_backend_impl = None

def get_backend_name():
    """Get the current backend name."""
    return _BACKEND_NAME

def set_backend(backend_name):
    """Set the array backend programmatically.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend to use ("numpy", "pytorch")
    """
    global _BACKEND_NAME, _backend_module, _backend_impl
    requested_backend = backend_name.lower()
    _backend_module, _backend_impl = _load_backend(requested_backend)
    # Update _BACKEND_NAME to reflect actual loaded backend (might be numpy if pytorch fails)
    if requested_backend in ["pytorch", "torch"] and hasattr(_backend_impl, '__name__'):
        if _backend_impl.__name__.endswith('_numpy'):
            _BACKEND_NAME = "numpy"
        else:
            _BACKEND_NAME = requested_backend
    else:
        _BACKEND_NAME = requested_backend
    _expose_backend_functions()

def _load_backend(backend_name):
    """Load the specified backend module and implementation."""
    if backend_name in ["numpy", "np"]:
        from . import _numpy as backend_impl
        backend_module = backend_impl.get_backend()
        return backend_module, backend_impl
    elif backend_name in ["pytorch", "torch"]:
        from . import _pytorch as backend_impl
        try:
            backend_module = backend_impl.get_backend()
            return backend_module, backend_impl
        except ImportError as e:
            # Don't fall back to NumPy - let the ImportError propagate
            raise ImportError("PyTorch not available. Install with: pip install torch") from e
    else:
        raise ValueError(f"Unsupported array backend '{backend_name}'. "
                        f"Supported backends: numpy, pytorch")

def _expose_backend_functions():
    """Expose backend functions at module level."""
    import numpy as np
    
    # Get the backend module
    backend = _backend_module
    impl = _backend_impl
    
    # Basic array creation - directly from backend
    globals()['array'] = backend.array if hasattr(backend, 'array') else backend.tensor
    globals()['zeros'] = backend.zeros
    globals()['ones'] = backend.ones
    globals()['zeros_like'] = backend.zeros_like
    globals()['ones_like'] = backend.ones_like
    globals()['empty'] = backend.empty
    globals()['empty_like'] = backend.empty_like
    globals()['full'] = backend.full
    globals()['full_like'] = backend.full_like
    globals()['arange'] = backend.arange
    globals()['linspace'] = backend.linspace
    globals()['logspace'] = backend.logspace if hasattr(backend, 'logspace') else np.logspace
    globals()['eye'] = backend.eye
    globals()['identity'] = backend.eye if not hasattr(backend, 'identity') else backend.identity
    globals()['diag'] = backend.diag
    globals()['diagonal'] = backend.diagonal if hasattr(backend, 'diagonal') else lambda a, offset=0, axis1=0, axis2=1: backend.diag(a)
    globals()['meshgrid'] = backend.meshgrid if hasattr(backend, 'meshgrid') else np.meshgrid
    
    # Shape manipulation
    globals()['reshape'] = backend.reshape
    globals()['ravel'] = backend.ravel if hasattr(backend, 'ravel') else lambda a: a.flatten()
    globals()['flatten'] = backend.flatten if hasattr(backend, 'flatten') else lambda a: a.flatten()
    globals()['transpose'] = backend.transpose if hasattr(backend, 'transpose') else lambda a, axes=None: a.T if axes is None else a.permute(axes)
    globals()['swapaxes'] = backend.swapaxes if hasattr(backend, 'swapaxes') else backend.transpose
    globals()['moveaxis'] = backend.moveaxis if hasattr(backend, 'moveaxis') else np.moveaxis
    globals()['flip'] = backend.flip
    globals()['flipud'] = backend.flipud if hasattr(backend, 'flipud') else lambda a: backend.flip(a, 0)
    globals()['fliplr'] = backend.fliplr if hasattr(backend, 'fliplr') else lambda a: backend.flip(a, 1)
    globals()['rot90'] = backend.rot90 if hasattr(backend, 'rot90') else np.rot90
    globals()['roll'] = backend.roll
    
    # Array properties
    globals()['shape'] = lambda x: x.shape
    def size_func(x):
        if hasattr(x, 'size'):
            return x.size
        elif hasattr(x, 'numel'):
            return x.numel()
        elif hasattr(x, 'shape'):
            # For objects with shape but no size attribute (like LinearOperator)
            import numpy as np
            return np.prod(x.shape)
        elif hasattr(x, '__len__'):
            return len(x)
        else:
            return 1  # Scalar
    globals()['size'] = size_func
    globals()['ndim'] = lambda x: x.ndim if hasattr(x, 'ndim') else x.dim() if hasattr(x, 'dim') else len(x.shape) if hasattr(x, 'shape') else 0
    
    # Array joining and splitting
    if _BACKEND_NAME in ["pytorch", "torch"]:
        globals()['concatenate'] = lambda arrays, axis=0: backend.cat(arrays, dim=axis)
        globals()['stack'] = lambda arrays, axis=0: backend.stack(arrays, dim=axis)
    else:
        globals()['concatenate'] = backend.concatenate
        globals()['stack'] = backend.stack
    
    globals()['vstack'] = backend.vstack
    globals()['hstack'] = backend.hstack
    globals()['dstack'] = backend.dstack if hasattr(backend, 'dstack') else np.dstack
    globals()['split'] = backend.split
    globals()['hsplit'] = backend.hsplit if hasattr(backend, 'hsplit') else lambda ary, indices_or_sections: backend.split(ary, indices_or_sections, 1)
    globals()['vsplit'] = backend.vsplit if hasattr(backend, 'vsplit') else lambda ary, indices_or_sections: backend.split(ary, indices_or_sections, 0)
    globals()['dsplit'] = backend.dsplit if hasattr(backend, 'dsplit') else lambda ary, indices_or_sections: backend.split(ary, indices_or_sections, 2)
    
    # Mathematical functions
    if _BACKEND_NAME in ["pytorch", "torch"]:
        globals()['sum'] = lambda x, axis=None, **kwargs: backend.sum(x, **kwargs) if axis is None else backend.sum(x, dim=axis, **kwargs)
        globals()['prod'] = lambda x, axis=None, **kwargs: backend.prod(x, **kwargs) if axis is None else backend.prod(x, dim=axis, **kwargs)
        globals()['mean'] = lambda x, axis=None, **kwargs: backend.mean(x, **kwargs) if axis is None else backend.mean(x, dim=axis, **kwargs)
        globals()['std'] = lambda x, axis=None, **kwargs: backend.std(x, **kwargs) if axis is None else backend.std(x, dim=axis, **kwargs)
        globals()['var'] = lambda x, axis=None, **kwargs: backend.var(x, **kwargs) if axis is None else backend.var(x, dim=axis, **kwargs)
        globals()['min'] = lambda x, axis=None, **kwargs: backend.min(x, **kwargs) if axis is None else backend.min(x, dim=axis, **kwargs)[0]
        globals()['max'] = lambda x, axis=None, **kwargs: backend.max(x, **kwargs) if axis is None else backend.max(x, dim=axis, **kwargs)[0]
        globals()['argmin'] = lambda x, axis=None, **kwargs: backend.argmin(x, **kwargs) if axis is None else backend.argmin(x, dim=axis, **kwargs)
        globals()['argmax'] = lambda x, axis=None, **kwargs: backend.argmax(x, **kwargs) if axis is None else backend.argmax(x, dim=axis, **kwargs)
    else:
        globals()['sum'] = backend.sum
        globals()['prod'] = backend.prod
        globals()['mean'] = backend.mean
        globals()['std'] = backend.std
        globals()['var'] = backend.var
        globals()['min'] = backend.min
        globals()['max'] = backend.max
        globals()['argmin'] = backend.argmin
        globals()['argmax'] = backend.argmax
    
    globals()['sort'] = backend.sort
    globals()['argsort'] = backend.argsort
    globals()['any'] = backend.any
    globals()['all'] = backend.all
    globals()['argwhere'] = backend.argwhere if hasattr(backend, 'argwhere') else np.argwhere
    globals()['cumsum'] = backend.cumsum
    globals()['cumprod'] = backend.cumprod
    globals()['diff'] = backend.diff if hasattr(backend, 'diff') else np.diff
    globals()['maximum'] = backend.maximum
    globals()['minimum'] = backend.minimum
    globals()['repeat'] = backend.repeat
    globals()['tile'] = backend.tile
    globals()['multiply'] = backend.multiply if hasattr(backend, 'multiply') else backend.mul
    globals()['isclose'] = backend.isclose
    globals()['allclose'] = backend.allclose
    globals()['array_equal'] = backend.array_equal if hasattr(backend, 'array_equal') else lambda a, b: backend.all(a == b)
    globals()['array_equiv'] = backend.array_equiv if hasattr(backend, 'array_equiv') else lambda a, b: backend.all(a == b)
    globals()['count_nonzero'] = backend.count_nonzero
    globals()['isscalar'] = lambda x: x.dim() == 0 if hasattr(x, 'dim') else np.isscalar(x)
    globals()['float_power'] = backend.float_power if hasattr(backend, 'float_power') else backend.pow
    
    # Linear algebra
    globals()['dot'] = backend.dot if hasattr(backend, 'dot') else backend.matmul
    globals()['matmul'] = backend.matmul
    globals()['inner'] = backend.inner if hasattr(backend, 'inner') else lambda a, b: backend.tensordot(a, b, dims=1)
    globals()['outer'] = backend.outer
    globals()['cross'] = backend.cross
    globals()['tensordot'] = backend.tensordot
    globals()['einsum'] = backend.einsum
    globals()['tril'] = backend.tril
    globals()['triu'] = backend.triu
    
    # Trigonometric functions
    globals()['sin'] = backend.sin
    globals()['cos'] = backend.cos
    globals()['tan'] = backend.tan
    globals()['arcsin'] = backend.arcsin if hasattr(backend, 'arcsin') else backend.asin
    globals()['arccos'] = backend.arccos if hasattr(backend, 'arccos') else backend.acos
    globals()['arctan'] = backend.arctan if hasattr(backend, 'arctan') else backend.atan
    globals()['arctan2'] = backend.arctan2 if hasattr(backend, 'arctan2') else backend.atan2
    globals()['sinh'] = backend.sinh
    globals()['cosh'] = backend.cosh
    globals()['tanh'] = backend.tanh
    
    # Exponential and logarithmic functions
    globals()['exp'] = backend.exp
    globals()['exp2'] = backend.exp2 if hasattr(backend, 'exp2') else lambda x: backend.pow(2, x)
    globals()['log'] = backend.log
    globals()['log2'] = backend.log2
    globals()['log10'] = backend.log10
    globals()['sqrt'] = backend.sqrt
    globals()['square'] = backend.square
    globals()['power'] = backend.power if hasattr(backend, 'power') else backend.pow
    globals()['abs'] = backend.abs
    globals()['absolute'] = backend.absolute if hasattr(backend, 'absolute') else backend.abs
    globals()['sign'] = backend.sign
    
    # Rounding functions
    globals()['floor'] = backend.floor
    globals()['ceil'] = backend.ceil
    globals()['round'] = backend.round
    globals()['clip'] = backend.clip if hasattr(backend, 'clip') else backend.clamp
    
    # Logic functions
    globals()['where'] = backend.where
    globals()['isnan'] = backend.isnan
    globals()['isinf'] = backend.isinf
    globals()['isfinite'] = backend.isfinite
    
    # Complex numbers
    globals()['real'] = backend.real if hasattr(backend, 'real') else lambda x: x.real
    globals()['imag'] = backend.imag if hasattr(backend, 'imag') else lambda x: x.imag
    globals()['conj'] = backend.conj
    globals()['angle'] = backend.angle
    
    # Array conversion
    globals()['asarray'] = backend.asarray if hasattr(backend, 'asarray') else backend.tensor
    globals()['asanyarray'] = backend.asanyarray if hasattr(backend, 'asanyarray') else backend.asarray if hasattr(backend, 'asarray') else backend.tensor
    globals()['ascontiguousarray'] = backend.ascontiguousarray if hasattr(backend, 'ascontiguousarray') else lambda x: x
    globals()['asfortranarray'] = backend.asfortranarray if hasattr(backend, 'asfortranarray') else lambda x: x
    globals()['copy'] = backend.copy if hasattr(backend, 'copy') else lambda x: x.clone() if hasattr(x, 'clone') else x.copy()
    globals()['deepcopy'] = backend.deepcopy if hasattr(backend, 'deepcopy') else lambda x: x.clone() if hasattr(x, 'clone') else x.copy()
    
    # Type information
    globals()['finfo'] = backend.finfo if hasattr(backend, 'finfo') else np.finfo
    globals()['iinfo'] = backend.iinfo if hasattr(backend, 'iinfo') else np.iinfo
    
    # Data types
    if _BACKEND_NAME in ["pytorch", "torch"]:
        globals()['int8'] = backend.int8
        globals()['int16'] = backend.int16
        globals()['int32'] = backend.int32
        globals()['int64'] = backend.int64
        globals()['uint8'] = backend.uint8
        globals()['float16'] = backend.float16
        globals()['float32'] = backend.float32
        globals()['float64'] = backend.float64
        globals()['complex64'] = backend.complex64
        globals()['complex128'] = backend.complex128
        globals()['dtype'] = backend.dtype
        # PyTorch doesn't have these, use numpy's
        globals()['uint16'] = np.uint16
        globals()['uint32'] = np.uint32
        globals()['uint64'] = np.uint64
        globals()['integer'] = np.integer
        globals()['floating'] = np.floating
        globals()['complexfloating'] = np.complexfloating
    else:
        globals()['int8'] = backend.int8
        globals()['int16'] = backend.int16
        globals()['int32'] = backend.int32
        globals()['int64'] = backend.int64
        globals()['uint8'] = backend.uint8
        globals()['uint16'] = backend.uint16
        globals()['uint32'] = backend.uint32
        globals()['uint64'] = backend.uint64
        globals()['float16'] = backend.float16
        globals()['float32'] = backend.float32
        globals()['float64'] = backend.float64
        globals()['complex64'] = backend.complex64
        globals()['complex128'] = backend.complex128
        globals()['dtype'] = backend.dtype
        globals()['integer'] = backend.integer
        globals()['floating'] = backend.floating
        globals()['complexfloating'] = backend.complexfloating
    
    # Constants
    globals()['newaxis'] = backend.newaxis if hasattr(backend, 'newaxis') else None
    globals()['inf'] = backend.inf
    globals()['nan'] = backend.nan
    globals()['pi'] = backend.pi
    globals()['e'] = backend.e if hasattr(backend, 'e') else np.e
    
    # Submodules
    globals()['random'] = backend.random if hasattr(backend, 'random') else backend
    globals()['linalg'] = backend.linalg
    globals()['fft'] = backend.fft if hasattr(backend, 'fft') else np.fft
    globals()['polynomial'] = backend.polynomial if hasattr(backend, 'polynomial') else np.polynomial
    
    # Backend-specific functions from implementation module
    globals()['to_numpy'] = impl.to_numpy
    globals()['pad'] = impl.pad
    globals()['gradient'] = impl.gradient
    globals()['percentile'] = impl.percentile
    globals()['median'] = impl.median
    globals()['piecewise'] = impl.piecewise
    globals()['sinc'] = impl.sinc
    globals()['fix'] = impl.fix
    globals()['trapz'] = impl.trapz
    globals()['interp'] = impl.interp
    globals()['histogram'] = impl.histogram
    globals()['digitize'] = impl.digitize
    globals()['bincount'] = impl.bincount
    globals()['unique'] = impl.unique
    globals()['setdiff1d'] = impl.setdiff1d
    globals()['intersect1d'] = impl.intersect1d
    globals()['union1d'] = impl.union1d
    globals()['in1d'] = impl.in1d
    globals()['isin'] = impl.isin
    globals()['ediff1d'] = impl.ediff1d
    
    # Sparse support
    from . import _sparse
    globals()['sparse'] = _sparse
    
    # ndarray type
    if _BACKEND_NAME in ["pytorch", "torch"]:
        globals()['ndarray'] = backend.Tensor
    else:
        globals()['ndarray'] = backend.ndarray

# Additional utility functions
def is_array_like(obj):
    """Check if object is array-like for the current backend."""
    if _backend_module is None:
        return False
    if _BACKEND_NAME in ["pytorch", "torch"]:
        import torch
        return isinstance(obj, (torch.Tensor, list, tuple)) or hasattr(obj, '__array__')
    else:
        import numpy as np
        return isinstance(obj, (np.ndarray, list, tuple)) or hasattr(obj, '__array__')

def from_numpy(arr):
    """Convert NumPy array to current backend array."""
    if _BACKEND_NAME in ["pytorch", "torch"]:
        return _backend_module.from_numpy(arr)
    else:
        return _backend_module.asarray(arr)

# Scipy linalg compatibility
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """Inverse discrete sine transform."""
    return _backend_impl.idst_function(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)

# Scipy linalg functions
def solve(a, b):
    """Solve linear system."""
    return _backend_impl.solve(a, b)

def inv(a):
    """Matrix inverse."""
    return _backend_impl.inv(a)

def det(a):
    """Matrix determinant."""
    return _backend_impl.det(a)

def logdet(a):
    """Log determinant."""
    return _backend_impl.logdet(a)

def slogdet(a):
    """Sign and log determinant."""
    return _backend_impl.slogdet(a)

def cholesky(a):
    """Cholesky decomposition."""
    return _backend_impl.cholesky(a)

def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """Cholesky factorization."""
    return _backend_impl.cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)

def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve using Cholesky factorization."""
    return _backend_impl.cho_solve(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)

# Initialize backend on import
_backend_module, _backend_impl = _load_backend(_BACKEND_NAME)
_expose_backend_functions()

# Define what gets exported
__all__ = [
    # Backend control
    'get_backend_name', 'set_backend', 'to_numpy', 'from_numpy', 'is_array_like',
    
    # Array creation
    'array', 'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like',
    'full', 'full_like', 'arange', 'linspace', 'logspace', 'eye', 'identity',
    'diag', 'diagonal', 'meshgrid',
    
    # Shape manipulation
    'reshape', 'ravel', 'flatten', 'transpose', 'swapaxes', 'moveaxis',
    'flip', 'flipud', 'fliplr', 'rot90', 'roll', 'shape', 'size', 'ndim',
    
    # Array joining and splitting
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    
    # Mathematical functions
    'sum', 'prod', 'mean', 'std', 'var', 'min', 'max', 'argmin', 'argmax',
    'sort', 'argsort', 'any', 'all', 'argwhere', 'cumsum', 'cumprod', 'diff',
    'gradient', 'maximum', 'minimum', 'repeat', 'tile', 'multiply', 'isclose',
    'percentile', 'median', 'piecewise', 'float_power',
    
    # Linear algebra
    'dot', 'matmul', 'inner', 'outer', 'cross', 'tensordot', 'einsum',
    'tril', 'triu', 'solve', 'inv', 'det', 'logdet', 'slogdet',
    'cholesky', 'cho_factor', 'cho_solve',
    
    # Trigonometric functions
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'sinh', 'cosh', 'tanh',
    
    # Exponential and logarithmic functions
    'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'square', 'power',
    'abs', 'absolute', 'sign',
    
    # Rounding functions
    'floor', 'ceil', 'round', 'clip',
    
    # Logic functions
    'where', 'isnan', 'isinf', 'isfinite', 'count_nonzero', 'allclose',
    'array_equal', 'array_equiv', 'isscalar',
    
    # Complex numbers
    'real', 'imag', 'conj', 'angle',
    
    # Array conversion
    'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray',
    'copy', 'deepcopy',
    
    # Special functions
    'sinc', 'fix', 'pad', 'trapz', 'interp', 'histogram', 'digitize',
    'bincount', 'unique', 'setdiff1d', 'intersect1d', 'union1d', 'in1d',
    'isin', 'ediff1d', 'idst',
    
    # Type information
    'finfo', 'iinfo', 'dtype', 'ndarray',
    
    # Data types
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128',
    'integer', 'floating', 'complexfloating',
    
    # Constants
    'newaxis', 'inf', 'nan', 'pi', 'e',
    
    # Modules
    'random', 'linalg', 'fft', 'polynomial', 'sparse',
    
    # CUQIarray
    'CUQIarray'
]
