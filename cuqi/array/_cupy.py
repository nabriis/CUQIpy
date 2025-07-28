"""CuPy backend implementation for CUQIpy array operations."""

import warnings
import numpy as np

def load_backend():
    """Load CuPy backend and return the module."""
    try:
        import cupy as cp
        return cp
    except ImportError:
        warnings.warn("CuPy not available, falling back to NumPy")
        import numpy as np
        return np

def get_backend_functions(backend_module):
    """Get CuPy-specific function implementations."""
    
    # CuPy has very similar API to NumPy, so we can use most functions directly
    functions = {
        'array': backend_module.array,
        'zeros': backend_module.zeros,
        'ones': backend_module.ones,
        'zeros_like': backend_module.zeros_like,
        'ones_like': backend_module.ones_like,
        'empty': backend_module.empty,
        'empty_like': backend_module.empty_like,
        'full': backend_module.full,
        'full_like': backend_module.full_like,
        'arange': backend_module.arange,
        'linspace': backend_module.linspace,
        'logspace': backend_module.logspace,
        'eye': backend_module.eye,
        'identity': backend_module.identity,
        'diag': backend_module.diag,
        'diagonal': backend_module.diagonal,
        
        # Shape manipulation
        'reshape': backend_module.reshape,
        'ravel': backend_module.ravel,
        'flatten': backend_module.ravel,
        'transpose': backend_module.transpose,
        'swapaxes': backend_module.swapaxes,
        'moveaxis': backend_module.moveaxis,
        'flip': backend_module.flip,
        'flipud': backend_module.flipud,
        'fliplr': backend_module.fliplr,
        'rot90': backend_module.rot90,
        'roll': backend_module.roll,
        
        # Array joining and splitting
        'concatenate': backend_module.concatenate,
        'stack': backend_module.stack,
        'vstack': backend_module.vstack,
        'hstack': backend_module.hstack,
        'dstack': backend_module.dstack,
        'split': backend_module.split,
        'hsplit': backend_module.hsplit,
        'vsplit': backend_module.vsplit,
        'dsplit': backend_module.dsplit,
        
        # Mathematical functions
        'sum': backend_module.sum,
        'prod': backend_module.prod,
        'mean': backend_module.mean,
        'std': backend_module.std,
        'var': backend_module.var,
        'min': backend_module.min,
        'max': backend_module.max,
        'argmin': backend_module.argmin,
        'argmax': backend_module.argmax,
        'sort': backend_module.sort,
        'argsort': backend_module.argsort,
        'any': backend_module.any,
        'all': backend_module.all,
        'argwhere': backend_module.argwhere,
        'cumsum': backend_module.cumsum,
        'cumprod': backend_module.cumprod,
        'diff': backend_module.diff,
        'gradient': backend_module.gradient,
        'maximum': backend_module.maximum,
        'minimum': backend_module.minimum,
        'repeat': backend_module.repeat,
        'isclose': backend_module.isclose,
        'percentile': backend_module.percentile,
        'median': backend_module.median,
        'multiply': backend_module.multiply,
        'tile': backend_module.tile,
        'float_power': backend_module.float_power,
        'piecewise': backend_module.piecewise,
        
        # Linear algebra
        'dot': backend_module.dot,
        'matmul': backend_module.matmul,
        'inner': backend_module.inner,
        'outer': backend_module.outer,
        'cross': backend_module.cross,
        'tensordot': backend_module.tensordot,
        'einsum': backend_module.einsum,
        'tril': backend_module.tril,
        'triu': backend_module.triu,
        
        # Trigonometric functions
        'sin': backend_module.sin,
        'cos': backend_module.cos,
        'tan': backend_module.tan,
        'arcsin': backend_module.arcsin,
        'arccos': backend_module.arccos,
        'arctan': backend_module.arctan,
        'arctan2': backend_module.arctan2,
        'sinh': backend_module.sinh,
        'cosh': backend_module.cosh,
        'tanh': backend_module.tanh,
        
        # Exponential and logarithmic functions
        'exp': backend_module.exp,
        'exp2': backend_module.exp2,
        'log': backend_module.log,
        'log2': backend_module.log2,
        'log10': backend_module.log10,
        'sqrt': backend_module.sqrt,
        'square': backend_module.square,
        'power': backend_module.power,
        'abs': backend_module.abs,
        'sign': backend_module.sign,
        
        # Rounding functions
        'floor': backend_module.floor,
        'ceil': backend_module.ceil,
        'round': backend_module.round,
        'clip': backend_module.clip,
        
        # Logic functions
        'where': backend_module.where,
        'isnan': backend_module.isnan,
        'isinf': backend_module.isinf,
        'isfinite': backend_module.isfinite,
        'count_nonzero': backend_module.count_nonzero,
        'allclose': backend_module.allclose,
        'array_equiv': backend_module.array_equiv,
        'array_equal': backend_module.array_equal,
        'isscalar': backend_module.isscalar,
        'sinc': backend_module.sinc,
        'fix': backend_module.fix,
        
        # Complex numbers
        'real': backend_module.real,
        'imag': backend_module.imag,
        'conj': backend_module.conj,
        'angle': backend_module.angle,
        'absolute': backend_module.absolute,
        
        # Array conversion
        'asarray': backend_module.asarray,
        'asanyarray': backend_module.asanyarray,
        'ascontiguousarray': backend_module.ascontiguousarray,
        'asfortranarray': backend_module.asfortranarray,
        'copy': backend_module.copy,
        
        # Data types
        'finfo': backend_module.finfo,
        'iinfo': backend_module.iinfo,
        
        # Constants
        'newaxis': backend_module.newaxis,
        'inf': backend_module.inf,
        'nan': backend_module.nan,
        'pi': backend_module.pi,
        'e': backend_module.e,
        
        # Data types
        'int8': backend_module.int8,
        'int16': backend_module.int16,
        'int32': backend_module.int32,
        'int64': backend_module.int64,
        'uint8': backend_module.uint8,
        'uint16': backend_module.uint16,
        'uint32': backend_module.uint32,
        'uint64': backend_module.uint64,
        'float16': backend_module.float16,
        'float32': backend_module.float32,
        'float64': backend_module.float64,
        'complex64': backend_module.complex64,
        'complex128': backend_module.complex128,
        'bool_': backend_module.bool_,
        
        # Array type
        'ndarray': backend_module.ndarray,
        'dtype': backend_module.dtype,
        
        # Type hierarchies
        'integer': getattr(backend_module, 'integer', np.integer),
        'floating': getattr(backend_module, 'floating', np.floating),
        'complexfloating': getattr(backend_module, 'complexfloating', np.complexfloating),
    }
    
    # Add modules
    functions['linalg'] = backend_module.linalg
    functions['random'] = backend_module.random
    functions['fft'] = backend_module.fft
    
    # Add polynomial module (fallback to NumPy if not available)
    if hasattr(backend_module, 'polynomial'):
        functions['polynomial'] = backend_module.polynomial
    else:
        import numpy as np
        functions['polynomial'] = np.polynomial
    
    return functions

def to_numpy(x):
    """Convert CuPy array to NumPy array."""
    if hasattr(x, 'get'):
        return x.get()  # CuPy method to transfer to CPU
    return np.asarray(x)

def pad(array, pad_width, mode='constant', **kwargs):
    """CuPy padding implementation."""
    if mode != 'constant' and 'constant_values' in kwargs:
        kwargs.pop('constant_values')
    try:
        import cupy as cp
        return cp.pad(array, pad_width, mode=mode, **kwargs)
    except ImportError:
        return np.pad(np.asarray(array), pad_width, mode=mode, **kwargs)