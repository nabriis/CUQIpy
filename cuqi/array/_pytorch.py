"""PyTorch backend implementation for CUQIpy array operations."""

import numpy as np

def load_backend():
    """Load PyTorch backend and return the module."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch not available. Install with: pip install torch")

def get_backend_functions(backend_module):
    """Get PyTorch-specific function implementations."""
    
    # Basic array creation with consistent dtype defaults
    def array(x, dtype=None, requires_grad=False):
        if x is None:
            return None
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.tensor(x, dtype=dtype, requires_grad=requires_grad)
    
    def zeros(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.zeros(*args, dtype=dtype, **kwargs)
    
    def ones(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.ones(*args, dtype=dtype, **kwargs)
    
    def eye(N, M=None, k=0, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        # PyTorch eye doesn't support k parameter
        if k != 0:
            if M is None:
                M = N
            result = backend_module.zeros(N, M, dtype=dtype, **kwargs)
            if k > 0:
                result[:-k, k:] = backend_module.eye(min(N, M-k), dtype=dtype)
            else:
                result[-k:, :M+k] = backend_module.eye(min(N+k, M), dtype=dtype)
            return result
        else:
            if M is None:
                return backend_module.eye(N, dtype=dtype, **kwargs)
            else:
                return backend_module.eye(N, M, dtype=dtype, **kwargs)
    
    def identity(n, dtype=None):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.eye(n, dtype=dtype)
    
    # PyTorch-specific function adaptations
    def linspace(start, end, steps=50, **kwargs):
        return backend_module.linspace(start, end, steps, **kwargs)
    
    def sum(x, axis=None, **kwargs):
        if axis is None:
            return backend_module.sum(x, **kwargs)
        return backend_module.sum(x, dim=axis, **kwargs)
    
    def mean(x, axis=None, **kwargs):
        if axis is None:
            return backend_module.mean(x, **kwargs)
        return backend_module.mean(x, dim=axis, **kwargs)
    
    def std(x, axis=None, **kwargs):
        if axis is None:
            return backend_module.std(x, **kwargs)
        return backend_module.std(x, dim=axis, **kwargs)
    
    def var(x, axis=None, **kwargs):
        if axis is None:
            return backend_module.var(x, **kwargs)
        return backend_module.var(x, dim=axis, **kwargs)
    
    def transpose(x, axes=None):
        if hasattr(x, 'dim') and x.dim() <= 1:
            return x  # 1D tensors don't need transpose
        elif axes is None:
            return x.T
        else:
            return x.permute(axes)
    
    # More one-liners
    where = lambda condition, x=None, y=None: (
        backend_module.where(condition) if x is None and y is None
        else backend_module.where(condition, x, y)
    )
    isscalar = lambda x: x.dim() == 0 if hasattr(x, 'dim') else np.isscalar(x)
    
    # Simplified one-liner functions
    quantile = lambda x, q, axis=None, **kwargs: (
        backend_module.quantile(x.flatten(), q, **kwargs) if axis is None
        else backend_module.quantile(x, q, dim=axis, **kwargs)
    )
    percentile = lambda x, q, axis=None, **kwargs: quantile(x, q/100.0, axis=axis, **kwargs)
    
    def piecewise(x, condlist, funclist, *args, **kwargs):
        x_np, condlist_np = to_numpy(x), [to_numpy(c) for c in condlist]
        result_np = np.piecewise(x_np, condlist_np, funclist, *args, **kwargs)
        return array(result_np, dtype=getattr(x, 'dtype', backend_module.float64))
    
    # Standard functions that work directly
    functions = {
        'array': array,
        'zeros': zeros,
        'ones': ones,
        'zeros_like': backend_module.zeros_like,
        'ones_like': backend_module.ones_like,
        'empty': backend_module.empty,
        'empty_like': backend_module.empty_like,
        'full': backend_module.full,
        'full_like': backend_module.full_like,
        'arange': backend_module.arange,
        'linspace': linspace,
        'eye': eye,
        'identity': identity,
        'diag': backend_module.diag,
        
        # Shape manipulation
        'reshape': lambda x, shape: x.reshape(shape),
        'ravel': lambda x: x.flatten(),
        'flatten': lambda x: x.flatten(),
        'transpose': transpose,
        'swapaxes': backend_module.swapaxes,
        'flip': backend_module.flip,
        'roll': backend_module.roll,
        
        # Array joining
        'concatenate': lambda arrays, axis=0: backend_module.cat(arrays, dim=axis),
        'stack': lambda arrays, axis=0: backend_module.stack(arrays, dim=axis),
        'vstack': lambda arrays: backend_module.vstack(arrays),
        'hstack': lambda arrays: backend_module.hstack(arrays),
        
        # Mathematical functions
        'sum': sum,
        'prod': lambda x, axis=None: backend_module.prod(x, dim=axis) if axis is not None else backend_module.prod(x),
        'mean': mean,
        'std': std,
        'var': var,
        'min': lambda x, axis=None: backend_module.min(x, dim=axis)[0] if axis is not None else backend_module.min(x),
        'max': lambda x, axis=None: backend_module.max(x, dim=axis)[0] if axis is not None else backend_module.max(x),
        'argmin': lambda x, axis=None: backend_module.argmin(x, dim=axis) if axis is not None else backend_module.argmin(x),
        'argmax': lambda x, axis=None: backend_module.argmax(x, dim=axis) if axis is not None else backend_module.argmax(x),
        'sort': lambda x, axis=-1: backend_module.sort(x, dim=axis)[0],
        'argsort': lambda x, axis=-1: backend_module.argsort(x, dim=axis),
        'any': lambda x, axis=None: backend_module.any(x, dim=axis) if axis is not None else backend_module.any(x),
        'all': lambda x, axis=None: backend_module.all(x, dim=axis) if axis is not None else backend_module.all(x),
        'cumsum': lambda x, axis=None: backend_module.cumsum(x, dim=axis or 0),
        'repeat': lambda x, repeats, axis=None: x.repeat_interleave(repeats, dim=axis),
        'isclose': backend_module.isclose,
        'percentile': percentile,
        'median': lambda x, axis=None: backend_module.median(x, dim=axis)[0] if axis is not None else backend_module.median(x),
        'multiply': backend_module.mul,
        'tile': lambda x, reps: x.repeat(reps),
        'piecewise': piecewise,
        
        # Linear algebra
        'dot': backend_module.dot,
        'matmul': backend_module.matmul,
        'outer': backend_module.outer,
        'cross': backend_module.cross,
        'einsum': backend_module.einsum,
        'tril': backend_module.tril,
        'triu': backend_module.triu,
        
        # Trigonometric functions
        'sin': lambda x: backend_module.sin(backend_module.as_tensor(x)),
        'cos': lambda x: backend_module.cos(backend_module.as_tensor(x)),
        'tan': lambda x: backend_module.tan(backend_module.as_tensor(x)),
        'arcsin': lambda x: backend_module.asin(backend_module.as_tensor(x)),
        'arccos': lambda x: backend_module.acos(backend_module.as_tensor(x)),
        'arctan': lambda x: backend_module.atan(backend_module.as_tensor(x)),
        'arctan2': lambda y, x: backend_module.atan2(backend_module.as_tensor(y), backend_module.as_tensor(x)),
        'sinh': lambda x: backend_module.sinh(backend_module.as_tensor(x)),
        'cosh': lambda x: backend_module.cosh(backend_module.as_tensor(x)),
        'tanh': lambda x: backend_module.tanh(backend_module.as_tensor(x)),
        
        # Exponential and logarithmic functions
        'exp': lambda x: backend_module.exp(backend_module.as_tensor(x)),
        'exp2': lambda x: backend_module.exp2(backend_module.as_tensor(x)),
        'log': lambda x: backend_module.log(backend_module.as_tensor(x)),
        'log2': lambda x: backend_module.log2(backend_module.as_tensor(x)),
        'log10': lambda x: backend_module.log10(backend_module.as_tensor(x)),
        'sqrt': lambda x: backend_module.sqrt(backend_module.as_tensor(x)),
        'square': lambda x: backend_module.square(backend_module.as_tensor(x)),
        'power': lambda x, y: backend_module.pow(backend_module.as_tensor(x), backend_module.as_tensor(y)),
        'abs': lambda x: backend_module.abs(backend_module.as_tensor(x)),
        'sign': lambda x: backend_module.sign(backend_module.as_tensor(x)),
        
        # Rounding functions
        'floor': backend_module.floor,
        'ceil': backend_module.ceil,
        'round': backend_module.round,
        'clip': backend_module.clamp,
        
        # Logic functions
        'where': where,
        'isnan': backend_module.isnan,
        'isinf': backend_module.isinf,
        'isfinite': backend_module.isfinite,
        'allclose': backend_module.allclose,
        'array_equal': backend_module.equal,
        'isscalar': isscalar,
        'fix': lambda x: backend_module.trunc(x),
        
        # Complex numbers
        'real': backend_module.real,
        'imag': backend_module.imag,
        'conj': backend_module.conj,
        'angle': backend_module.angle,
        'absolute': backend_module.abs,
        
        # Array conversion
        'asarray': lambda x: backend_module.as_tensor(x),
        'copy': backend_module.clone,
        
        # Constants
        'newaxis': None,
        'inf': float('inf'),
        'nan': float('nan'),
        'pi': backend_module.pi,
        'e': backend_module.e,
        
        # Data types
        'int8': backend_module.int8,
        'int16': backend_module.int16,
        'int32': backend_module.int32,
        'int64': backend_module.int64,
        'uint8': backend_module.uint8,
        'float16': backend_module.float16,
        'float32': backend_module.float32,
        'float64': backend_module.float64,
        'complex64': backend_module.complex64,
        'complex128': backend_module.complex128,
        'bool_': backend_module.bool,
        
        # Array type
        'ndarray': backend_module.Tensor,
        'dtype': backend_module.dtype,
        
        # Type hierarchies (fallback to NumPy)
        'integer': np.integer,
        'floating': np.floating,
        'complexfloating': np.complexfloating,
    }
    
    # Add modules
    functions['linalg'] = backend_module.linalg
    functions['fft'] = backend_module.fft
    
    # Random module - simplified with lambda functions
    functions['random'] = type('RandomModule', (), {
        'randn': lambda *args, **kwargs: backend_module.randn(*args, **kwargs),
        'rand': lambda *args, **kwargs: backend_module.rand(*args, **kwargs),
        'randint': lambda low, high, size=None, dtype=None: (
            backend_module.randint(low, high, (1,), dtype=dtype).item() if size is None
            else backend_module.randint(low, high, size, dtype=dtype)
        ),
                          'default_rng': staticmethod(lambda seed=None: (_ for _ in ()).throw(NotImplementedError("random.default_rng not implemented for PyTorch backend")).__next__())
    })()
    
    # Polynomial module - simplified
    functions['polynomial'] = type('PolynomialModule', (), {
        'legendre': type('LegendreModule', (), {
                                      'leggauss': staticmethod(lambda deg: (_ for _ in ()).throw(NotImplementedError("polynomial.legendre.leggauss not implemented for PyTorch backend")).__next__())
        })()
    })()
    
    # Add astype method to PyTorch tensors for compatibility
    if not hasattr(backend_module.Tensor, 'astype'):
        def astype(self, dtype):
            if dtype == float:
                return self.float()
            elif dtype == int:
                return self.int()
            else:
                return self.to(dtype)
        backend_module.Tensor.astype = astype
    
    return functions

def to_numpy(x):
    """Convert PyTorch tensor to NumPy array."""
    if hasattr(x, 'detach'):
        if x.requires_grad:
            return x.detach().cpu().numpy()
        return x.cpu().numpy()
    return np.asarray(x)

def pad(array, pad_width, mode='constant', **kwargs):
    """PyTorch padding implementation."""
    import torch.nn.functional as F
    
    # Convert pad_width to PyTorch format (reverse order, flattened)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * array.dim()
    elif len(pad_width) == 2 and not isinstance(pad_width[0], (list, tuple)):
        pad_width = [pad_width] * array.dim()
    
    # Flatten and reverse for PyTorch
    pad_list = []
    for pw in reversed(pad_width):
        if isinstance(pw, (list, tuple)):
            pad_list.extend(pw)
        else:
            pad_list.extend([pw, pw])
    
    # Map mode names
    mode_map = {'constant': 'constant', 'reflect': 'reflect', 'replicate': 'replicate'}
    torch_mode = mode_map.get(mode, 'constant')
    
    constant_value = kwargs.get('constant_values', 0)
    return F.pad(array, pad_list, mode=torch_mode, value=constant_value)