import os
import importlib
import sys
import numpy as _np
ndarray = _np.ndarray

# Discover available backends
BACKENDS = {}
for name in ['numpy', 'pytorch', 'jax', 'cupy']:
    try:
        mod = importlib.import_module(f'._{name}', __name__)
        BACKENDS[name] = mod
    except ImportError:
        pass

_backend_name = os.environ.get('CUQI_ARRAY_BACKEND', 'numpy').lower()
if _backend_name not in BACKENDS:
    raise ImportError(f"Unknown or unavailable CUQI array backend: {_backend_name}")
xp = BACKENDS[_backend_name]

def set_backend(name):
    """Switch backend at runtime (experimental)."""
    global xp, _backend_name
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    _backend_name = name
    xp = BACKENDS[name]
    sys.modules[__name__ + '.xp'] = xp

# Expose xp as module attribute
sys.modules[__name__ + '.xp'] = xp

# Re-export all public array functions from the selected backend
for _name in dir(xp):
    if not _name.startswith('_') and callable(getattr(xp, _name)):
        globals()[_name] = getattr(xp, _name)

from ._array import CUQIarray
