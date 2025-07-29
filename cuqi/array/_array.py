"""
CUQIarray class - Array wrapper with geometry information.

This module provides the CUQIarray class which wraps arrays from any backend
with geometry information for plotting and parameter/function value conversion.
"""

from cuqi.geometry import _DefaultGeometry1D

class CUQIarray:
    """
    Array wrapper that adds geometry information to backend arrays.
    
    This class provides a unified interface for arrays from any backend
    (NumPy, PyTorch, CuPy, JAX) with additional CUQI-specific functionality
    like geometry-aware plotting and parameter/function value conversion.
    """
    
    def __init__(self, input_array, is_par=True, geometry=None):
        """
        Initialize CUQIarray.
        
        Parameters
        ----------
        input_array : array-like
            Input array from any supported backend
        is_par : bool, default True
            Whether array represents parameters (True) or function values (False)
        geometry : Geometry, optional
            Geometry object for spatial information
        """
        # Import here to avoid circular imports
        import cuqi.array as xp
        
        # Convert input to current backend array
        self._array = xp.asarray(input_array)
        self.is_par = is_par
        
        # Validate inputs
        if not is_par and geometry is None:
            raise ValueError("geometry cannot be None when is_par=False")
        if is_par and hasattr(self._array, 'ndim') and self._array.ndim > 1:
            raise ValueError("input_array cannot be multidimensional when is_par=True")
        
        # Set geometry
        if geometry is None:
            self.geometry = _DefaultGeometry1D(grid=len(self._array))
        else:
            self.geometry = geometry
    
    def __repr__(self):
        """String representation of CUQIarray."""
        import cuqi.array as xp
        backend_name = xp.get_backend_name().title()
        
        return (f"CUQIarray: {backend_name} array with geometry\n"
                f"{'=' * 45}\n\n"
                f"Geometry:\n{self.geometry}\n\n"
                f"Is Parameters: {self.is_par}\n\n"
                f"Array:\n{self._array}")
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying array."""
        return getattr(self._array, name)
    
    def __getitem__(self, key):
        """Array indexing."""
        return self._array[key]
    
    def __setitem__(self, key, value):
        """Array item assignment."""
        self._array[key] = value
    
    def __len__(self):
        try:
            return len(self._array)
        except TypeError:
            # If self._array is a scalar, return 1 (or raise informative error)
            import numpy as np
            arr = np.asarray(self._array)
            if arr.shape == ():
                return 1
            raise
    
    def __abs__(self):
        return abs(self._array)
    
    # Mathematical operations - return backend arrays, not CUQIarrays
    def __add__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array + other_array
    
    def __sub__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array - other_array
    
    def __mul__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array * other_array
    
    def __truediv__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array / other_array
    
    def __pow__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array ** other_array
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator, supporting CUQIarray and backend arrays."""
        other_array = other._array if isinstance(other, CUQIarray) else other
        result = self._array @ other_array
        import numpy as np
        arr = np.asarray(result)
        if arr.ndim == 0:
            return arr.item()  # Return as a Python scalar
        elif arr.ndim == 1:
            try:
                return CUQIarray(result, is_par=self.is_par, geometry=self.geometry)
            except Exception:
                return result
        else:
            return result
    
    # Properties that delegate to the underlying array
    @property
    def shape(self):
        """Array shape."""
        return self._array.shape
    
    @property
    def dtype(self):
        """Array data type."""
        return self._array.dtype
    
    @property
    def ndim(self):
        """Number of array dimensions."""
        import cuqi.array as xp
        if hasattr(self._array, 'ndim'):
            return self._array.ndim
        elif hasattr(self._array, 'dim'):  # PyTorch
            return self._array.dim()
        else:
            return len(self._array.shape)
    
    def reshape(self, *args, order=None):
        """Reshape the underlying array. Ignores 'order' kwarg for numpy compatibility."""
        import cuqi.array as xp
        return xp.reshape(self._array, *args)
    
    @property
    def funvals(self):
        """Return array as function values."""
        if self.is_par:
            vals = self.geometry.par2fun(self._array)
        else:
            vals = self._array
        
        # Return new CUQIarray if appropriate
        import cuqi.array as xp
        if hasattr(vals, 'dtype') and vals.dtype == xp.bool_:
            return vals  # Return raw boolean arrays
        else:
            return CUQIarray(vals, is_par=False, geometry=self.geometry)
    
    @property
    def parameters(self):
        """Return array as parameters."""
        if self.is_par:
            vals = self._array
        else:
            vals = self.geometry.fun2par(self._array)
        
        return CUQIarray(vals, is_par=True, geometry=self.geometry)
    
    def to_numpy(self):
        """Convert to NumPy array."""
        import cuqi.array as xp
        return xp.to_numpy(self._array)
    
    def plot(self, plot_par=False, **kwargs):
        """Plot the array data."""
        if plot_par:
            kwargs["is_par"] = True
            return self.geometry.plot(self.parameters, plot_par=plot_par, **kwargs)
        else:
            kwargs["is_par"] = False
            return self.geometry.plot(self.funvals, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert all CUQIarray inputs to their backend arrays
        arrays = [x._array if isinstance(x, CUQIarray) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        # Wrap result as CUQIarray if shape matches self, else return raw
        import numpy as np
        arr = np.asarray(result)
        if arr.shape == self._array.shape:
            return CUQIarray(result, is_par=self.is_par, geometry=self.geometry)
        return result

    def __array_function__(self, func, types, args, kwargs):
        # Convert all CUQIarray args to backend arrays
        arrays = [x._array if isinstance(x, CUQIarray) else x for x in args]
        result = func(*arrays, **kwargs)
        import numpy as np
        arr = np.asarray(result)
        if arr.shape == self._array.shape:
            return CUQIarray(result, is_par=self.is_par, geometry=self.geometry)
        return result

    def __radd__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return other_array + self._array

    def __rsub__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return other_array - self._array

    def __rmul__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return other_array * self._array

    def __rtruediv__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return other_array / self._array

    def __rpow__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return other_array ** self._array

    def __le__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array <= other_array

    def __lt__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array < other_array

    def __ge__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array >= other_array

    def __gt__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array > other_array

    def __eq__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array == other_array

    def __ne__(self, other):
        other_array = other._array if isinstance(other, CUQIarray) else other
        return self._array != other_array
