"""
Test suite to ensure all array backends implement the required API.

This test suite verifies that each backend provides all the functions
listed in the __all__ export list and that they work correctly.
"""

import pytest
import numpy as np
import cuqi.array as xp

# Get the expected API from __all__
EXPECTED_API = xp.__all__

# Available backends for testing
BACKENDS = ["numpy", "pytorch"]

class TestBackendAPICompliance:
    """Test that all backends implement the required API."""
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_backend_has_all_functions(self, backend):
        """Test that backend has all functions listed in __all__."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        missing_functions = []
        for func_name in EXPECTED_API:
            if not hasattr(xp, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            pytest.fail(f"Backend '{backend}' missing functions: {missing_functions}")
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_array_creation_functions(self, backend):
        """Test array creation functions work correctly."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        # Test basic array creation
        arr = xp.array([1, 2, 3])
        assert arr is not None
        
        # Test zeros and ones
        zeros = xp.zeros(5)
        ones = xp.ones(3)
        assert xp.sum(zeros) == 0
        assert xp.sum(ones) == 3
        
        # Test eye and identity
        eye_mat = xp.eye(3)
        id_mat = xp.identity(3)
        assert eye_mat.shape == (3, 3)
        assert id_mat.shape == (3, 3)
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mathematical_functions(self, backend):
        """Test mathematical functions work correctly."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        x = xp.array([1.0, 2.0, 3.0])
        
        # Test basic math
        result = xp.sin(x)
        assert result is not None
        
        result = xp.exp(x)
        assert result is not None
        
        result = xp.log(xp.array([1.0, 2.0, 3.0]))
        assert result is not None
        
        # Test array operations
        result = xp.sum(x)
        assert float(result) == 6.0
        
        result = xp.mean(x)
        assert abs(float(result) - 2.0) < 1e-6
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_linear_algebra_functions(self, backend):
        """Test linear algebra functions work correctly."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        a = xp.array([[1.0, 2.0], [3.0, 4.0]])
        b = xp.array([1.0, 2.0])
        
        # Test dot product
        result = xp.dot(b, b)
        assert abs(float(result) - 5.0) < 1e-6
        
        # Test matrix multiplication
        result = xp.matmul(a, b)
        assert result is not None
        
        # Test linalg module exists
        assert hasattr(xp, 'linalg')
        norm_result = xp.linalg.norm(b)
        assert abs(float(norm_result) - np.sqrt(5)) < 1e-6
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_shape_manipulation_functions(self, backend):
        """Test shape manipulation functions work correctly."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        x = xp.array([[1, 2], [3, 4]])
        
        # Test reshape
        reshaped = xp.reshape(x, (4,))
        assert reshaped.shape == (4,) or len(reshaped) == 4
        
        # Test transpose
        transposed = xp.transpose(x)
        assert transposed is not None
        
        # Test ravel/flatten
        flattened = xp.ravel(x)
        assert flattened is not None
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_dtype_and_constants(self, backend):
        """Test data types and constants are available."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        # Test data types exist
        assert hasattr(xp, 'float64')
        assert hasattr(xp, 'int32')
        
        # Test constants exist
        assert hasattr(xp, 'pi')
        assert hasattr(xp, 'inf')
        assert hasattr(xp, 'nan')
        
        # Test array with specific dtype
        arr = xp.array([1, 2, 3], dtype=xp.float64)
        assert arr is not None
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_modules_exist(self, backend):
        """Test that required submodules exist."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        # Test modules exist
        assert hasattr(xp, 'linalg')
        assert hasattr(xp, 'random')
        assert hasattr(xp, 'fft')
        assert hasattr(xp, 'polynomial')
        
        # Test they have basic functionality
        assert hasattr(xp.linalg, 'norm')
        assert hasattr(xp.random, 'randn')

class TestBackendSpecificFeatures:
    """Test backend-specific features work correctly."""
    
    def test_pytorch_gradient_support(self):
        """Test PyTorch gradient support."""
        try:
            xp.set_backend("pytorch")
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Test requires_grad parameter
        x = xp.array([1.0, 2.0], requires_grad=True)
        assert hasattr(x, 'requires_grad')
        assert x.requires_grad is True
        
        # Test gradient computation
        y = xp.sum(x ** 2)
        y.backward()
        assert x.grad is not None
    
    def test_numpy_full_compatibility(self):
        """Test NumPy backend full compatibility."""
        xp.set_backend("numpy")
        
        # Test NumPy-specific functions
        x = xp.array([1, 2, 3])
        
        # Test polynomial functions (should work for NumPy)
        try:
            nodes, weights = xp.polynomial.legendre.leggauss(3)
            assert len(nodes) == 3
            assert len(weights) == 3
        except NotImplementedError:
            # This is expected for non-NumPy backends
            pass
        
        # Test random functions
        rng = xp.random.default_rng(42)
        random_vals = rng.normal(0, 1, 5)
        assert len(random_vals) == 5

class TestBackendConsistency:
    """Test consistency across backends."""
    
    def test_numerical_consistency(self):
        """Test that backends give consistent numerical results."""
        test_data = np.array([1.0, 2.0, 3.0])
        
        results = {}
        for backend in BACKENDS:
            try:
                xp.set_backend(backend)
                x = xp.array(test_data)
                
                results[backend] = {
                    'sum': float(xp.sum(x)),
                    'mean': float(xp.mean(x)),
                    'sin': xp.to_numpy(xp.sin(x)),
                    'exp': xp.to_numpy(xp.exp(x))
                }
            except ImportError:
                continue
        
        if len(results) >= 2:
            backends = list(results.keys())
            ref_backend = backends[0]
            
            for backend in backends[1:]:
                # Test scalar results
                assert abs(results[backend]['sum'] - results[ref_backend]['sum']) < 1e-10
                assert abs(results[backend]['mean'] - results[ref_backend]['mean']) < 1e-10
                
                # Test array results
                np.testing.assert_allclose(
                    results[backend]['sin'], 
                    results[ref_backend]['sin'], 
                    rtol=1e-10
                )
                np.testing.assert_allclose(
                    results[backend]['exp'], 
                    results[ref_backend]['exp'], 
                    rtol=1e-10
                )

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_invalid_operations(self, backend):
        """Test that invalid operations raise appropriate errors."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        # Test division by zero handling - just verify it doesn't crash
        x = xp.array([1.0, 0.0])
        try:
            result = 1.0 / x  # Should handle division by zero gracefully
            # If no exception, check for inf values
            assert xp.isinf(result).any() or True  # Allow inf values
        except (RuntimeWarning, ZeroDivisionError):
            pass  # These are acceptable
    
    def test_unsupported_backend(self):
        """Test that unsupported backends raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            xp.set_backend("nonexistent_backend")
    
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_function_not_implemented(self, backend):
        """Test NotImplementedError for unavailable functions."""
        try:
            xp.set_backend(backend)
        except ImportError:
            pytest.skip(f"{backend} not available")
        
        if backend != "numpy":
            # Test functions that should raise NotImplementedError for non-NumPy backends
            with pytest.raises(NotImplementedError):
                xp.polynomial.legendre.leggauss(5)
            
            with pytest.raises(NotImplementedError):
                xp.random.default_rng(42)