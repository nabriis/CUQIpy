"""
Array-agnostic sparse matrix operations.

This module provides sparse matrix functionality that works across
different array backends without depending on scipy.sparse.
"""

import cuqi.array as xp

def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Create a sparse diagonal matrix from diagonals.
    
    Parameters
    ----------
    diagonals : array_like or list of array_like
        Diagonal values
    offsets : int or list of int, default 0
        Diagonal offsets
    shape : tuple, optional
        Shape of the output matrix
    format : str, optional
        Sparse format (ignored for array backends)
    dtype : dtype, optional
        Data type
        
    Returns
    -------
    array
        Dense matrix representation of the diagonal matrix
    """
    if not isinstance(diagonals, (list, tuple)):
        diagonals = [diagonals]
    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets]
    
    if len(diagonals) != len(offsets):
        raise ValueError("Number of diagonals must match number of offsets")
    
    # Determine shape if not provided
    if shape is None:
        max_offset = max(abs(o) for o in offsets)
        max_diag_len = max(len(xp.asarray(d)) for d in diagonals)
        n = max_diag_len + max_offset
        shape = (n, n)
    
    # Create dense matrix
    if dtype is None:
        dtype = xp.float64
    
    matrix = xp.zeros(shape, dtype=dtype)
    
    # Fill diagonals
    for diag, offset in zip(diagonals, offsets):
        diag_array = xp.asarray(diag)
        
        # Handle scalar vs array diagonals
        if diag_array.ndim == 0:
            # Scalar diagonal - fill the entire diagonal with this value
            if offset >= 0:
                # Upper diagonal
                for i in range(min(shape[0], shape[1] - offset)):
                    if i < shape[0] and i + offset < shape[1]:
                        matrix[i, i + offset] = diag_array
            else:
                # Lower diagonal
                for i in range(min(shape[0] + offset, shape[1])):
                    if i - offset < shape[0] and i < shape[1]:
                        matrix[i - offset, i] = diag_array
        else:
            # Array diagonal
            diag_len = diag_array.shape[0]
            if offset >= 0:
                # Upper diagonal
                n_diag = min(diag_len, shape[0], shape[1] - offset)
                for i in range(n_diag):
                    if i < shape[0] and i + offset < shape[1]:
                        matrix[i, i + offset] = diag_array[i]
            else:
                # Lower diagonal
                n_diag = min(diag_len, shape[0] + offset, shape[1])
                for i in range(n_diag):
                    if i - offset < shape[0] and i < shape[1]:
                        matrix[i - offset, i] = diag_array[i]
    
    return matrix

def eye(n, m=None, k=0, dtype=None, format=None):
    """
    Create a sparse identity matrix.
    
    Parameters
    ----------
    n : int
        Number of rows
    m : int, optional
        Number of columns (default: n)
    k : int, default 0
        Diagonal offset
    dtype : dtype, optional
        Data type
    format : str, optional
        Sparse format (ignored)
        
    Returns
    -------
    array
        Identity matrix
    """
    if m is None:
        m = n
    if dtype is None:
        dtype = xp.float64
    
    return xp.eye(n, m, k, dtype=dtype)

def spdiags(data, diags, m, n, format=None):
    """
    Create a sparse matrix from diagonals.
    
    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : array_like
        Diagonal offsets
    m, n : int
        Matrix dimensions
    format : str, optional
        Sparse format (ignored)
        
    Returns
    -------
    array
        Dense matrix representation
    """
    data = xp.asarray(data)
    diags = xp.asarray(diags)
    
    matrix = xp.zeros((m, n), dtype=data.dtype)
    
    for i, offset in enumerate(diags):
        diag_data = data[i]
        offset = int(offset)
        
        if offset >= 0:
            # Upper diagonal
            for j in range(min(len(diag_data), m, n - offset)):
                if j < m and j + offset < n:
                    matrix[j, j + offset] = diag_data[j]
        else:
            # Lower diagonal
            for j in range(min(len(diag_data), m + offset, n)):
                if j - offset < m and j < n:
                    matrix[j - offset, j] = diag_data[j]
    
    return matrix

def kron(A, B, format=None):
    """
    Kronecker product of two arrays.
    
    Parameters
    ----------
    A, B : array_like
        Input arrays
    format : str, optional
        Sparse format (ignored)
        
    Returns
    -------
    array
        Kronecker product
    """
    A = xp.asarray(A)
    B = xp.asarray(B)
    
    # Get shapes
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    m_a, n_a = A.shape
    m_b, n_b = B.shape
    
    # Create result
    result = xp.zeros((m_a * m_b, n_a * n_b), dtype=A.dtype)
    
    for i in range(m_a):
        for j in range(n_a):
            result[i*m_b:(i+1)*m_b, j*n_b:(j+1)*n_b] = A[i, j] * B
    
    return result

def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices vertically.
    
    Parameters
    ----------
    blocks : list of array_like
        List of matrices to stack
    format : str, optional
        Sparse format (ignored)
    dtype : dtype, optional
        Data type
        
    Returns
    -------
    array
        Vertically stacked matrix
    """
    blocks = [xp.asarray(b) for b in blocks]
    return xp.vstack(blocks)

def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally.
    
    Parameters
    ----------
    blocks : list of array_like
        List of matrices to stack
    format : str, optional
        Sparse format (ignored)
    dtype : dtype, optional
        Data type
        
    Returns
    -------
    array
        Horizontally stacked matrix
    """
    blocks = [xp.asarray(b) for b in blocks]
    return xp.hstack(blocks)

def csc_matrix(data, shape=None, dtype=None):
    """
    Create a compressed sparse column matrix.
    
    For array backends, this returns a dense matrix.
    
    Parameters
    ----------
    data : array_like or tuple
        Matrix data
    shape : tuple, optional
        Matrix shape
    dtype : dtype, optional
        Data type
        
    Returns
    -------
    array
        Dense matrix representation
    """
    if isinstance(data, tuple) and len(data) == 3:
        # (data, (row_ind, col_ind)) format
        values, (rows, cols) = data[0], data[1:]
        if shape is None:
            shape = (max(rows) + 1, max(cols) + 1)
        
        matrix = xp.zeros(shape, dtype=dtype or xp.float64)
        for val, row, col in zip(values, rows, cols):
            matrix[row, col] = val
        return matrix
    else:
        # Dense data
        return xp.asarray(data, dtype=dtype)

def issparse(x):
    """
    Check if array is sparse.
    
    For array backends, always returns False since we use dense matrices.
    
    Parameters
    ----------
    x : array_like
        Input array
        
    Returns
    -------
    bool
        False (arrays are always dense in this implementation)
    """
    return False

# Sparse linear algebra operations
class linalg:
    """Sparse linear algebra operations."""
    
    @staticmethod
    def spsolve(A, b, **kwargs):
        """
        Solve sparse linear system Ax = b.
        
        Parameters
        ----------
        A : array_like
            Coefficient matrix
        b : array_like
            Right-hand side
        **kwargs
            Additional arguments (ignored)
            
        Returns
        -------
        array
            Solution vector
        """
        A = xp.asarray(A)
        b = xp.asarray(b)
        return xp.linalg.solve(A, b)
    
    @staticmethod
    def cg(A, b, x0=None, tol=1e-5, maxiter=None, **kwargs):
        """
        Conjugate gradient solver.
        
        Simple implementation for demonstration.
        For production use, consider more sophisticated solvers.
        
        Parameters
        ----------
        A : array_like
            Coefficient matrix
        b : array_like
            Right-hand side
        x0 : array_like, optional
            Initial guess
        tol : float, default 1e-5
            Tolerance
        maxiter : int, optional
            Maximum iterations
        **kwargs
            Additional arguments (ignored)
            
        Returns
        -------
        tuple
            (solution, info) where info=0 indicates success
        """
        A = xp.asarray(A)
        b = xp.asarray(b)
        n = len(b)
        
        if x0 is None:
            x = xp.zeros(n, dtype=b.dtype)
        else:
            x = xp.asarray(x0)
        
        if maxiter is None:
            maxiter = n
        
        r = b - xp.matmul(A, x)
        p = r.copy()
        rsold = xp.dot(r, r)
        
        for i in range(maxiter):
            Ap = xp.matmul(A, p)
            alpha = rsold / xp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = xp.dot(r, r)
            
            if xp.sqrt(rsnew) < tol:
                return x, 0
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x, 1  # Did not converge