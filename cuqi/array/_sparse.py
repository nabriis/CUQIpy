"""Sparse array support for CUQIpy array backends."""

import scipy.sparse as sp

# Expose scipy.sparse classes and functions
csr_matrix = sp.csr_matrix
csc_matrix = sp.csc_matrix
coo_matrix = sp.coo_matrix
dok_matrix = sp.dok_matrix
lil_matrix = sp.lil_matrix
dia_matrix = sp.dia_matrix
bsr_matrix = sp.bsr_matrix

# Sparse array creation functions
eye = sp.eye
identity = sp.identity
diags = sp.diags
spdiags = sp.spdiags
block_diag = sp.block_diag
random = sp.random
rand = sp.rand

# Sparse matrix functions
issparse = sp.issparse
isspmatrix = sp.isspmatrix
isspmatrix_csr = sp.isspmatrix_csr
isspmatrix_csc = sp.isspmatrix_csc
isspmatrix_coo = sp.isspmatrix_coo
isspmatrix_dok = sp.isspmatrix_dok
isspmatrix_lil = sp.isspmatrix_lil
isspmatrix_dia = sp.isspmatrix_dia
isspmatrix_bsr = sp.isspmatrix_bsr

# Sparse linear algebra
if hasattr(sp, 'linalg'):
    linalg = sp.linalg
else:
    # Fallback for older scipy versions
    import scipy.sparse.linalg as linalg

# Additional utilities
hstack = sp.hstack
vstack = sp.vstack
kron = sp.kron
kronsum = sp.kronsum
save_npz = sp.save_npz
load_npz = sp.load_npz