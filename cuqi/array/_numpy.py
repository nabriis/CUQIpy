# NumPy backend for CUQI array abstraction
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.sparse
import numpy.random

# Array API
array = np.array
asarray = np.asarray
zeros = np.zeros
ones = np.ones
empty = np.empty
full = np.full
arange = np.arange
linspace = np.linspace
reshape = np.reshape
transpose = np.transpose
concatenate = np.concatenate
stack = np.stack
split = np.split
copy = np.copy
# ... (add more as needed)

# Linalg API
linalg = scipy.linalg

# Stats API
stats = scipy.stats

# Sparse API
sparse = scipy.sparse

# Random API
class Random:
    seed = staticmethod(np.random.seed)
    rand = staticmethod(np.random.rand)
    randn = staticmethod(np.random.randn)
    randint = staticmethod(np.random.randint)
    choice = staticmethod(np.random.choice)
    # ... (add more as needed)

random = Random()