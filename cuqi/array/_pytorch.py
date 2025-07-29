# PyTorch backend for CUQI array abstraction
import torch
import torch.linalg
import torch.special
import torch.sparse

# Array API
array = torch.tensor
asarray = torch.as_tensor
zeros = torch.zeros
ones = torch.ones
empty = torch.empty
full = torch.full
arange = torch.arange
linspace = torch.linspace
reshape = torch.reshape
transpose = torch.transpose
concatenate = torch.cat
stack = torch.stack
split = torch.split
copy = torch.clone
# ... (add more as needed)

# Linalg API
linalg = torch.linalg

# Stats API (stub, to be expanded)
class Stats:
    # TODO: Implement or map to torch.special or custom fallbacks
    pass
stats = Stats()

# Sparse API (stub, to be expanded)
sparse = torch.sparse

# Random API
class Random:
    def __init__(self):
        self._generator = torch.Generator()
    def seed(self, seed):
        self._generator.manual_seed(seed)
    def rand(self, *size, device=None):
        return torch.rand(*size, generator=self._generator, device=device)
    def randn(self, *size, device=None):
        return torch.randn(*size, generator=self._generator, device=device)
    def randint(self, low, high=None, size=None, device=None):
        if size is None:
            size = ()
        return torch.randint(low, high, size, generator=self._generator, device=device)
    # ... (add more as needed)
random = Random()