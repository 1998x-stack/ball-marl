"""Explicit compatibility shim; install extras for official PettingZoo compliance."""
import numpy as np
try:
    from gymnasium import spaces
    from pettingzoo.utils.env import ParallelEnv
    HAVE_PETTINGZOO = True
except ImportError:
    HAVE_PETTINGZOO = False
    class ParallelEnv:
        pass
    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype
        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.isfinite(x).all() and (x >= self.low).all() and (x <= self.high).all()
        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
    class _Discrete:
        def __init__(self, n): self.n = n
        def contains(self, x): return isinstance(x, (int, np.integer)) and 0 <= x < self.n
        def sample(self): return np.random.randint(self.n)
    class _MultiBinary:
        def __init__(self, n): self.n = n
        def contains(self, x): return np.asarray(x).shape == (self.n,) and np.isin(x, [0,1]).all()
        def sample(self): return np.random.randint(2, size=self.n, dtype=np.int8)
    class _Dict:
        def __init__(self, spaces): self.spaces = spaces
        def contains(self, obj): return set(obj) == set(self.spaces) and all(s.contains(obj[k]) for k,s in self.spaces.items())
        def sample(self): return {k:v.sample() for k,v in self.spaces.items()}
    class spaces:
        Box, Discrete, MultiBinary, Dict = _Box, _Discrete, _MultiBinary, _Dict
