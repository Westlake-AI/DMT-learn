import torch
import torch.nn as nn
from ..manifolds.poincare import PoincareBall
from ..manifolds.hyperboloid import Hyperboloid


class ToEuclidean(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Euclidean space
    """
    def __init__(self,):
        super(ToEuclidean, self).__init__()

    def forward(self, x):
        return x

class ToPoincare(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """
    def __init__(self,):
        super(ToPoincare, self).__init__()
        self.c = 1
        self.manifold = PoincareBall()

    def forward(self, x):
        z = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return z

class ToLorentz(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Lorentz space
    """
    def __init__(self,):
        super(ToLorentz, self).__init__()
        self.c = 1
        self.manifold = Hyperboloid()

    def forward(self, x):
        z = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return z