import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils import weight_norm, spectral_norm

from ptflops import get_model_complexity_info
from thop import profile

import pdb
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm_type = norm
    def forward(self, x):
        x = self.conv(x)
        return x

class NormConvTranspose1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm_type = norm
    def forward(self, x):
        x = self.conv(x)
        return x

class NormConv2d(nn.Module):
    def __init__(self, *args, norm, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return x