import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

import _msknn as _backend

def msknn(x, y, 
          index, x_index, k,
          ptr_x=None, ptr_y=None, cosine=False, num_workers=1):
    results = _backend.msknn(x, y, index, x_index,
          ptr_x, ptr_y, k, cosine, num_workers)
    return results

