import torch
import random
import numpy as np

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def set_seed(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)