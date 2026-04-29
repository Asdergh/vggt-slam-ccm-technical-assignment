import torch
import numpy as np
import torch.nn as nn


def min_max_normalization(x: np.ndarray | torch.Tensor, a: float=0, b: float=1):
    return a + (((x - x.min()) * (b - a)) / (x.max() - x.min()))

def as_learnable(x: torch.Tensor | np.ndarray, device: str="cuda"):
    x = x \
        if isinstance(x, torch.Tensor)\
        else torch.from_numpy(x)
    return nn.Parameter(x.requires_grad_(True).to(device))
    