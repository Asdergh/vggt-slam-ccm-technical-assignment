import torch
import numpy as np


def min_max_normalization(x: np.ndarray | torch.Tensor, a: float=0, b: float=1):
    return a + (((x - x.min()) * (b - a)) / (x.max() - x.min()))