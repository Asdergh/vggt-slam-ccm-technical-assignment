import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from typing import (Tuple, Optional, Any, Dict)

class L2Loss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class GaussianKernel(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('kernel', self._create_kernel())
    
    def _create_kernel(self):
        coords = torch.arange(self.kernel_size) - self.kernel_size // 2
        g = torch.exp(-coords**2 / (2 * self.sigma**2))
        kernel = g.outer(g) / (2 * math.pi * self.sigma**2)
        return kernel / kernel.sum()
    
    def forward(self, x):
        _, _, h, w = x.shape
        k = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        k = k.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, k, padding=self.kernel_size//2, groups=x.shape[1])


class DSSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5, c1: float = 0.01**2, c2: float = 0.03**2):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2
        self.gaussian = GaussianKernel(kernel_size, sigma)
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor,
                return_mask: bool=False) -> torch.Tensor:        
        mu_pred = self.gaussian(pred)
        mu_target = self.gaussian(target)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = self.gaussian(pred ** 2) - mu_pred_sq
        sigma_target_sq = self.gaussian(target ** 2) - mu_target_sq
        sigma_pred_target = self.gaussian(pred * target) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)) / \
                   ((mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2))
        
        output = {}
        output["loss"] = (1 - ssim_map).mean()
        if return_mask:
            output["masks"] = {"sigma_pred_sq_mask": sigma_pred_sq,
                            "sigma_target_sq_mask": sigma_target_sq,
                            "sigma_pred_target_mask": sigma_pred_target,
                            "dssim_mask": (1 - ssim_map)}
        return output


class GradientLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sobel_x, self.sobel_y = self._create_sobel(kernel_size)
    
    def _create_sobel(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if size == 1:
            sobel_x = torch.tensor([[[[1.0]]]])
            sobel_y = torch.tensor([[[[1.0]]]])
        elif size == 3:
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        else:
            sobel_x = self._sobel_any_size(size)
            sobel_y = self._sobel_any_size(size).transpose(2, 3)
        
        sobel_x = sobel_x / sobel_x.abs().sum()
        sobel_y = sobel_y / sobel_y.abs().sum()
        return sobel_x, sobel_y
    
    def _sobel_any_size(self, size: int) -> torch.Tensor:
        assert size % 2 == 1, "Kernel size must be odd"
        center = size // 2
        kernel = torch.zeros(1, 1, size, size)
        
        for i in range(size):
            weight = center - abs(i - center)
            kernel[0, 0, i, :center] = -weight
            kernel[0, 0, i, center+1:] = weight
        
        return kernel
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor,
                conf_mask: Optional[torch.Tensor]=None,
                alpha: float=1e-2,
                return_mask: bool=False) -> torch.Tensor:
        self.sobel_x = self.sobel_x.repeat(pred.size(1), 1, 1, 1)
        self.sobel_y = self.sobel_y.repeat(pred.size(1), 1, 1, 1)
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=self.kernel_size//2, groups=pred.shape[1])
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=self.kernel_size//2, groups=pred.shape[1])
        target_grad_x = F.conv2d(target, self.sobel_x, padding=self.kernel_size//2, groups=target.shape[1])
        target_grad_y = F.conv2d(target, self.sobel_y, padding=self.kernel_size//2, groups=target.shape[1])
        
        output = {}
        if conf_mask is None:
            grad_diff_x = pred_grad_x - target_grad_x
            grad_diff_y = pred_grad_y - target_grad_y
            add = (grad_diff_x.abs() + grad_diff_y.abs())
            output["loss"] = add.mean()
            if return_mask:
                output["masks"] = {"grad_diff_x": grad_diff_x,
                                   "grad_diff_y": grad_diff_y,
                                   "summ_mask": add}
        else:
            pred_grad = torch.sqrt((pred_grad_x**2) + (pred_grad_y**2))
            target_grad = torch.sqrt((target_grad_x**2) + (target_grad_y**2))
            global_part = conf_mask * (pred - target)
            local_part = conf_mask * (pred_grad - target_grad)
            result_map = (global_part + local_part) - alpha*torch.log(conf_mask + 1e-4)
            output["loss"] = result_map.mean()
            if return_mask:
                output["masks"] = {"pred_grad": pred_grad,
                                   "target_grad": target_grad,
                                   "global_estimation_mask": global_part,
                                   "local_estimation_mask": local_part,
                                   "confidence_restricted_mask": result_map}
        return output

class VisualLossModule(nn.Module):
    def __init__(self, use_dssim: bool=True,
                 use_gradient: bool=True,
                 use_l2: bool=True,
                 **kwargs) -> None:
        super(VisualLossModule, self).__init__()
        def _check_argument(cls: Any):
            return {k: v 
                    for (k, v) in kwargs.items()
                    if k in cls.__init__.__code__.co_varnames}
            
        self.dssim = DSSIMLoss(**_check_argument(DSSIMLoss)) if use_dssim else None
        self.gradient = GradientLoss(**_check_argument(GradientLoss)) if use_gradient else None
        self.l2 = L2Loss() if use_l2 else None
        self.losses = nn.ModuleDict({k: module 
                                     for k in ["dssim", "gradient", "l2"]
                                     if (module := getattr(self, k)) is not None})
    
    def forward(self, pred: torch.Tensor,
                target: torch.Tensor,
                return_masks: bool=False,
                **kwargs) -> Dict[str, Any]:
        
        output = {}
        for (key, module) in self.losses.items():
            sig_args = inspect.signature(module.forward).parameters.items()
            args = {k: v for (k, v) in sig_args if k in kwargs}
            if "return_masks" in sig_args:
                args.update({"return_masks": return_masks})
            output.update({key: module(pred, target, **args)})
        return output
    


        
            
            

        
        
