import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random as rd
import wandb
import yaml
import cv2
import open3d
import inspect
from torchvision.utils import make_grid
from copy import deepcopy
from dataclasses import (dataclass, field, asdict)
from typing import (Optional, Any, Literal, Tuple, List, Dict, Any, NamedTuple)
from gsplat.strategy import DefaultStrategy
from gsplat.rendering import rasterization
from torch.optim import Adam
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector as vec3d 
from .utils import  (get_local_basis,
                     generate_trajectory,
                     getProjectionMatrix,
                     build_scaling_rotation,
                     strip_symmetric,
                     render,
                     VisualLossModule,
                     min_max_normalization,
                     as_learnable)
from simple_knn._C import distCUDA2
# from diff_gaussian_rasterization import (GaussianRasterizaer, 
#                                          GaussianRasterizationSettings)

                  
@dataclass 
class SplatModuleInput:
    pts: Optional[PointCloud]=None
    extrinsics: Optional[np.ndarray | torch.Tensor]=None
    intrinsiscs: Optional[np.ndarray | torch.Tensor]=None
    frames: Optional[np.ndarray | torch.Tensor]=None
    
@dataclass 
class OptConfig:
    scales_lr: float=1e-2
    opacities_lr: float=1e-2
    means_lr: float=1e-2
    quats_lr: float=1e-2
    colors_lr: float=1e-2
    loss_module_conig: Dict[str, Any]=field(default_factory=lambda: {
        "reduction": "mean",
        "gradient_kernel_size": 3,
        "dssim_kernel_size": 11,
        "sigma": 1.5,
        "c1": 0.01**2,
        "c2": 0.03**2,
    })


@dataclass
class RefineConfig:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 100
    refine_stop_iter: int = 1000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

@dataclass
class TrainingConfig:
    steps: int=1000
    view_batch_size: int=12
    logging_path: Optional[str]=None
    verbose: bool=False

@dataclass
class SplatModuleConfig:
    opt_config: OptConfig=field(default_factory=OptConfig)
    refine_config: RefineConfig=field(default_factory=RefineConfig)
    training_config: TrainingConfig=field(default_factory=TrainingConfig)
    near: float=1
    far: float=10
    scaling_modifier: float=1.0
    resolution: Optional[Tuple[int]]=None
    log_training_process: bool=True
    log_every_step: int=100
    project_name: str="vggt-slam-splat"
    sh_degree: int=2
    device: str="cuda"
    def __post_init__(self):
        if self.training_config.steps < self.refine_config.refine_stop_iter:
            self.refine_config.refine_stop_iter = self.training_config.steps

class RenderOutput(NamedTuple):
    render_rgb: torch.Tensor
    render_depth: torch.Tensor
    meta: Dict[str, Any]

    # @property
    # def get_max_vis_idx(self) -> int:
    #     if self.render_rgb.ndim == 3:
    #         return None
    #     else:
    #         max_val = -torch.inf
    #         max_idx = None
    #         for idx in range(self.render_rgb.size(0)):
    #             vis_mask_power = self.visibility_filter.sum()
    #             if max_val < vis_mask_power:
    #                 max_val = vis_mask_power
    #                 max_idx = idx
    #         return max_idx
        
class Frame:
    def __init__(self, uid: int,
                 resolution: Tuple[int, int],
                 viewmatrix: torch.Tensor,
                 K: torch.Tensor,
                 image: Optional[torch.Tensor]=None,
                 depth: Optional[torch.Tensor]=None,
                 device: str = "cuda",
                 near: float = 1.0,
                 far: float = 100.0,
                 embedding: Optional[torch.Tensor]=None) -> None:

        self.resolution = resolution
        self.image = image 
        self.depth = depth
        self.K = K
        self.viewmatrix = viewmatrix
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.near = near; self.far = far
        self.embedding = embedding

        self.projection = getProjectionMatrix(near, far, self.FoVx, self.FoVy).to(device)
        self.full_projection = self.viewmatrix @ self.projection
        for (k, v) in vars(self).items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                setattr(self, k, v)


    @property
    def tanfov_x(self):
        return self.resolution[0] / (2 * self.K[0, 0])

    @property
    def tanfov_y(self):
        return self.resolution[1] / (2 * self.K[1, 1])

    @property
    def FoVx(self):
        return math.atan(self.tanfov_x)

    @property
    def FoVy(self):
        return math.atan(self.tanfov_y)


    def to(self, device: str):
        self.device = device
        self.K = self.K.to(device)
        self.viewmatrix = self.viewmatrix.to(device)
        self.projection = self.projection.to(device)
        self.full_projection = self.full_projection.to(device)
        return self
    
    def cam2world_projection(self, pix: torch.Tensor):
        homo = torch.cat([pix, torch.ones(pix.size(0), 1)], dim=-1)
        local_pts = self.depth[pix[:, 0], pix[:, 1]] * (torch.linalg.inv(self.K) @ homo.T).T
        local_homo = torch.cat([local_pts, torch.ones(pix.size(0), 1)], dim=-1)
        global_pts = (torch.linalg.inv(self.viewmatrix) @ local_homo.T).T[:, :-1]
        return global_pts

    @property
    def camera_center(self):
        return self.viewmatrix[:3, 3]
    
class SplatModule(nn.Module):
    def __init__(self, config: SplatModuleConfig):
        super(SplatModule, self).__init__()
        self.config = config
        self.opt_config = config.opt_config
        self.training_config = config.training_config
        self.criterion = VisualLossModule(True, True, True)
    
    def get_covarience(self, scaling_modifier: float=1.0):
        assert (hasattr(self, "scales") and hasattr(self, "quats")), \
        ("To get covarience you first need to load map")
        print(self.scales.size())
        M = build_scaling_rotation(self.scales * scaling_modifier, self.quats)
        covarience = strip_symmetric(M @ M.transpose(-1, -2))
        return covarience
    
    def get_dict(self, scaling_modifier: float=1.0, factor: float=1.0) -> Dict[str, Any]:
        return {
            "means": self.means * factor,
            "quats": self.quats,
            "scales": self.scales * scaling_modifier,
            "opacities": self.opacities.squeeze(),
            "colors": self.colors.squeeze(),
            "features_sh": self.features_sh,
            "features": self.features,
            "covarience": self.get_covarience(scaling_modifier)
        }
    
    def load(self, pts_cloud: Optional[PointCloud]=None,
                 intrinsics: Optional[np.ndarray | torch.Tensor]=None,
                 extrinsiscs: Optional[np.ndarray | torch.Tensor]=None,
                 frames: Optional[np.ndarray | torch.Tensor]=None,
                 path: Optional[str]=None):
        assert (path is not None) \
            or ((pts_cloud is not None)\
            and (intrinsics is not None)\
            and (extrinsiscs is not None)), \
            ("""to load map you can ither od it 
             from folder gnerated by VisualPerceptionSLAM 
             or load from specified arguments""")
        if path is None:
            self.load_pts_map(pts_cloud)
            self.load_frames(intrinsics, extrinsiscs, frames)
        else:
            self.load_from_folder(path)

    def load_from_folder(self, path: str):
        def _check_path(path: str):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path
            
        assert os.path.exists(path), \
        (f"couldn't fidn location: {path}")

        ptsf = _check_path(os.path.join(path, "point_cloud.ply"))
        pts = open3d.io.read_point_cloud(ptsf)
        print(pts)
        annotsf = _check_path(os.path.join(path, "annotations.yaml"))
        with open(annotsf, "r") as file:
            annots = yaml.safe_load(file)
        
        _ = _check_path(os.path.join(path, "images"))
        (images, Twc, K) = ([], [], [])
        for frame in annots.values():
            img = min_max_normalization(torch\
                .from_numpy(cv2.imread(frame["image_path"]))\
                .permute(2, 0, 1)\
                .float())
            scale = torch.ones((3, 1))
            if (self.config.resolution is not None) \
                and (img.shape[-2:] != self.config.resolution):
                old_res = torch.Tensor(list(img.shape[-2:]) + [1, ])
                new_res = torch.Tensor(list(self.config.resolution) + [1, ])
                scale = (new_res / old_res).view(3, 1)
            exts = torch.Tensor(frame["extrinsics"])
            ints = torch.Tensor(frame["intrinsics"])
            ints[:3, :3] *= scale
            images.append(img); Twc.append(exts); K.append(ints)

        images = torch.stack(images)
        Twc = torch.stack(Twc)
        K = torch.stack(K)
        self.load_frames(K, Twc, images)
        self.load_pts_map(pts)
        
    def load_frames(self, intrinsics: np.ndarray | torch.Tensor,
                    extrinsiscs: np.ndarray | torch.Tensor,
                    frames: np.ndarray | torch.Tensor):
        n = intrinsics.shape[0]
        self.frames = {}
        for idx in range(n):
            image = frames[idx, ...]
            resolution = image.shape[-2:]\
                            if self.config.resolution is None\
                            else self.config.resolution
            camera = Frame(
                uid=idx,
                frame=image,
                viewmatrix=extrinsiscs[idx, ...],
                K=intrinsics[idx, ...],
                near=self.config.near,
                far=self.config.far,
                resolution=resolution
            )
            self.frames[idx] = camera
        if self.config.resolution is None:
            self._resolution = resolution
        
        self.indices = [idx for idx in range(n)]
        self._indices_stack = deepcopy(self.indices)

    def load_pts_map(self, pts_cloud: PointCloud):
        
        points_xyz = np.asarray(pts_cloud.points)
        points_rgb = np.asarray(pts_cloud.colors)
        n = points_xyz.shape[0]
        self.means = as_learnable(points_xyz)
        self.scales = as_learnable(distCUDA2(self.means.float()).view(-1, 1).repeat(1, 3))
                                    
        colors = min_max_normalization(torch\
                    .from_numpy(points_rgb)\
                    .view(-1, 1, 3))
        features_sh = torch.zeros((n, ((self.config.sh_degree + 1) ** 2) - 1, 3))
        features = torch.cat([colors, features_sh], dim=1)
        self.opacities = as_learnable(torch.rand(n, 1), self.config.device)
        self.quats = as_learnable(torch.ones((n, 4)), self.config.device)
        self.colors = as_learnable(colors, self.config.device)
        self.features_sh = as_learnable(features_sh, self.config.device)
        self.features = as_learnable(features, self.config.device)
        
        self.optimizers = {}
        for attrib in ("means", "opacities", "scales", "colors"):
            opt = Adam([getattr(self, attrib)], lr=(getattr(self.opt_config, f"{attrib}_lr")))
            self.optimizers[attrib] = opt
    
    def optimizer_zero(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    
    def optimizer_step(self):
        for opt in self.optimizers.values():
            opt.step()
    
    def set_refine_status(self, refine_config: Optional[RefineConfig]):
        self.ref_strategy = DefaultStrategy(**asdict(refine_config))

    def sample_viewpoints(self, n: int=12) -> List[Frame]:
        output = []
        for _ in range(n):
            if not self._indices_stack:
                self._indices_stack = deepcopy(self.indices)
            idx = rd.cohice(self._indices_stack)
            viewpoint = self.frames[idx]
            output.append(viewpoint)
        return output
    
    def render(self, 
               viewpoints_idx: Optional[List[int]]=None,
               viewpoints: Optional[List[Frame]]=None,
               viewpoints_extrinsics: Optional[torch.Tensor]=None,
               viewpoints_intrinsics: Optional[torch.Tensor]=None,
               scaling_modifier: float=1.0,
               xyz_factor: float=1.0,
               bg_color: Optional[torch.Tensor]=None,
               mask: Optional[torch.Tensor]=None):
        
        bg_color = bg_color.to(self.config.device) \
                    if bg_color is not None \
                    else torch.zeros((3, )).to(self.config.device) 
        assert (viewpoints_idx is not None) \
            or (viewpoints is not None)\
            or ((viewpoints_extrinsics is not None)
                and (viewpoints_intrinsics is not None)), \
            ("""One of inputs types must be passed:
             1) viewpoint_idx: List[int] - indices of current existing viewpoints,
             2) viewpoint: List[Frame] - object with all nececesry attributes
             3) viewpoints_extrinsics/intrinsics: torch.Tensor - batch of extrinsics/intrinsics""")
        (exts, ints) = ([], [])
        if viewpoints is not None:
            for cam in viewpoints:
                exts.append(cam.viewmatrix)
                exts.append(cam.K)
            exts = torch.stack(exts)
            ints = torch.stack(ints)
        elif viewpoints_idx is not None:
            for idx in viewpoints_idx:
                exts.append(self.frames[idx].viewmatrix)
                ints.append(self.frames[idx].K)
            exts = torch.stack(exts)
            ints = torch.stack(ints)
        elif viewpoints_intrinsics and viewpoints_extrinsics:
            exts = viewpoints_extrinsics
            ints = viewpoints_intrinsics
        
        exts = exts.float().to(self.config.device)
        ints = ints.float().to(self.config.device)
        rasterization_pkg = {key: value.float() 
                  for (key, value) in self.get_dict(scaling_modifier, xyz_factor).items() 
                  if key in inspect.signature(rasterization).parameters}
        for (k, v) in rasterization_pkg.items():
            print(k, v.size(), v.dtype)

        resolution = self.config.resolution\
                        if self.config.resolution is not None\
                        else self._resolution
        (render_rgb, render_depth, meta) = rasterization(
            **rasterization_pkg,
            Ks=ints[:, :3, :3],
            viewmats=exts,
            width=resolution[0],
            height=resolution[1],
            near_plane=self.config.near,
            far_plane=self.config.far
        )
        return RenderOutput(render_rgb, render_depth, meta)
    
    def fit(self, input_pkg: SplatModuleInput):

        self.load_map(input_pkg.pts)
        if not hasattr(self, "ref_strategy"):
            self.set_refine_status(self.config.refine_config)
        strategy_state = self.ref_strategy.initialize_state()

        for step in range(self.training_config.steps):
            self.optimizer_zero()
            viewpoints = self.sample_viewpoints(self.training_config.view_batch_size)
            gt_rgb = torch.stack([cam.frame for cam in viewpoints]).to(self.config.device)
            render_pkg = self.render(viewpoints=viewpoints, scaling_modifier=self.config.scaling_modifier)
            criterion_pkg = self.criterion(render_pkg.render_rgb, gt_rgb)

            scalars = {}
            masks = {}
            for (k, v) in criterion_pkg.items():
                if isinstance(v, Dict):
                    scalars[f"{k}_loss"] = v["loss_scalar"]
                    for (mask_name, values) in v["masks"].items():
                        values = min_max_normalization(values)
                        values = make_grid(values)\
                            .cpu().detach().numpy()
                        masks[f"{k}_{mask_name}"] = wandb.Image(values)

            self.ref_strategy.step_pre_backward(self.get_dict(), self.optimizers, strategy_state, step, render_pkg.meta)
            loss = sum(list(scalars.values())).backward()
            self.ref_strategy.step_post_backward(self.get_dict(), self.optimizers, strategy_state, step, render_pkg.meta)
            self.optimizer_step()
            if self.config.log_training_process \
            and (step % self.config.log_every_step == 0):
                wandb.log(scalars | masks | {"total_loss": loss, "epoch": step})
        wandb.finish()
    

if __name__ == "__main__":
    # from .visual_perceptive_slam import (VisualPerceptiveSLAM,
    #                                     VisualPerceptiveSLAMConfig)
    # path = "/home/ram/Desktop/own_projects/vggt-slam-research/modules/vggt-slam/office_loop"
    # config = VisualPerceptiveSLAMConfig(image_folder=path, sequences_n=1)
    # pipline = VisualPerceptiveSLAM(config, verbose=True)
    # pipline.run_optimization()
    # pipline.save_scene("/home/ram/Desktop/own_projects/vggt-slam-research/scene_path")
    
    path = "/home/ram/Desktop/own_projects/vggt-slam-research/scene_path"
    splat_config = SplatModuleConfig()
    splat_module = SplatModule(splat_config)
    splat_module.load(path=path)
    
    random_idx = rd.sample(splat_module.indices, 12)
    render_pkg1 = splat_module.render(random_idx, xyz_factor=1e+1)
    render_pkg2 = splat_module.render(random_idx, xyz_factor=1.0)
    rgb1 = render_pkg1.render_rgb.permute(0, 3, 1, 2)
    rgb1 = make_grid(rgb1).permute(1, 2, 0).cpu().detach().numpy()
    rgb2 = render_pkg2.render_rgb.permute(0, 3, 1, 2)
    rgb2 = make_grid(rgb2).permute(1, 2, 0).cpu().detach().numpy()

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    _, axis = plt.subplots(nrows=2)
    axis[0].imshow(rgb1)
    axis[1].imshow(rgb2)
    plt.show()

    
    
    
    
        


                        
                    
                
                    
                
            
            
            
        
        
        
        

    
    
    


        

        
        
        

    