import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random as rd

from copy import deepcopy
from dataclasses import (dataclass, field, asdict)
from typing import (Optional, Any, Literal, Tuple, List, Dict, Any, NamedTuple)
from gsplat.strategy import DefaultStrategy
from torch.optimzer import Adam
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector as vec3d 
from .utils import  (get_local_basis,
                     generate_trajectory,
                     getProjectionMatrix,
                     build_scaling_rotation,
                     strip_symmetric,
                     render)
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import (GaussianRasterizaer, 
                                         GaussianRasterizationSettings)






class RenderOutput(NamedTuple):
    render_rgb: torch.Tensor
    render_depth: torch.Tensor
    viewspace_points: torch.Tensor
    visibility_filter: torch.Tensor
    viewspace_points: torch.Tensor

    @property
    def get_max_vis_idx(self) -> int:
        if self.render_rgb.ndim == 3:
            return None
        else:
            max_val = -torch.inf
            max_idx = None
            for idx in range(self.render_rgb.size(0)):
                vis_mask_power = self.visibility_filter.sum()
                if max_val < vis_mask_power:
                    max_val = vis_mask_power
                    max_idx = idx
            return max_idx
                    
                

@dataclass 
class SplatModuleInput:
    pts: Optional[PointCloud]=None
    extrinsics: Optional[np.ndarray | torch.Tensor]=None
    intrinsiscs: Optional[np.ndarray | torch.Tensor]=None
    frames: Optional[np.ndarray | torch.Tensor]=None
    
@dataclass 
class OptConfig:
    pass

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
    opt_config: OptConfig=OptConfig()
    refine_config: RefineConfig=RefineConfig()
    training_config: TrainingConfig=TrainingConfig()
    near: float=1
    far: float=100
    scaling_modifier: float=1.0
    resolutions: Tuple[int]=field(default_factory=(224, 224))
    sh_degree: int=2
    def __post_init__(self):
        if self.training_config.steps < self.refine_config.refine_stop_iter:
            self.refine_config.refine_stop_iter = self.training_config.steps
    
class Camera:
    def __init__(self, uid: int,
                 resolution: Tuple[int, int],
                 viewmatrix: torch.Tensor,
                 K: torch.Tensor,
                 frame: Optional[np.ndarray | torch.Tensor]=None,
                 device: str = "cuda",
                 near: float = 1.0,
                 far: float = 100.0) -> None:

        self.frame = frame \
                    if isinstance(frame, torch.Tensor)\
                    else torch.from_numpy(frame)\
                        .float().permute(2, 0, 1)
        self.K = K.to(self.device)
        self.viewmatrix = viewmatrix.to(device)
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.near = near; self.far = far

        self.projection = getProjectionMatrix(near, far, self.FoVx, self.FoVy).to(device)
        self.full_projection = viewmatrix @ self.projection

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
    
    def get_rays(self, uv_grid: torch.Tensor = None):
        if uv_grid is None:
            w_space = torch.linspace(0, self.resolution[0], steps=self.resolution[0])
            h_space = torch.linspace(0, self.resolution[1], steps=self.resolution[1])
            uv_grid = torch.stack(torch.meshgrid(w_space, h_space, indexing='ij'), dim=-1).reshape(-1, 2)
        else:
            if isinstance(uv_grid, np.ndarray):
                uv_grid = torch.from_numpy(uv_grid).float()
        n = uv_grid.shape[0]
        uv_grid = torch.cat([uv_grid, torch.ones((n, 1), device=uv_grid.device)], dim=1)
        K_inv = torch.linalg.inv(self.K)
        d = (K_inv @ uv_grid.T).T
        d_norm = d / torch.linalg.norm(d, dim=1, keepdim=True)
        return d_norm.to(self.device)

    @property
    def world2cam_rays(self):
        rays = self.get_rays()
        n = rays.shape[0]
        rays_hom = torch.cat([rays, torch.ones((n, 1), device=rays.device)], dim=1)
        rays_w2c = (self.viewmatrix @ rays_hom.T).T[:, :3]
        return rays_w2c

    @property
    def cam2world_rays(self):
        rays = self.get_rays()
        n = rays.shape[0]
        rays_hom = torch.cat([rays, torch.ones((n, 1), device=rays.device)], dim=1)
        Twc_inv = torch.linalg.inv(self.viewmatrix)
        rays_c2w = (Twc_inv @ rays_hom.T).T[:, :3]
        return rays_c2w

    @property
    def camera_center(self):
        return self.viewmatrix[:3, 3]

    
class SplatModule(nn.Module):
    def __init__(self, config: SplatModuleConfig):
        super(SplatModule, self).__init__()
        self.config = config
        self.loss = nn.MSELoss()
    
    def get_covarience(self, scaling_modifier: float=1.0):
        assert (hasattr(self, "scales") and hasattr(self, "quats")), \
        ("To get covarience you first need to load map")
        M = build_scaling_rotation(self.scales * scaling_modifier, self.quats)
        covarience = strip_symmetric(M @ M.transpose(-1, -2))
        return covarience
    
    def get_dict(self, scaling_modifier: float=1.0) -> Dict[str, Any]:
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "features_rgb": self.colors,
            "features_sh": self.features_sh,
            "features": self.features,
            "covarience": self.get_covarience(scaling_modifier)
        }
    
    def load_frames(self, intrinsics: np.ndarray | torch.Tensor,
                    extrinsiscs: np.ndarray | torch.Tensor,
                    frames: np.ndarray | torch.Tensor):
        n = intrinsics.shape[0]
        self.frames = {}
        for idx in range(n):
            camera = Camera(
                uid=idx,
                frame=frames,
                viewmatrix=extrinsiscs[idx, ...],
                K=intrinsics[idx, ...],
                near=self.config.near,
                far=self.config.far,
                resolution=self.config.resolution
            )
            self.frames[idx] = camera
        
        self.idnices = [idx for idx in range(n)]
        self._indices_stack = deepcopy(self.idnices)

    def load_pts_map(self, pts_cloud: PointCloud, opt_config: Optional[Any]=None):
        n = pts_cloud.points.shape[0]
        self.means = nn.Parameter(torch\
                                .from_numpy(pts_cloud.points)\
                                .float())
        self.opacities = nn.Parameter(torch.rand(n, 1))
        self.quats = nn.Parameter(torch.ones((n, 4)))
        self.scales = nn.Parameter(distCUDA2(self.means)\
            .view(-1, 1).repeat(1, 3))
        colors = torch\
                    .from_numpy((pts_cloud.colors / 255.0))\
                    .view(-1, 1, 3)
        features_sh = torch.zeros((n, ((self.sh_degree + 1) ** 2) - 1), 3)
        features = nn.torch.cat([self.colors, self.features_sh], dim=1)
        self.colors = nn.Parameter(colors)
        self.features_sh = nn.Parameter(features_sh)
        self.features = nn.Parameter(features)
        
        self.optimizers = {}
        for attrib in ("means", "opacities", "scales", "colors"):
            opt = Adam([getattr(self, attrib)], lr=(1e-3 
                                                    if opt_config is None 
                                                    else opt_config[attrib]))
            self.optimizers[attrib] = opt
    
    def set_refine_status(self, refine_config: Optional[RefineConfig]):
        self.ref_strategy = DefaultStrategy(**asdict(refine_config))

    def sample_viewpoints(self, n: int=12) -> List[Camera]:
        output = []
        for _ in range(n):
            if not self._indices_stack:
                self._indices_stack = deepcopy(self.idnices)
            idx = rd.cohice(self._indices_stack)
            viewpoint = self.frames[idx]
            output.append(viewpoint)
        return output
    
    def render(self, 
               viewpoints_idx: Optional[List[int]]=None,
               viewpoints: Optional[List[Camera]]=None,
               viewpoints_extrinsics: Optional[torch.Tensor]=None,
               viewpoints_intrinsics: Optional[torch.Tensor]=None,
               scaling_modifier: float=1.0,
               bg_color: Optional[torch.Tensor]=None,
               mask: Optional[torch.Tensor]=None):
        
        batched = {}
        bg_color = bg_color.to(self.device) \
                    if bg_color is not None \
                    else torch.zeros((3, )).to(self.device) 
        assert (viewpoints_idx is not None) \
            or (viewpoints is not None)\
            or ((viewpoints_extrinsics is not None)
                and (viewpoints_intrinsics is not None)), \
            ("""One of inputs types must be passed:
             1) viewpoint_idx: List[int] - indices of current existing viewpoints,
             2) viewpoint: List[Camera] - object with all nececesry attributes
             3) viewpoints_extrinsics/intrinsics: torch.Tensor - batch of extrinsics/intrinsics""")
        if viewpoints is not None:
            pass
        elif viewpoints_idx is not None:
            viewpoints = [self.frames[idx] for idx in viewpoints_idx]
        elif viewpoints_intrinsics and viewpoints_extrinsics:
            viewpoints = [Camera(uid=idx, 
                                 K=K, viewmatrix=viewmatrix,
                                 near=self.config.near,
                                 far=self.config.far)
                                for idx, (K, viewmatrix) \
                                    in enumerate(zip(viewpoints_intrinsics, 
                                                     viewpoints_extrinsics))]
        for (idx, view) in enumerate(viewpoints):
            render_pkg = render(
                viewpoint_camera=view,
                pc=self.get_dict(scaling_modifier),
                bg_color=bg_color,
                override_color=(mask if mask is None else mask[idx, ...]),
                sh_degree=self.sh_deggree
            )
            for (key, value) in render_pkg.items():
                if key not in batched:
                    batched[key] = []
                batched[key].append(value)
        for (key, value) in batched:
            batched[key] = torch.stack(value).to(self.device)
        return RenderOutput(**batched)
    
    def fit(self, input_pkg: SplatModuleInput):

        self.load_map(input_pkg.pts, self.config.opt_cofig)
        if not hasattr(self, "ref_strategy"):
            self.set_refine_status(self.config.refine_config)
        strategy_state = self.ref_strategy.initialize_state()
        for step in range(self.config.train_config.steps):
            viewpoints = self.sample_viewpoints(self.config.train_config.view_batch_size)
            gt_rgb = torch.stack([cam.frame for cam in viewpoints]).to(self.device)
            render_pkg = self.render(viewpoints=viewpoints, scaling_modifier=self.config.scaling_modifier)
            loss = self.loss(render_pkg.render_rgb, gt_rgb)
            
            
        
        
        
        

    
    
    


        

        
        
        

    