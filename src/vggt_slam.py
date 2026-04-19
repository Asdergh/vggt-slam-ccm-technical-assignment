import torch 
import numpy as np
import rerun as rr
import os
from dataclasses import (dataclass, fields)
from torch.utils.data import (Dataset, DataLoader)
from typing import (Optional, List, Dict, Any)
from warnings import warn
from .vggt_slam.solver import Solver
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.models import VGGT
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class VGGTImagesLoader(Dataset):
    def __init__(self, 
                 source: Optional[str]=None,
                 submap_size: Optional[int]=12,
                 overlapping_window: Optional[int]=1,
                 dom_to_inclue: Optional[str]="RGB") -> None:
        super(VGGTImagesLoader, self).__init__()
        self.submap_size = submap_size
        self.overlap_w = overlapping_window
        self.dom = dom_to_inclue

        self._last_images_stack = []
        if source is not None:
            self.sources = self.load_data()
    
    def load_data(self, source: str):
        if os.path.exists(source):
            sources = [os.path.join(fname, source)
                        for fname in os.listidr(source)
                        if (".png" in fname) and ("depth" not in fname)]
            return sources
    
    def __len__(self) -> int:
        assert (hasattr(self, "sources")), \
        (f"call load_data()")
        return (len(self.sources) // self.submap_size) \
            + int(len(self.sources) % self.submap_size)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        assert (hasattr(self, "sources")), \
        (f"call load_data()")
        images_stack = self.sources[idx*self.submap_size: 
                                    (idx + 1)*self.submap_size]
        
        self._last_images_stack += images_stack
        images = load_and_preprocess_images(self._last_images_stack)
        self._last_images_stack = self._last_images_stack[-self.overlap_w:]
        return {"images_rgb": images, "submap_idx": idx}
    
        
    

@dataclass
class VGGTSLAMConfig:
    image_folder: str = "examples/kitchen/images/"
    vis_map: bool = False
    vis_voxel_size: Optional[float] = None
    run_os: bool = False
    vis_flow: bool = False
    log_results: bool = False
    skip_dense_log: bool = False
    log_path: str = "poses.txt"
    submap_size: int = 16
    overlapping_window_size: int = 1
    max_loops: int = 1
    min_disparity: float = 50.0
    conf_threshold: float = 25.0
    lc_thres: float = 0.95
    verbose_training: bool=False
    pg_optimizer_steps: int=10
    models_ckpt_sources: Dict[str, str]={
        "vggt": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "sam": "/home/ram/Downloads/sam3.pt",
    }
    
    

class VGGTSlamPipeline:
    def __init__(self, config: VGGTSLAMConfig,
                 logger_origin: str="origin",
                 verbose: bool=True):
        self.cfg = config
        self.loader = DataLoader(dataset=VGGTImagesLoader(
            source=self.cfg.image_folder,
            submap_size=self.cfg.submap_size,
            overlapping_window=self.cfg.overlapping_window_size,
            dom_to_inclue="RGB"
        ), batch_size=1)
        if self.cfg.models_ckpt_sources:
            models = self.load_models(self.cfg.models_ckpt_sources)
            for (k, v) in models.items():
                setattr(self, k, v)
            if "vggt" in models:
                self.solver = Solver(lc_thres=self.cfg.lc_thres,
                                     vis_voxel_size=self.cfg.vis_voxel_size,
                                     init_conf_threshold=self.cfg.conf_threshold)
            else:
                warn("without VGGT pipline would work only in viewer mode")
        
        if verbose:
            rr.init(logger_origin, spawn=True)
            rr.set_time("time", sequence=0)
              
    def load_models(self, ckpt_sources: Dict[str, str]):
        models = {}
        if "vggt" in ckpt_sources:
            vggt = VGGT()
            vggt_weights = torch.hub.load_state_dict_from_url(ckpt_sources["vggt"])
            vggt.load_state_dict(vggt_weights)
            models["vggt"] = vggt
        if "sam" in ckpt_sources:
            sam = build_sam3_image_model(chekpoint_path=ckpt_sources["sam"], device="cuda")
            sam_predictor = Sam3Processor(sam, device="cuda")
            models["sam"] = sam_predictor
        
        return models
    
    def run_optimization(self):
        if hasattr(self, "vggt"):
            for idx, frames_stack in enumerate(self.loader):
                predictions = self.solver\
                    .run_predictions(
                        images_names=frames_stack,
                        model=self.vggt,
                        max_loops=self.cfg.max_loops
                    )
                self.solver.add_points(predictions)
                self.solver.grpah.optimizer()
                loop_closure_detected = len(predictions["detected_loops"]) > 0
                if loop_closure_detected:
                    self.solver.update_all_submap_vis()
                else:
                    self.solver.update_latest_submap_vis()
        else:
            warn("without VGGT pipline would work only in viewer mode")
            
            
            
        

    
        
        
    

    
    
            
            
            
            
            





        

         
        

        

    