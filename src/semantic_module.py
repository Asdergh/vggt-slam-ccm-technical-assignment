import torch
import cv2
import os
import yaml
import clip
import open3d
import torch.nn.functional as F
import torchvision.transformers.functional as Fv
import heapq
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from warnings import warn
from torchvision.utils import make_grid
from typing import (Optional, Any, Dict, List)
from .utils import min_max_normalization


class SemanticModule:
    def __init__(self, 
                 clip_model: Optional[str]=None,
                 loftr_model: Optional[str]=None,
                 device: str="cpu"):
        
        self.device = device
        if (clip_model is not None) \
            and (loftr_model is not None):
            self.load_models()

    def load_models(self, clip: str, loftr: str):
        self.set_clip(clip)

    def set_frames(self, frames: Dict[str | int, Any]):
        self.frames = frames
        if hasattr(self, "clip_model"):
            for (key, frame) in self.frames.items():
                if not hasattr(self, "_resolution"):
                    self._resolution = frame.resolution
                image = self.clip_preprocessor(frame.image)\
                        .unsqueeze(0)
                embedding = self.clip.encod_image(image)
                embedding_normalized = embedding / (embedding.norm(dim=1) + 1e-4)
                frame.semantic_embeddings = embedding_normalized
                self.frames[key] = frame
        else:
            warn("Cannot calculate semantic featuers for iamges without clip model")
    
    def set_clip(self, clip: str):
        (self.clip, self.clip_preprocessor) = clip.load(clip)
        if hasattr(self, "frames"):
            self.set_frames(self.frames)

    def search_text(self, text: str, 
                    k_best: int=3,
                    get_fusion_map: bool=False,
                    track_layers: List[int]=[1, 14, 22],
                    cmap: Optional[str]=None):
        
        if hasattr(self, "clip"):
            txt_embedding = clip.tokenize(text)
            txt_embedding = self.clip.encode_text(txt_embedding)
            txt_embedding_normalized = txt_embedding / (txt_embedding.norm(dim=1) + 1e-4)
            
            candidates = []
            for (idx, frame) in self.frames.items():
                img_embedding = frame.embedding
                dssim = 1 - F.cossine_similarity(txt_embedding_normalized, img_embedding)
                heapq.heappush(candidates, (dssim, idx))
            
            output = {}
            valid_candidates = heapq.nsmallest(k_best, candidates, key=lambda x: x[0])
            valid_candidates = [x[1] for x in valid_candidates]
            output["close_frame_indices"] = valid_candidates
            if get_fusion_map:
                fusion_maps = {}
                activations = {}
                track_hooks = []
                def activation_hook(module, input, output, name: str):
                    if int(name) in track_layers:
                        activations[name] = output

                for (name, module) in self.clip\
                                        .visual.transformer\
                                            .resblocks.named_children():
                    hook = module.registry_forward_hook(
                        lambda module, input, output, name=name: \
                        activation_hook(module, input, output, name))
                    track_hooks.append(hook)

                for idx in valid_candidates:
                    heatmaps = {}
                    frame = self.frames[idx]
                    image = self.clip_preprocessor(frame.image)\
                            .unsqueeze(dim=0)
                    _ = self.clip.encode_image(image)
                    for (key, activation) in activations.items():
                        activation = activation.squeeze()[:-1, :]
                        query = F.interpolate(txt_embedding.view(1, 1, -1), 
                                              activation.size(-1), 
                                              ode="linear")
                        ssim_map = F.interpolate(F\
                            .cosine_similiarity(query, activation)\
                            .view(1, 1, 16, 16), self._resolution, mode="bilinear")
                        ssim_map = min_max_normalization(ssim_map.squeeze())
                        if cmap is not None:
                            ssim_map[ssim_map < ssim_map.mean()] = 0.0
                            ssim_map = getattr(cm, cmap)(ssim_map)
                            ssim_map = (ssim_map[..., :-1] * ssim_map[..., -1, np.newaxis])
                        heatmaps[key] = ssim_map
                    fusion_maps[idx] = heatmaps
                    output["fusion_heatmaps"] = heatmaps
                
                        
                        
                
                
                    
                    
                    
                
        
            

            
            

            
                
        
