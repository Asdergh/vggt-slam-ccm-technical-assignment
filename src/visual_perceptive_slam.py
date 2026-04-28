import torch 
import numpy as np
import rerun as rr
import os
import cv2
import open3d
import yaml 

from dataclasses import (dataclass, field)
from torch.utils.data import (IterableDataset, DataLoader)
from typing import (Optional, List, Dict, Any)
from warnings import warn
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from vggt_slam.slam_utils import sort_images_by_number
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector as vec3d 
from .utils import min_max_normalization



@dataclass
class VisualPerceptiveSLAMConfig:
    image_folder: str = "examples/kitchen/images/"
    vis_map: bool = False
    vis_voxel_size: Optional[float] = None
    run_os: bool = False
    vis_flow: bool = False
    log_results: bool = False
    skip_dense_log: bool = False
    log_path: str = "poses.txt"
    max_submap_size: int = 16
    overlapping_window_size: int = 1
    max_loops: int = 1
    min_disparity: float = 50.0
    conf_threshold: float = 25.0
    lc_thres: float = 0.95
    verbose_training: bool=False
    pg_optimizer_steps: int=10
    vggt_model_checkpoints: str="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    sam_model_checkpoints: Optional[str]=None
    device: Optional[str]="cuda"
    use_optf_downsampling: bool=True
    sequences_n: Optional[int]=None

class SequentialImageLoader(IterableDataset):
    def __init__(self, 
                 source: Optional[str]=None,
                 max_submap_size: Optional[int]=12,
                 overlapping_window: Optional[int]=1,
                 dom_to_inclue: Optional[str]="RGB",
                 return_type: Optional[str]="string",
                 min_disparity: Optional[float]=50.0,
                 use_optf_downsampling: bool=True,
                 sequences_n: Optional[int]=None) -> None:
        super(SequentialImageLoader, self).__init__()
        self.max_submap_size = max_submap_size
        self.overlap_w = overlapping_window
        self.dom = dom_to_inclue
        self.return_type = return_type
        self.min_disparity = min_disparity
        self.use_optf_downsampling = use_optf_downsampling
        self.sequences_n = sequences_n
        self.frame_tracker = FrameTracker()

        self._counter = 0
        self._full_sequences_counter = 0
        self._last_images_stack = []
        self.sources = None
        self._optimized = False
        if source is not None:
            self.sources = self.load_data(source)
    
    def load_data(self, source: str):
        if os.path.exists(source):
            sources = sort_images_by_number([
                os.path.join(source, fname)
                for fname in os.listdir(source)
                if (".png" in fname) 
                    or (".jpg" in fname) 
                    and ("depth" not in fname)
            ])
            return sources
        else:
            warn(f"Couldn't find location: {source}")
            
    def __len__(self) -> int:
        assert (self.sources is not None),\
        ("you must load the data first from existed source folder!!!")
        return len(self.sources)
    
    def collate(self, _) -> List:
        for imgf in self.sources[self._counter + 1:]:
            if self.use_optf_downsampling:
                image = cv2.imread(imgf)
                enoght_disp = self.frame_tracker\
                    .compute_disparity(image, 
                                       self.min_disparity, 
                                       False)
                if enoght_disp:
                    self._last_images_stack.append(imgf)
            else:
                self._last_images_stack.append(imgf)
            self._counter += 1
            if (len(self._last_images_stack) > (self.max_submap_size + self.overlap_w)):
                batch = self._last_images_stack
                self._last_images_stack = self._last_images_stack[-self.overlap_w:]
                self._full_sequences_counter += 1
                return batch
    
    def __iter__(self):
        while self._counter < (len(self) - 1):
            if self.sequences_n is not None \
            and self._full_sequences_counter == self.sequences_n:
                break
            yield []

class VisualPerceptiveSLAM:
    def __init__(self, config: VisualPerceptiveSLAMConfig,
                 logger_origin: str="origin",
                 verbose: bool=True):
        self.cfg = config
        dataset = SequentialImageLoader(
            source=self.cfg.image_folder,
            max_submap_size=self.cfg.max_submap_size,
            overlapping_window=self.cfg.overlapping_window_size,
            dom_to_inclue="RGB",
            return_type="string",
            sequences_n=self.cfg.sequences_n
        )
        self.loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 collate_fn=dataset.collate)
        self.setup_pipline()
        if verbose:
            self.logger_origin = logger_origin
            rr.init(logger_origin, spawn=True)
            rr.set_time("time", sequence=0)

    def setup_pipline(self, vggt_ckpt: str=None, sam3_ckpt: str=None):
        self.load_models(vggt_ckpt, sam3_ckpt)
        if hasattr(self, "vggt"):
            self.solver = Solver(lc_thres=self.cfg.lc_thres,
                                 vis_voxel_size=self.cfg.vis_voxel_size,
                                 init_conf_threshold=self.cfg.conf_threshold)
            
    def load_models(self, vggt_ckpt: str=None, sam3_ckpt: str=None):
        vggt_ckpt = vggt_ckpt \
            if vggt_ckpt is not None \
            else self.cfg.vggt_model_checkpoints
        sam3_ckpt = sam3_ckpt \
            if sam3_ckpt is not None \
            else self.cfg.sam_model_checkpoints
        if vggt_ckpt is not None:
            self.vggt = VGGT()
            weights = torch.hub\
                .load_state_dict_from_url(vggt_ckpt)
            self.vggt.load_state_dict(weights)
            self.vggt\
                .eval()\
                .to(torch.bfloat16)\
                .to(self.cfg.device)
        if sam3_ckpt is not None:
            self.sam3 = build_sam3_image_model(checkpoint_path=sam3_ckpt)
            self.sam3_processor = Sam3Processor(self.sam3, confidence_threshold=0.5)
            
    def run_optimization(self):
        self._optimized = True
        if hasattr(self, "vggt"):
            for idx, frames_stack in enumerate(self.loader):
                if frames_stack:
                    predictions = self.solver\
                        .run_predictions(
                            image_names=frames_stack,
                            model=self.vggt,
                            max_loops=self.cfg.max_loops,
                            clip_model=None,
                            clip_preprocess=None
                        )
                    self.solver.add_points(predictions)
                    self.solver.graph.optimize()
                    loop_closure_detected = len(predictions["detected_loops"]) > 0
                    if loop_closure_detected:
                        # rr.set_time("grpah_optimization_steps", sequence=0)
                        for (gpo_idx, submap) in enumerate(self.solver.map\
                            .get_submaps()):
                            self.log_submap(submap, idx + gpo_idx)
                    else:
                        print(f"LOADING NON LOOP CLOSURE SUBMAPS: {idx}")
                        self.log_submap(self.solver\
                                        .map\
                                        .get_latest_submap(), idx)
        else:
            warn("without VGGT pipline would work only in viewer mode")
    
    def log_submap(self, submap: Submap, time: int=None, sequence_name: str="time"):
        if time is not None:
            rr.set_time(sequence_name, sequence=time)
        base_path = f"{self.logger_origin}/Submap_{submap.get_id()}"
        color = np.random.rand(3)
        points_xyz = submap.get_points_in_world_frame(self.solver.graph)
        points_rgb = (submap.get_points_colors() / 255.0)
        extrinsics = submap.get_all_poses_world(self.solver.graph)
        images = submap.get_all_frames()
        for frame_idx in range(extrinsics.shape[0]):
            K = submap.proj_mats[frame_idx, ...]
            world2cam = extrinsics[frame_idx, ...]
            image = images[frame_idx, ...]
            rr.log(f"{base_path}/Frames/frame_{frame_idx}",
                   rr.Transform3D(mat3x3=world2cam[:3, :3],
                                  translation=world2cam[:3, 3]),
                                  rr.Pinhole(image_from_camera=K[:3, :3], color=color),
                                  rr.Image(image.permute(1, 2, 0)))
        rr.log(f"{base_path}/3DReconstraction",
                rr.Points3D(positions=points_xyz,
                            colors=points_rgb,
                            radii=[0.001]))
    
    def get_scene(self) -> Dict[str, Any]:
        if not self._optimized:
            raise RuntimeError("full scene can be generated only with run_optimization() procedure!!")
        all_points = []
        all_colors = []
        all_exts = []
        all_ints = []
        all_frames = []
        for submap in self.solver.map.submaps.values():
            frames = submap.get_all_frames()
            pts = submap.get_points_in_world_frame(self.solver.graph)
            pts_colors = (submap.get_points_colors() / 255.0)
            poses_stack_mat4x4 = submap.get_all_poses_world(self.solver.graph)
            intrinsics_stack_mat3x3 = submap.proj_mats
            all_frames.append(frames)
            all_points.append(pts)
            all_colors.append(pts_colors)
            all_exts.append(poses_stack_mat4x4)
            all_ints.append(intrinsics_stack_mat3x3)
        
        pts = PointCloud()
        pts.points = vec3d(np.concatenate(all_points))
        pts.colors = vec3d(np.concatenate(all_colors))
        all_frames = torch.cat(all_frames).permute(0, 2, 3, 1).cpu().numpy()
        all_exts = np.concatenate(all_exts)
        all_ints = np.concatenate(all_ints)
        return {
            "pts_cloud": pts,
            "frames": all_frames,
            "intrinsics": all_ints,
            "extrinsics": all_exts,
        }
    
    def save_scene(self, path: str):
        os.makedirs(path, exist_ok=True)
        scene_dict = self.get_scene()
        pts_path = os.path.join(path, "point_cloud.ply")
        open3d.io.write_point_cloud(pts_path, scene_dict["pts_cloud"])
        
        frames_path = os.path.join(path, "images")
        os.makedirs(frames_path, exist_ok=True)
        images_paths = []
        for idx, image in enumerate(scene_dict["frames"]):
            image = min_max_normalization(image, 0, 255)
            print(image.shape)
            image = image.astype(np.uint8)
            img_path = os.path.join(frames_path, f"image_{idx}.png")
            cv2.imwrite(img_path, image)
            images_paths.append(img_path)
        
        annotations = os.path.join(path, "annotations.yaml")
        with open(annotations, "w") as file:
            data = {}
            for idx in range(scene_dict["frames"].shape[0]):
                data[f"frame_{idx}"] = {
                    "extrinsics": scene_dict["extrinsics"][idx, ...].tolist(),
                    "intrinsics": scene_dict["intrinsics"][idx, ...].tolist(),
                    "image_path": images_paths[idx],
                }
            yaml.safe_dump(data, file)
            
    


if __name__ == "__main__":
    path = "/home/ram/Desktop/own_projects/vggt-slam-research/modules/vggt-slam/office_loop"
    config = VisualPerceptiveSLAMConfig(image_folder=path)
    # self.cfg = config
    dataset = SequentialImageLoader(
        source=config.image_folder,
        max_submap_size=config.max_submap_size,
        overlapping_window=config.overlapping_window_size,
        dom_to_inclue="RGB",
        return_type="string"
    )
    loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 collate_fn=dataset.collate)
    for (idx, sample) in enumerate(loader):
        # print(idx)
        pass
    # pipline = VisualPerceptiveSLAM(config, verbose=True)
    # pipline.run_optimization()
    # dataset = SequentialImageLoader(source=path)
    # print(len(dataset))
    # loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.collate)
    # try:
    #     for idx, sample in enumerate(loader):
    #         print(idx, sample.keys(), len(sample["images_rgb"]))
    # except BaseException:
    #     pass

            