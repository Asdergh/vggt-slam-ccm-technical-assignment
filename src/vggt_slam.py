# import torch 
# import numpy as np
# import rerun as rr
# import os
# from dataclasses import (dataclass, fields)
# from torch.utils.data import (Dataset)
# from typing import (Optional, List, Dict, Any)
# from vggt.vggt.utils.load_fn import load_and_preprocess_images
# from vggt.models import VGGT

# class VGGTImagesLoader(Dataset):
#     def __init__(self, 
#                  source: Optional[str]=None,
#                  submap_size: Optional[int]=12,
#                  overlapping_window: Optional[int]=1,
#                  dom_to_inclue: Optional[str]="RGB") -> None:
#         super(VGGTImagesLoader, self).__init__()
#         self.submap_size = submap_size
#         self.overlap_w = overlapping_window
#         self.dom = dom_to_inclue

#         self._last_images_stack = []
#         if source is not None:
#             self.sources = self.load_data()
    
#     def load_data(self, source: str):
#         if os.path.exists(source):
#             sources = [os.path.join(fname, source)
#                         for fname in os.listidr(source)
#                         if (".png" in fname) and ("depth" not in fname)]
#             return sources
    
#     def __len__(self) -> int:
#         assert (hasattr(self, "sources")), \
#         (f"call load_data()")
#         return (len(self.sources) // self.submap_size) \
#             + int(len(self.sources) % self.submap_size)
    
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
        
#         assert (hasattr(self, "sources")), \
#         (f"call load_data()")
#         images_stack = self.sources[idx*self.submap_size: 
#                                     (idx + 1)*self.submap_size]
        
#         self._last_images_stack += images_stack
#         images = load_and_preprocess_images(self._last_images_stack)
#         self._last_images_stack = self._last_images_stack[-self.overlap_w:]
#         return {"images_rgb": images, "submap_idx": idx}
    

# @dataclass
# class VGGTSLAMConfig:
#     image_folder: str = "examples/kitchen/images/"
#     vis_map: bool = False
#     vis_voxel_size: Optional[float] = None
#     run_os: bool = False
#     vis_flow: bool = False
#     log_results: bool = False
#     skip_dense_log: bool = False
#     log_path: str = "poses.txt"
#     submap_size: int = 16
#     overlapping_window_size: int = 1
#     max_loops: int = 1
#     min_disparity: float = 50.0
#     conf_threshold: float = 25.0
#     lc_thres: float = 0.95
#     verbose_training: bool=False
#     pg_optimizer_steps: int=10
#     models_ckpt_sources: Dict[str, str]={
#         "vggt": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
#         "sam": None,
#         "salad": None
#     }
    
    

# class VGGTSlamPipeline:
#     def __init__(self, config: VGGTSLAMConfig,
#                  logger_origin: str="origin"):
#         self.cfg = config
#         self.loader = VGGTImagesLoader(
#             source=self.cfg.image_folder,
#             submap_size=self.cfg.submap_size,
#             overlapping_window=self.cfg.overlapping_window_size,
#             dom_to_inclue="RGB"
#         )
    
#     def load_models(self, ckpt_sources: Dict[str, str]):
#         models = {}
#         if "vggt" in ckpt_sources:
#             vggt = VGGT()
#             vggt_weights = torch.hub.load_state_dict_from_url(ckpt_sources["vggt"])
#             vggt.load_state_dict(vggt_weights)
#             models["vggt"] = vggt
#         if "sam":
#             pass




if __name__ == "__main__":
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    # from segment_anything import (SamPredictor, sam_model_registry)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # model = sam_model_registry["vit_b"](checkpoint="/home/ram/Downloads/sam_vit_b_01ec64(2).pth")
    # predictor = SamPredictor(model)
    sam = build_sam3_image_model(checkpoint_path="/home/ram/Downloads/sam3.pt", device="cuda")
    predictor = Sam3Processor(sam, device="cuda")
    image = "/home/ram/Pictures/Screenshots/Screenshot from 2026-01-01 00-26-15.png"
# image = np.asarray(Image.open(image))
    image = cv2.imread(image)
    image = (image / image.max())
    image = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).to("cuda")
    inference_state = predictor.set_image(image)
    for (k, v) in inference_state.items():
        if isinstance(v, torch.Tensor):
            inference_state[k] = v.to("cuda")
    print(list(inference_state.keys()))
    prediction_output = predictor.set_text_prompt("Table", state=inference_state)
    for (k, v) in prediction_output.items():
        if isinstance(v, torch.Tensor):
            prediction_output[k] = v\
                                    .detach()\
                                    .cpu()
    print(list(prediction_output.keys()))
    (free, total) = torch.cuda.mem_get_info()
    print(prediction_output["masks"].size())
    print(f"Free: {free / (1014 ** 3)}")
    masks = prediction_output["masks"].squeeze()
    rgb_mask = torch.zeros_like(image)
    for (idx, mask) in enumerate(masks):
        rgb_mask[0, torch.where(mask > 0.6, True, False)] = torch.rand((1, ))
        rgb_mask[1, torch.where(mask > 0.6, True, False)] = torch.rand((1, ))
        rgb_mask[2, torch.where(mask > 0.6, True, False)] = torch.rand((1, ))

    print(rgb_mask.size())
    alpha = 0.56
    result = (1 - alpha) * image + alpha * rgb_mask

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    _, axis = plt.subplots(ncols=2)
    axis[0].imshow(result\
                   .cpu().detach()\
                    .permute(1, 2, 0).numpy()[..., ::-1])
    axis[1].imshow(image.cpu().permute(1, 2, 0).numpy()[..., ::-1])
    plt.show()        
        
        

         
        

        

    