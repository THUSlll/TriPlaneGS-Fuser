import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import tgs
from tgs.models.image_feature import UNet, add_positional_encoding, flatten_spatial, add_plucker_channels
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *
from tgs.utils.loss_def import l1_loss, ssim
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pdb
from tgs.models.renderer import Camera
import math

from diff_gaussian_rasterization import GaussianRasterizer
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.data_GS import CustomGaussianTransDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Render Depth")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    parser.add_argument("--exp_name", default='test', type=str, help="exp name, used to save argument and result")
    parser.add_argument("--source_path", default='', type=str, help="source path")
    args, extras = parser.parse_known_args()
    # pdb.set_trace
    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    # pdb.set_trace
    cfg.data.source_path = args.source_path

    # checkpoint = torch.load(model_path, map_location="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'])


    dataset = CustomGaussianTransDataset(cfg.data)
    dataloader = DataLoader(dataset,
                        batch_size=1, 
                        num_workers=0,
                        shuffle=True,
                        collate_fn=dataset.collate,
                    )
    print('load Gaussian data done.')

    # save_dir = os.path.join("exp", args.exp_name)
    # os.makedirs(save_dir, exist_ok=True)

    save_dir = "/home/u202320081001061/TriplaneGaussian/data/Fuse_dataset/Lego/0/depth_npy"
    data_size = dataset.__len__()

        # #训练阶段进度条
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"render Epoch")):  # [[新增]] 
        torch.cuda.empty_cache()
        batch = todevice(batch)
        viewpoint_camera = Camera.from_c2w(batch['c2w'][0,0], batch['intrinsic'][0,0], batch['height'], batch['width'])
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(batch['xyz'], dtype=batch['xyz'].dtype, requires_grad=True, device='cuda') + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means3D = batch['xyz'][0]
        means2D = screenspace_points[0]
        shs = torch.cat([batch['features_dc'], batch['features_extra']], dim =-1)[0].transpose(1, 2)
        opacity = torch.sigmoid(batch['opacities'][0])
        scales = torch.exp(batch['scales'][0])
        rotations = torch.nn.functional.normalize(batch['rots'][0])

        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
        # pdb.set_trace()
        # alpha_mask = batch['mask'].permute(0, 1, 3, 4, 2).squeeze(1).squeeze(3)
        # msk = (alpha_mask < 1) & (alpha_mask > 0)
        # alpha_mask[msk] = 1
        # depth = depth * alpha_mask

        depth_np = depth.detach().cpu().numpy()  # 假设 depth 是 [H, W] 的张量
        name = batch['image_name'][0].split('.')[0]
        np.savez_compressed(os.path.join(batch["image_path"][0], name + '_gaussian.npz'), depth=depth_np)
        # a,b = 0,0
        # min = 100
        # for i in range(800):
        #     for j in range(800):
        #         if depth_np[0,i,j]!=0 and depth_np[0,i,j]<min:
        #             min = depth_np[0,i,j]
        #             a, b=i, j

        # for i in range(a-4, a +4):
        #     for j in range(b-4, b+4):
        #         depth_np[0,i,j] = depth_np.max()

        # depth_np =( depth_np - depth_np.min()) / ( depth_np.max() - depth_np.min()) 
        # depth_np = (depth_np * 255).astype('uint8')
        # depth_save = Image.fromarray(depth_np[0])
        # depth_path = '/'.join(batch["image_path"][0].split('/')[:-1])
        # depth_dir = os.path.join(depth_path, 'depth')
        # os.makedirs(depth_dir, exist_ok=True)
        # img_path = os.path.join(depth_dir, name + 'depth.jpg')
        # depth_save.save(img_path)
        # # pdb.set_trace()