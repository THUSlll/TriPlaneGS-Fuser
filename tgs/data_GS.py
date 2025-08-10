# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is part of a project based on:
#   Original project: https://github.com/vast-engineering/xxx
#   Copyright 2024 VAST AI Research, licensed under Apache-2.0

import json
import math
from dataclasses import dataclass, field

import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from tgs.utils.config import parse_structured

from tgs.utils.typing import *

from tgs.utils.camera_utils import JSON_to_camera, cameraList_from_camInfos
from gaussian.gaussian_pcd import load_gaussian_ply
from tgs.models.pointclouds.samplefarthest import remove_outliers_knn
import pdb

@dataclass
class CustomDatasetConfig:
    image_list: Any = ""
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )

    relative_pose: bool = False
    cond_height: int = 512
    cond_width: int = 512
    cond_camera_distance: float = 1.6
    cond_fovy_deg: float = 40.0
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 0.0
    eval_camera_distance: float = 1.6
    eval_fovy_deg: float = 40.0
    n_test_views: int = 120
    num_views_output: int = 120
    only_3dgs: bool = False

    source_path: str = ""
    resolution: int = 1
    data_device: str = "cuda"


class CustomGaussianTransDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: CustomDatasetConfig = parse_structured(CustomDatasetConfig, cfg)
        self.scene_paths = [
                os.path.join(self.cfg.source_path, p)
                for p in os.listdir(self.cfg.source_path)
                if os.path.isdir(os.path.join(self.cfg.source_path, p))
            ]
        
        for i, scene_path in enumerate(self.scene_paths):
            self.scene_paths[i] = os.path.join(self.cfg.source_path, scene_path)

        self.all_scenes = []

        print("Loading Training Data")

        self.train_camera_items = []

        for i, scene_path in enumerate(self.scene_paths):
  
            image_path = None

            # if os.path.exists(os.path.join(scene_path, "seg_images")):
            #     image_path = os.path.join(scene_path, "seg_images")
            # else: 
            #     image_path = os.path.join(scene_path, "images")
            image_path = os.path.join(scene_path, "images")


            train_cameras_list = []
            cond_cameras_list = []
            with open(os.path.join(scene_path, "output", "cameras.json"), 'r') as file:
                data = json.load(file)
                for camera_entry in data:
                    camera = JSON_to_camera(camera_entry, image_path)
                    train_cameras_list.append(camera)
                    # cond_camera_entry = camera_entry.copy() 
                    # camera_entry["width"] = self.cfg.cond_width
                    # camera_entry["height"] = self.cfg.cond_height
                    # cond_camera = JSON_to_camera(cond_camera_entry, image_path)
                    # cond_cameras_list.append(cond_camera)

            train_cameras = cameraList_from_camInfos(train_cameras_list, 1.0, self.cfg, False)
            #cond_cameras = cameraList_from_camInfos(cond_cameras_list, 1.0, self.cfg)
            cond_cameras = train_cameras.copy()

            gaussian_ply = load_gaussian_ply(os.path.join(scene_path, "output", "point_cloud", "iteration_10000", "point_cloud.ply"))
            point_cloud = gaussian_ply.xyz.to('cuda')

            # mask, dist = remove_outliers_knn(point_cloud.unsqueeze(0))
            # opt_point_cloud = point_cloud.unsqueeze(0)[mask].to(self.cfg.data_device)
            opt_point_cloud = point_cloud
            mask = torch.full([point_cloud.size(0)], True, dtype=torch.bool)
            focalY_length = train_cameras[0].image_height * 0.5 / np.tan(0.5 *train_cameras[0].FoVy)
            focalX_length = train_cameras[0].image_width * 0.5 / np.tan(0.5 *train_cameras[0].FoVx)
            
            intrinsic = torch.eye(3, dtype=torch.float32)
            intrinsic[0, 0] = focalX_length
            intrinsic[1, 1] = focalY_length
            intrinsic[0, 2] = train_cameras[0].image_width * 0.5
            intrinsic[1, 2] = train_cameras[0].image_height * 0.5

            c2w_cond = []
            depth_cond = []
            rgb_cond = []
            proj_cond = []
            cam_center_cond = []
            intrinsic_cond = []
            for cam in train_cameras:
                c2w_cond.append(cam.world_view_transform.inverse().transpose(0, 1))
                # depth_cond.append(cam.depth)
                rgb_cond.append(cam.original_image[:3,:,:])
                proj_cond.append(cam.full_proj_transform.transpose(0, 1))
                cam_center_cond.append(cam.camera_center)
                intrinsic_cond.append(intrinsic)
            self.sample_p = int(len(train_cameras)/8)
            scene = {
                "scene_path": scene_path,
                "image_path": image_path,
                "train_cameras": train_cameras,
                "cond_cameras": cond_cameras,
                "scene_name": scene_path.split("/")[-1],
                "gaussian_ply": gaussian_ply,
                "opt_point_cloud": opt_point_cloud,
                "opt_mask": mask,
                "intrinsic": intrinsic,
                "c2w_cond": torch.stack(c2w_cond, dim=0),
                # "depth_cond": torch.stack(depth_cond, dim=0),
                "rgb_cond": torch.stack(rgb_cond, dim=0),
                "proj_cond": torch.stack(proj_cond, dim=0),
                "cam_center_cond": torch.stack(cam_center_cond, dim=0),
                "intrinsic_cond": torch.stack(intrinsic_cond, dim=0),
                "sample_p": self.sample_p
                }
            
            for camera in train_cameras:
                self.train_camera_items.append({"Camera": camera, "Scene_id": i})

            self.all_scenes.append(scene)
            self.sample_offset = 0

    def offset_update(self):
        self.sample_offset += 1
        if self.sample_offset >= self.sample_p:
            self.sample_offset = 0

    def __len__(self):
        return len(self.train_camera_items)


    def __getitem__(self, index):
        
        scene = self.all_scenes[self.train_camera_items[index]["Scene_id"]]
        #pdb.set_trace()
        return {
            "rgb": self.train_camera_items[index]["Camera"].original_image[:3,:,:].unsqueeze(0),
            "mask" : self.train_camera_items[index]["Camera"].original_image[3:4,:,:].unsqueeze(0),
            "c2w" : self.train_camera_items[index]["Camera"].world_view_transform.inverse().transpose(0, 1).unsqueeze(0),
            "proj" : self.train_camera_items[index]["Camera"].full_proj_transform.unsqueeze(0),
            "cam_center": self.train_camera_items[index]["Camera"].camera_center.unsqueeze(0),
            # "depth" : self.train_camera_items[index]["Camera"].depth.unsqueeze(0),

            "c2w_cond": scene["c2w_cond"][self.sample_offset::scene["sample_p"]],
            "rgb_cond": scene["rgb_cond"][self.sample_offset::scene["sample_p"]],
            "proj_cond": scene["proj_cond"][self.sample_offset::scene["sample_p"]],
            "cam_center_cond": scene["cam_center_cond"][self.sample_offset::scene["sample_p"]],
            "intrinsic_cond": scene["intrinsic_cond"][self.sample_offset::scene["sample_p"]],
            # "depth_cond": scene["depth_cond"][self.sample_offset::scene["sample_p"]],

            "xyz": scene["gaussian_ply"].xyz,
            "rots": scene["gaussian_ply"].rots,
            "scales": scene["gaussian_ply"].scales,
            "opacities": scene["gaussian_ply"].opacities,
            "features_dc": scene["gaussian_ply"].features_dc,
            "features_extra": scene["gaussian_ply"].features_extra,
            "opt_point_cloud": scene["opt_point_cloud"],
            "opt_mask": scene["opt_mask"],
            "intrinsic": scene["intrinsic"].unsqueeze(0),
            "image_name": self.train_camera_items[index]["Camera"].image_name,
            "image_path": scene["image_path"]
        }

    def collate(self, batch):
        batch_size = len(batch)
        batch = {k: [sample[k] for sample in batch] for k in batch[0]}
        for k in batch:
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k], dim=0)
        batch.update({"height": batch["intrinsic"][0][0][1, 2] * 2, "width": batch["intrinsic"][0][0][0, 2] * 2})
        return batch