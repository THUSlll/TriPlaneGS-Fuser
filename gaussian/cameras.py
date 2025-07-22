#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from tgs.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image
import math
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, image_path, resolution,
                 gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 fixed_depth_path = None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.fixed_depth_path = fixed_depth_path
        self.image_path = image_path
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if image != None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def read_pic(self):

        resolution = (self.image_width, self.image_height)
        image = Image.open(self.image_path)
        resized_image_rgb = PILtoTorch(image, resolution)
        # 将深度插值为与image相同的尺寸


        return resized_image_rgb
    
    def read_depth(self):
        depth_fixed_data = np.load(self.fixed_depth_path)
        try:
            depth_fixed = torch.tensor(depth_fixed_data['depth'], device=self.data_device).float().unsqueeze(0) 
        except:
            depth_fixed = torch.tensor(depth_fixed_data, dtype=torch.float32).unsqueeze(0) 

        depth_fixed = torch.nn.functional.interpolate(
            depth_fixed,  # [1, 1, H, W]
            size=(self.image_height, self.image_width),  # (new_H, new_W)
            mode='bilinear',
            align_corners=True
        ).squeeze(0)  # [1, new_H, new_W]

        return depth_fixed



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

import torch
import numpy as np
import math

# (get_orthographic_projection_matrix 和 create_rotation_translation_from_lookat 函数保持不变)
def get_orthographic_projection_matrix(left, right, bottom, top, near, far, device):
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = -2.0 / (far - near)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(far + near) / (far - near)
    P[3, 3] = 1.0
    return P

def create_rotation_translation_from_lookat(camera_pos_np, look_at_target_np, up_vec_np):
    forward_z = look_at_target_np - camera_pos_np
    forward_z /= np.linalg.norm(forward_z)
    right_x = np.cross(forward_z, up_vec_np)
    right_x /= np.linalg.norm(right_x)
    up_y = np.cross(right_x, forward_z)
    up_y /= np.linalg.norm(up_y)
    R_wc_np = np.array([right_x, up_y, -forward_z])
    T_wc_np = -R_wc_np @ camera_pos_np
    return R_wc_np, T_wc_np

# --- 终极修正版函数：自动计算紧凑视锥体 ---
def get_ortho_camera_params(plane_type, grid_size, scene_range_min_np, scene_range_max_np, data_device="cuda"):
    """
    通过自动计算紧凑视锥体，为高斯溅射生成鲁棒的正交相机参数。
    此版本可解决裁剪问题导致的黑屏。
    """
    # 1. 定义场景包围盒的8个顶点
    min_c = scene_range_min_np
    max_c = scene_range_max_np
    corners = np.array([
        [min_c[0], min_c[1], min_c[2]],
        [max_c[0], min_c[1], min_c[2]],
        [min_c[0], max_c[1], min_c[2]],
        [min_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], min_c[2]],
        [min_c[0], max_c[1], max_c[2]],
        [max_c[0], min_c[1], max_c[2]],
        [max_c[0], max_c[1], max_c[2]],
    ])
    corners = torch.from_numpy(corners).float().to(data_device)
    # 添加齐次坐标 w=1
    corners_hom = torch.cat([corners, torch.ones(8, 1, device=data_device)], dim=1)

    # 2. 根据 plane_type 设置相机位置和姿态
    scene_center_np = (scene_range_min_np + scene_range_max_np) / 2.0
    # 将相机放远一点，确保在场景外
    max_dim = np.max(scene_range_max_np - scene_range_min_np)
    camera_distance_from_center = max_dim * 2.0 
    
    if plane_type == 'xy':
        campos_np = scene_center_np + np.array([0.0, 0.0, camera_distance_from_center])
        up_vec_np = np.array([0.0, 1.0, 0.0])
    elif plane_type == 'xz':
        campos_np = scene_center_np + np.array([0.0, camera_distance_from_center, 0.0])
        up_vec_np = np.array([0.0, 0.0, 1.0])
    elif plane_type == 'yz':
        campos_np = scene_center_np + np.array([camera_distance_from_center, 0.0, 0.0])
        up_vec_np = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError("Invalid plane_type.")

    # 3. 计算视图矩阵 V
    R_np, T_np = create_rotation_translation_from_lookat(campos_np, scene_center_np, up_vec_np)
    R = torch.tensor(R_np, dtype=torch.float32, device=data_device)
    T = torch.tensor(T_np, dtype=torch.float32, device=data_device)
    view_matrix = torch.eye(4, dtype=torch.float32, device=data_device)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = T

    # 4. 【核心步骤】将包围盒顶点变换到相机空间，并计算边界
    # V @ P_world, P_world是 (4, N) 的矩阵, V 是 (4, 4)
    # 所以需要 P_world.T @ V.T, 或者 V @ P_world
    corners_cam_space = (view_matrix @ corners_hom.T).T
    
    # 直接在相机空间中找到 xyz 的最小/最大值
    left = corners_cam_space[:, 0].min()
    right = corners_cam_space[:, 0].max()
    bottom = corners_cam_space[:, 1].min()
    top = corners_cam_space[:, 1].max()
    
    # Z 坐标在相机空间中是负的。near/far需要是正距离。
    # 加一个小的epsilon防止 near/far 太接近或为0。
    near = -corners_cam_space[:, 2].max()
    far = -corners_cam_space[:, 2].min()
    near = max(0.01, near.item()) # 确保 near 是一个正的小数
    
    # 5. 使用计算出的精确边界来创建投影矩阵 P
    projection_matrix = get_orthographic_projection_matrix(left, right, bottom, top, near, far, device=data_device)

    # 6. 构建组合后的 P @ V 矩阵
    full_proj_transform = projection_matrix @ view_matrix

    # 7. 返回与3DGS渲染管线兼容的相机参数字典
    return {
        "image_height": grid_size,
        "image_width": grid_size,
        "tanfovx": 1.0, # 正交投影的占位符
        "tanfovy": 1.0, # 正交投影的占位符
        "viewmatrix": view_matrix,
        "projmatrix": full_proj_transform,
        "sh_degree": 0,
        "campos": torch.tensor(campos_np, dtype=torch.float32, device=data_device),
    }