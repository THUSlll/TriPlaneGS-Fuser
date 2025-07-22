import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from pytorch3d import io
from pytorch3d.renderer import (
    PointsRasterizationSettings, 
    PointsRasterizer)
from pytorch3d.ops import knn_points
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import cv2

from tgs.utils.typing import *
from tqdm import tqdm
ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]

def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)

def points_projection(points: Float[Tensor, "B Np 3"],
                    c2ws: Float[Tensor, "B Nv 4 4"],
                    intrinsics: Float[Tensor, "B Nv 3 3"],
                    local_features: Float[Tensor, "B Nv C H W"],
                    # Rasterization settings
                    raster_point_radius: float = 0.0075,  # point size
                    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                    bin_size: int = 0):
    
    B, Nv, C, H, W = local_features.shape
    device = local_features.device
    raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )
    Np = points.shape[1]
    R = raster_settings.points_per_pixel

    w2cs = torch.inverse(c2ws)
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(w2cs.shape[0], -1).to(device)
#############################################

    R_flat = w2cs[:, :, :3, :3].view(B*Nv, 3, 3)
    tvec_flat = w2cs[:, :, :3, 3].view(B*Nv, 3)    
    intrinsics_flat = intrinsics.view(B*Nv, 3, 3) 

    points = points.unsqueeze(1)  # (B, 1, NP, 3) <button class="citation-flag" data-index="10">
    points = points.repeat(1, Nv, 1, 1)  # (B, NV, NP, 3) <button class="citation-flag" data-index="5">
    points = points.view(B*Nv, Np, 3)
#################################################
    cameras = cameras_from_opencv_projection(R_flat, tvec_flat, intrinsics_flat, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]


    local_features = local_features.view(B*Nv, C, H, W)
    # Reshape local features to (B*Nv, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Nv * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Nv, Np, C)
    local_features_proj = local_features_proj.sum(dim=1)

    return local_features_proj

def points_projection_with_score_optimized(
    points: Float[Tensor, "B Np 3"],
    c2ws: Float[Tensor, "B Nv 4 4"],
    intrinsics: Float[Tensor, "B Nv 3 3"],
    local_features: Float[Tensor, "B Nv C H W"],
    ply_score: Float[Tensor, "B Np Nv"], # 你的可见性分数
    # 这些参数在这里不再需要，或者可以用于验证点是否在图像内
    raster_point_radius: float = 0.0075,
    raster_points_per_pixel: int = 1,
    bin_size: int = 0
):
    B, Nv, C, H, W = local_features.shape
    Np = points.shape[1]
    device = local_features.device

    # 1. 计算世界到相机变换矩阵 (w2cs)
    w2cs = torch.inverse(c2ws) # (B, Nv, 4, 4)

    # 2. 准备点云数据以适应批量操作
    # (B, Np, 3) -> (B, 1, Np, 3) -> (B, Nv, Np, 3)
    points_expanded = points.unsqueeze(1).repeat(1, Nv, 1, 1)

    # 将 3D 点转换为齐次坐标 (B, Nv, Np, 4)
    points_homogeneous = torch.cat([points_expanded, torch.ones_like(points_expanded[..., :1])], dim=-1)

    # 3. 将 3D 点从世界坐标系转换到相机坐标系 (B, Nv, Np, 3)
    # (B, Nv, Np, 4) @ (B, Nv, 4, 4).transpose(-1, -2) -> (B, Nv, Np, 4)
    # 或者更直接的矩阵乘法: (B, Nv, 4, 4) @ (B, Nv, 4, Np) -> (B, Nv, 4, Np) -> (B, Nv, Np, 4)
    # 这里需要广播机制，或者手动展平/重复
    
    # 将 w2cs 和 points_homogeneous 展平以便矩阵乘法
    w2cs_flat = w2cs.view(B * Nv, 4, 4)
    points_homogeneous_flat = points_homogeneous.view(B * Nv, Np, 4).transpose(1, 2) # (B*Nv, 4, Np)

    # 执行变换 (B*Nv, 4, 4) @ (B*Nv, 4, Np) -> (B*Nv, 4, Np)
    points_camera_homogeneous_flat = torch.matmul(w2cs_flat, points_homogeneous_flat)
    points_camera_homogeneous = points_camera_homogeneous_flat.transpose(1, 2).view(B, Nv, Np, 4)
    
    # 提取相机坐标系下的 3D 点 (x_c, y_c, z_c)
    points_camera = points_camera_homogeneous[..., :3] # (B, Nv, Np, 3)
    
    # 注意：在相机坐标系中，Z 轴通常指向前方，代表深度。
    # 确保只处理 Z > 0 的点（在相机前方）
    depths = points_camera[..., 2:3] # (B, Nv, Np, 1)
    
    # 4. 将相机坐标系下的点投影到 2D 像素坐标 (u, v)
    intrinsics_flat = intrinsics.view(B * Nv, 3, 3)
    points_camera_flat = points_camera.view(B * Nv, Np, 3).transpose(1, 2) # (B*Nv, 3, Np)

    # (B*Nv, 3, 3) @ (B*Nv, 3, Np) -> (B*Nv, 3, Np)
    points_pixel_homogeneous_flat = torch.matmul(intrinsics_flat, points_camera_flat)
    points_pixel_homogeneous = points_pixel_homogeneous_flat.transpose(1, 2).view(B, Nv, Np, 3)

    # 齐次坐标除以深度 Z_c
    # (B, Nv, Np, 2)
    # 避免除以零和处理负深度（点在相机后面）
    z_c = points_pixel_homogeneous[..., 2:3].clamp(min=1e-6) # 避免除以零
    pixel_coords_uv = points_pixel_homogeneous[..., :2] / z_c

    # 5. 归一化像素坐标到 [-1, 1] 范围，以适应 grid_sample
    # u_norm = (u / (W - 1)) * 2 - 1
    # v_norm = (v / (H - 1)) * 2 - 1
    # grid_sample 期望 (x, y) 而不是 (u, v)，所以交换列
    # x 对应宽度 W, y 对应高度 H
    
    # 原始像素坐标 (u, v)
    u_coords = pixel_coords_uv[..., 0] # (B, Nv, Np)
    v_coords = pixel_coords_uv[..., 1] # (B, Nv, Np)

    # 归一化到 [-1, 1] 范围
    # (x_grid, y_grid) -> x_grid corresponds to width (u), y_grid corresponds to height (v)
    grid_x = (u_coords / (W - 1)) * 2 - 1 # (B, Nv, Np)
    grid_y = (v_coords / (H - 1)) * 2 - 1 # (B, Nv, Np)

    # 将其组合成 (B, Nv, Np, 2) 形状，grid_sample 期望 (N, H_out, W_out, 2)
    # 对于点采样，可以理解为 (B * Nv, Np, 1, 2) 或者 (B * Nv, 1, Np, 2)
    # grid_sample 的 grid 维度是 (N, H_out, W_out, 2)
    # 我们希望对每个点进行采样，可以把 Np 看作 H_out * W_out
    grid = torch.stack([grid_x, grid_y], dim=-1) # (B, Nv, Np, 2)

    # 处理越界和负深度点：将它们设置为无效采样点（例如，归一化坐标设为非常大的值，或在后续掩码）
    # 或者直接在 `grid_sample` 之前通过 `padding_mode='zeros'` 或 `mode='nearest'` 来处理
    # 更好的做法是创建一个 mask
    # 确保点在图像边界内且深度为正
    # (B, Nv, Np)
    valid_points_mask = (u_coords >= 0) & (u_coords <= W - 1) & \
                        (v_coords >= 0) & (v_coords <= H - 1) & \
                        (depths.squeeze(-1) > 0) # 确保 Z > 0

    # 6. 使用 grid_sample 采样特征
    # local_features: (B, Nv, C, H, W)
    # grid: (B, Nv, Np, 2)
    # grid_sample 期望 input 为 (N, C, H_in, W_in)，grid 为 (N, H_out, W_out, 2)
    # 我们可以展平 B 和 Nv，然后对每个视图进行采样
    
    local_features_flat = local_features.view(B * Nv, C, H, W)
    grid_flat = grid.view(B * Nv, 1, Np, 2) # (B*Nv, H_out=1, W_out=Np, 2)
    
    # out_features_flat: (B*Nv, C, 1, Np)
    # align_corners=False 避免半像素偏移
    out_features_flat = F.grid_sample(local_features_flat, grid_flat, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 重新塑形回 (B, Nv, C, Np)
    out_features = out_features_flat.squeeze(2).view(B, Nv, C, Np).permute(0, 1, 3, 2) # (B, Nv, Np, C)

    # 7. 应用有效点掩码：将无效采样点的特征置为零
    # valid_points_mask 维度是 (B, Nv, Np)
    # out_features 维度是 (B, Nv, Np, C)
    out_features = out_features * valid_points_mask.unsqueeze(-1)

    # 8. 应用可见性分数进行加权求和
    # ply_score: (B, Np, Nv)
    # 为了广播，调整 ply_score 维度为 (B, Nv, Np, 1)
    # out_features: (B, Nv, Np, C)
    
    # 将 ply_score 转换为 (B, Nv, Np, 1) 以便与 out_features 广播
    weighted_features = out_features * ply_score.permute(0, 2, 1).unsqueeze(-1) # (B, Nv, Np, C)

    # 对 Nv 维度求和 (B, Np, C)
    local_features_proj = weighted_features.sum(dim=1)
    # 计算权重和并避免除以零
    weight_sum = ply_score.sum(dim=-1, keepdim=False).clamp(min=1e-6).unsqueeze(-1)
    # 进行平均
    local_features_proj = local_features_proj / weight_sum

    return local_features_proj

def points_projection_with_score_chunked(
    points: Float[Tensor, "B Np 3"],
    c2ws: Float[Tensor, "B Nv 4 4"],
    intrinsics: Float[Tensor, "B Nv 3 3"],
    local_features: Float[Tensor, "B Nv C H W"],
    ply_score: Float[Tensor, "B Np Nv"],
    chunk_size: int = 500000,  # 每次处理的点数
    raster_point_radius: float = 0.0075,
    raster_points_per_pixel: int = 1,
    bin_size: int = 0
):
    """
    分块处理版本，避免内存溢出
    """
    B, Nv, C, H, W = local_features.shape
    Np = points.shape[1]
    device = local_features.device
    
    # 如果点数不多，直接使用原函数
    if Np <= chunk_size:
        return points_projection_with_score_optimized(
            points, c2ws, intrinsics, local_features, ply_score,
            raster_point_radius, raster_points_per_pixel, bin_size
        )
    
    tqdm.write(f"点数过多({Np})，使用分块处理，块大小: {chunk_size}")
    
    # 分块处理
    results = []
    for i in range(0, Np, chunk_size):
        end_idx = min(i + chunk_size, Np)
        
        # 提取当前块的数据
        points_chunk = points[:, i:end_idx, :]  # (B, chunk_size, 3)
        ply_score_chunk = ply_score[:, i:end_idx, :]  # (B, chunk_size, Nv)
        
        # 处理当前块
        chunk_result = points_projection_with_score_optimized(
            points_chunk, c2ws, intrinsics, local_features, ply_score_chunk,
            raster_point_radius, raster_points_per_pixel, bin_size
        )  # (B, chunk_size, C)
        
        results.append(chunk_result)
        
        # 清理内存
        del points_chunk, ply_score_chunk, chunk_result
        torch.cuda.empty_cache()
    
    # 合并结果
    local_features_proj = torch.cat(results, dim=1)  # (B, Np, C)
    
    return local_features_proj

def find_nearest_neighbors_with_distance_and_indices_pytorch3d(point_cloud1, point_cloud2):
    """
    遍历第一个点云，为每个点找到在第二个点云中最近的点的索引和距离。
    使用 pytorch3d 的 knn_points。

    Args:
        point_cloud1 (torch.Tensor): 第一个点云，形状为 (N, D)。
        point_cloud2 (torch.Tensor): 第二个点云，形状为 (M, D)。

    Returns:
        tuple:
            - torch.Tensor: 包含point_cloud1中每个点在point_cloud2中最近邻的索引。形状为 (N,)。
            - torch.Tensor: 包含point_cloud1中每个点到其最近邻的距离。形状为 (N,)。
    """

    # distances 形状: (Batch, N, K)
    # indices 形状: (Batch, N, K)
    # _  (可选): 最近邻点的坐标，我们这里不需要

    distances, indices, _ = knn_points(point_cloud1, point_cloud2, K=1)

    # 移除 Batch 和 K 维度，得到 (N,) 形状的索引和距离
    nearest_neighbor_distances = distances.squeeze(0).squeeze(1)
    nearest_neighbor_indices = indices.squeeze(0).squeeze(1)

    return nearest_neighbor_indices, nearest_neighbor_distances

def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform