import torch
import torch.nn as nn
import pdb
def project_depth_to_3d(depth, c2w_cond, K):
    """
    depth: [H, W]
    c2w: [4, 4] (相机到世界矩阵)
    K: [3, 3] (内参矩阵)
    返回: [H, W, 3] (世界坐标系下的3D点)
    """

    H, W = depth.shape
    device = depth.device
    depth_mask = depth!=0.0
    depth[depth_mask] += 0.01 
    # 生成像素网格
     # 创建像素坐标网格 (u, v)
    # meshgrid 默认创建 (x, y) 对应 (W, H)，所以需要调整或转置
    # 使用 indexing='ij' 可以直接得到 (H, W) 的网格
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # 将像素坐标转换为齐次坐标 [u, v, 1]
    uv_homogeneous = torch.stack([u.float(), v.float(), torch.ones_like(u).float()], dim=-1) # [H, W, 3]

    # 计算相机内参矩阵的逆
    K_inv = torch.inverse(K).to(device) # [3, 3]

    # 将像素坐标转换到相机坐标系下的方向向量
    # (K_inv @ uv_homogeneous^T)^T = uv_homogeneous @ K_inv^T
    # reshape 以进行批处理矩阵乘法
    uv_homogeneous_flat = uv_homogeneous.reshape(-1, 3) # [H*W, 3]
    cam_coords_flat = uv_homogeneous_flat @ K_inv.T # [H*W, 3]

    # 乘以深度得到相机坐标系下的 3D 点
    depth_flat = depth.reshape(-1, 1) # [H*W, 1]
    points_cam_flat = cam_coords_flat * depth_flat # [H*W, 3]

    # u, v = torch.meshgrid(torch.arange(W), torch.arange(H))
    # uv = torch.stack([u, v, torch.ones_like(u)], dim=-1).float().cuda()  # [H, W, 3]
    
    # # 反投影到相机空间
    # K_inv = torch.inverse(K)
    # points_cam_flat = (K_inv @ uv.reshape(-1, 3).T).T  # [H*W, 3]
    # points_cam_flat *= depth.reshape(-1, 1)  # 乘以深度

    # 将相机坐标系下的 3D 点转换为齐次坐标 [X, Y, Z, 1]
    points_cam_homogeneous_flat = torch.cat([points_cam_flat, torch.ones_like(depth.reshape(-1, 1))], dim=-1) # [H*W, 4]

    # 使用相机到世界坐标系的变换矩阵 c2w_cond 进行变换
    # P_world = c2w_cond @ P_cam_homogeneous
    # (c2w_cond @ points_cam_homogeneous_flat^T)^T = points_cam_homogeneous_flat @ c2w_cond^T
    points_world_homogeneous_flat = points_cam_homogeneous_flat @ c2w_cond.T # [H*W, 4]

    # 取前三个分量得到世界坐标系下的 3D 点 [X, Y, Z]
    points_world_flat = points_world_homogeneous_flat[:, :3] # [H*W, 3]

    # 将结果 reshape 回原始图像尺寸 [H, W, 3]
    world_points = points_world_flat.reshape(H, W, 3) # [H, W, 3]

    return world_points[:,:,:3], depth_mask

import torch

def scatter_features_fast(points, mask, features, plane, grid_size):


    coords = (points * 0.5 + 0.5) * (grid_size - 1)
    coords = coords.long().clamp(0, grid_size-1)

    # 展平所有维度
    coords_flat = coords.view(-1, 2)  # [N=640000, 2]

    features_flat = features[:, mask].view(features.shape[0], -1)  # [C=128, N=640000]

    times_flat = torch.ones([1, features_flat.shape[1]], device=features.device)

    # 创建扁平化的输出张量 [C, grid_size * grid_size]
    plane_feat_flat = torch.zeros((features.shape[0], grid_size * grid_size),
                                  device=features.device)
    plane_times_flat = torch.zeros((1, grid_size * grid_size),
                                  device=features.device)
    # 构造 1D 索引张量 [N]
    index_1d = coords_flat[:, 0] * grid_size + coords_flat[:, 1]  # [N=640000]
    # 扩展索引以匹配 src 的形状 [C, N]
    index = index_1d[None, :].expand(features.shape[0], -1)  # [C=128, N=640000]

    index_times = index_1d[None, :].expand(1, -1)  # [1, N=640000]

    # import pdb
    # pdb.set_trace() # 保留此行以进行调试（如果需要）

    # 执行散射操作到扁平化张量
    plane_feat_flat.scatter_reduce_(
        dim=1,
        index=index,
        src=features_flat,
        reduce="mean",
        include_self=False
    )

    plane_times_flat.scatter_reduce_(
        dim=1,
        index=index_times,
        src=times_flat,
        reduce="mean",
        include_self=False
    )

    # 将扁平化张量重新塑形为 [C, grid_size, grid_size]
    plane_feat = plane_feat_flat.view(features.shape[0], grid_size, grid_size)
    plane_times = plane_times_flat.view(1, grid_size, grid_size)

    return plane_feat, plane_times




# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[:, :, 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[:, :, 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[:, :, 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(2), normed_y.unsqueeze(2), normed_z.unsqueeze(2)], dim=-1)
    
def get_plane_coords(world_points: torch.Tensor, mask, plane: str):
    """
    将3D世界坐标投影到指定平面坐标
    Args:
        world_points: [H, W, 3] 或 [N, 3] 的3D坐标
        plane: 'xy' | 'xz' | 'yz'
        scene_bounds: 场景坐标范围，用于归一化
    Returns:
        plane_coords: [..., 2] 对应平面的2D坐标 (归一化到[-1,1])
    """

    
    if plane == 'xy':
        coords = world_points[mask][..., [0, 1]]  # 取XY坐标
    elif plane == 'xz':
        coords = world_points[mask][..., [0, 2]]  # 取XZ坐标
    elif plane == 'yz':
        coords = world_points[mask][..., [1, 2]]  # 取YZ坐标
    else:
        raise ValueError(f"Invalid plane type: {plane}")

    return coords


class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128):
        super().__init__()
        self.grid_size = grid_size
        
    def forward(self, image_features, depths, c2w_cond, intrinsic_cond, scene_bounds= None):
        B, Nv = image_features.shape[:2]

        tri_planes = {
            'xy': torch.zeros((B, image_features.shape[2], self.grid_size, self.grid_size)).cuda(),
            'xz': torch.zeros((B, image_features.shape[2], self.grid_size, self.grid_size)).cuda(),
            'yz': torch.zeros((B, image_features.shape[2], self.grid_size, self.grid_size)).cuda(),
            'xy_times': torch.zeros((B, 1, self.grid_size, self.grid_size)).cuda(),
            'xz_times': torch.zeros((B, 1, self.grid_size, self.grid_size)).cuda(),
            'yz_times': torch.zeros((B, 1, self.grid_size, self.grid_size)).cuda(),
        }
        bounds = torch.tensor([0,0,0,0,0,0], dtype=float ,device= 'cuda')
        for b in range(B):
            for v in range(Nv):
                # 1. 反投影3D点

                world_points, mask = project_depth_to_3d(
                    depths[b, v, 0], c2w_cond[b, v], intrinsic_cond[b, v]
                )

                bounds[0] = min(world_points[mask][:, 0].min(), bounds[0])
                bounds[1] = max(world_points[mask][:, 0].max(), bounds[1])
                bounds[2] = min(world_points[mask][:, 1].min(), bounds[2])
                bounds[3] = max(world_points[mask][:, 1].max(), bounds[3])
                bounds[4] = min(world_points[mask][:, 2].min(), bounds[4])
                bounds[5] = max(world_points[mask][:, 2].max(), bounds[5])


        padding = 0.05  # 扩展5%的范围
        scene_bounds = (
            bounds[0] - padding * (bounds[1] - bounds[0]),
            bounds[1] + padding * (bounds[1] - bounds[0]),
            bounds[2] - padding * (bounds[3] - bounds[2]),
            bounds[3] + padding * (bounds[3] - bounds[2]),
            bounds[4] - padding * (bounds[5] - bounds[4]),
            bounds[5] + padding * (bounds[5] - bounds[4]),            
        )


        for b in range(B):
            for v in range(Nv):
                # 1. 反投影3D点
                
                world_points, mask = project_depth_to_3d(
                    depths[b, v, 0], c2w_cond[b, v], intrinsic_cond[b, v]
                )        
                bounded_points = norm_points_bounds(world_points, scene_bounds)
                # 2. 散射特征到三平面
                for plane in ['xy', 'xz', 'yz']:

                    plane_coords = get_plane_coords(bounded_points, mask, plane)

                    plane_feat, plane_times = scatter_features_fast(
                        plane_coords, mask, image_features[b, v], plane, self.grid_size
                    )
                    tri_planes[plane][b] += plane_feat
                    tri_planes[plane + '_times'][b] += plane_times

        for b in range(B):
            for plane in ['xy', 'xz', 'yz']:
                tri_planes[plane + '_times'][b] = tri_planes[plane + '_times'][b].clamp(min=1e-6)
                tri_planes[plane][b] = tri_planes[plane][b] / tri_planes[plane + '_times'][b]

        return torch.cat([tri_planes['xy'].unsqueeze(1), tri_planes['xz'].unsqueeze(1), tri_planes['yz'].unsqueeze(1)], dim=1), scene_bounds
    

import numpy as np

def save_xyz_tensor_as_ply(tensor, output_file):
    """
    将 (H, W, 3) 的张量（表示 XYZ 坐标）保存为 PLY 点云文件。
    
    参数:
        tensor (numpy.ndarray): 输入张量，形状为 (H, W, 3)，表示 XYZ 坐标。
        output_file (str): 输出的 PLY 文件路径。
    """
    
    # 展平张量为 (H*W, 3)
    point_cloud = tensor.reshape(-1, 3)
    
    # 写入 PLY 文件
    with open(output_file, 'w') as f:
        # 写入头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for point in point_cloud:
            x, y, z = point
            f.write(f"{x} {y} {z}\n")
