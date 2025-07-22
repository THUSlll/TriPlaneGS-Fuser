import torch
import torch.nn as nn
import pdb
from einops import rearrange
import torch.nn.functional as F

def project_depth_to_3d(depth, c2w_cond, K):
    """
    depth: [H, W]
    c2w: [4, 4] (相机到世界矩阵)
    K: [3, 3] (内参矩阵)
    返回: [H, W, 3] (世界坐标系下的3D点)
    """
    # 下采样深度图到1/4分辨率
    # depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # depth = F.max_pool2d(depth, kernel_size=4, stride=4)  # [1, 1, H/4, W/4]
    # depth = depth.squeeze(0).squeeze(0)  # [H/4, W/4]
    
    H, W = depth.shape
    device = depth.device
    depth_mask = depth!=0.0
    depth[depth_mask] += 0.01 
    
    # 调整内参矩阵以适应1/4分辨率
    K = K.clone()
    K[0, 2] = K[0, 2] / 1  # 调整cx
    K[1, 2] = K[1, 2] / 1  # 调整cy
    K[0, 0] = K[0, 0] / 1  # 调整fx
    K[1, 1] = K[1, 1] / 1  # 调整fy
    
    # 生成像素网格
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # 将像素坐标转换为齐次坐标 [u, v, 1]
    uv_homogeneous = torch.stack([u.float(), v.float(), torch.ones_like(u).float()], dim=-1) # [H, W, 3]

    # 计算相机内参矩阵的逆
    K_inv = torch.inverse(K).to(device) # [3, 3]

    # 将像素坐标转换到相机坐标系下的方向向量
    uv_homogeneous_flat = uv_homogeneous.reshape(-1, 3) # [H*W, 3]
    cam_coords_flat = uv_homogeneous_flat @ K_inv.T # [H*W, 3]

    # 乘以深度得到相机坐标系下的 3D 点
    depth_flat = depth.reshape(-1, 1) # [H*W, 1]
    points_cam_flat = cam_coords_flat * depth_flat # [H*W, 3]

    # 将相机坐标系下的 3D 点转换为齐次坐标 [X, Y, Z, 1]
    points_cam_homogeneous_flat = torch.cat([points_cam_flat, torch.ones_like(depth.reshape(-1, 1))], dim=-1) # [H*W, 4]

    # 使用相机到世界坐标系的变换矩阵 c2w_cond 进行变换
    points_world_homogeneous_flat = points_cam_homogeneous_flat @ c2w_cond.T # [H*W, 4]

    # 取前三个分量得到世界坐标系下的 3D 点 [X, Y, Z]
    points_world_flat = points_world_homogeneous_flat[:, :3] # [H*W, 3]

    # 将结果 reshape 回原始图像尺寸 [H, W, 3]
    world_points = points_world_flat.reshape(H, W, 3) # [H, W, 3]

    return world_points[:,:,:3], depth_mask


def project_3d_to_image(world_points, w2c, K, H, W):
    """
    将 3D 世界坐标点投影到多个相机视图的 2D 图像坐标和深度。
    (此版本移除了批处理维度 B，适用于在外部循环处理批次的情况)

    Args:
        world_points: 张量 (Tensor)，形状为 [N, 3]。
        w2c: 张量 (Tensor)，形状为 [M, 4, 4]。
        K: 张量 (Tensor)，形状为 [M, 3, 3]。
        H: 整数 (Integer)，目标图像的高度。
        W: 整数 (Integer)，目标图像的宽度。

    Returns:
        uv_coords: 张量 (Tensor)，形状为 [N, M, 2]。
        depths: 张量 (Tensor)，形状为 [N, M]。
        mask: 张量 (Tensor)，形状为 [N, M]，布尔类型。
    """
    N, _ = world_points.shape
    M, _, _ = w2c.shape
    device = world_points.device
    dtype = world_points.dtype

    # 1. 将世界坐标点转换为齐次坐标 [N, 4]
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    world_points_homogeneous = torch.cat([world_points, ones], dim=-1) # Shape: [N, 4]

    # 2. 执行世界坐标到相机坐标的转换 (使用 einsum)
    #    公式: P_cam_homo[n, m, :] = P_world_homo[n, :] @ w2c[m, :, :].T
    #    einsum 表示: 'nl,mkl->nmk'
    #      n: 点索引, m: 视图索引, l: 齐次坐标维度(4), k: 输出齐次坐标维度(4)
    #      对 l (值为 4) 进行求和
    cam_points_homogeneous = torch.einsum(
        'nl,mkl->nmk',
        world_points_homogeneous, # Shape: [N, 4]
        w2c                     # Shape: [M, 4, 4]
    )
    # 输出 cam_points_homogeneous 的形状应为 [N, M, 4]

    # 3. 获取相机坐标系下的 3D 坐标和深度
    cam_points = cam_points_homogeneous[..., :3] # Shape: [N, M, 3] (Xc, Yc, Zc)
    depths = cam_points_homogeneous[..., 2]     # Shape: [N, M] (Zc - 深度)

    # 4. 使用相机内参 K 将相机坐标点投影到图像平面 (使用 einsum)
    #    公式: P_img_homo[n, m, :] = P_cam[n, m, :] @ K[m, :, :].T
    #    einsum 表示: 'nmk,mhk->nmh'
    #      n: 点索引, m: 视图索引, k: 相机坐标维度(3), h: 图像齐次坐标维度(3)
    #      对 k (值为 3) 进行求和
    uv_homogeneous = torch.einsum(
        'nmk,mhk->nmh',
        cam_points, # Shape: [N, M, 3]
        K           # Shape: [M, 3, 3]
    )
    # 输出 uv_homogeneous 的形状应为 [N, M, 3] (u*Zc, v*Zc, Zc)


    # 5. 执行透视除法获得像素坐标 (u, v)
    epsilon = 1e-8
    depths_safe = depths + epsilon # 避免除以零 Shape: [N, M]

    u = uv_homogeneous[..., 0] / depths_safe # Shape: [N, M]
    v = uv_homogeneous[..., 1] / depths_safe # Shape: [N, M]

    # 将 u, v 堆叠起来
    uv_coords = torch.stack([u, v], dim=-1) # Shape: [N, M, 2]

    # 6. 创建有效点 MASK
    mask = (depths > 0) & \
           (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < W) & \
           (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < H) # Shape: [N, M]

    return uv_coords, depths, mask


def get_grid_coords(world_points: torch.Tensor, grid_size: int) -> tuple:
    """
    获取有点云落入的网格中心坐标和对应的网格索引（支持批量输入）。

    参数:
        world_points (torch.Tensor): 归一化到 [-1, 1] 的点云，形状为 (B, N, 3)。
        grid_size (int): 网格的尺寸。

    返回:
        tuple: 包含两个元素的元组
            - List[torch.Tensor]: 每个样本对应的唯一网格中心坐标列表，每个元素形状为 (M_i, 3)
            - List[torch.Tensor]: 每个样本对应的唯一网格索引列表，每个元素形状为 (M_i, 3)
    """
    # 确保输入点云在 [-1, 1] 范围内
    if not ((world_points >= -1.0) & (world_points <= 1.0)).all():
        raise ValueError("world_points 必须归一化到 [-1, 1] 范围内。")

    # 初始化结果列表
    batch_grid_centers = []
    batch_grid_indices = []

    # 遍历每个样本
    for sample in world_points:  # sample 形状为 (N, 3)
        # 将点云从 [-1, 1] 映射到 [0, grid_size-1]
        scaled_points = ((sample + 1.0) / 2.0) * (grid_size - 1)

        # 取整得到网格索引
        grid_indices = torch.floor(scaled_points).long()

        # 去重，保留唯一的网格索引
        unique_grid_indices = torch.unique(grid_indices, dim=0)

        # 计算网格中心坐标
        cell_width = 2.0 / grid_size  # 每个网格的宽度
        grid_centers = unique_grid_indices * cell_width + cell_width / 2 - 1

        # 添加到结果列表
        batch_grid_centers.append(grid_centers)
        batch_grid_indices.append(unique_grid_indices)

    return batch_grid_centers, batch_grid_indices

def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[:, :, 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[:, :, 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[:, :, 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(2), normed_y.unsqueeze(2), normed_z.unsqueeze(2)], dim=-1)

def denorm_points_bounds(bounded_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    denormed_x = (bounded_points[:, :, 0] + 1) * (scene_bounds[1] - scene_bounds[0]) / 2 + scene_bounds[0]
    denormed_y = (bounded_points[:, :, 1] + 1) * (scene_bounds[3] - scene_bounds[2]) / 2 + scene_bounds[2]
    denormed_z = (bounded_points[:, :, 2] + 1) * (scene_bounds[5] - scene_bounds[4]) / 2 + scene_bounds[4]

    return torch.cat([denormed_x.unsqueeze(2), denormed_y.unsqueeze(2), denormed_z.unsqueeze(2)], dim=-1)

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


class TriplaneAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = query_dim // n_heads
        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"
        
        # 初始化线性层，每个头单独处理
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, query, context):
        # query: [B, H*W, C]
        # context: [B, H*W, L, C]  L是context的长度
        B, HWC, L, C = context.shape
        query = self.norm(query)
        
        # 计算K, V
        k = self.to_k(context)  # [B, H*W, L, C]
        v = self.to_v(context)  # [B, H*W, L, C]
        
        # 重塑为多头形式
        query = query.view(B, HWC, self.n_heads, self.d_head)  # [B, H*W, n_heads, d_head]
        k = k.view(B, HWC, L, self.n_heads, self.d_head)  # [B, H*W, L, n_heads, d_head]
        v = v.view(B, HWC, L, self.n_heads, self.d_head)  # [B, H*W, L, n_heads, d_head]
        
        # 转置以进行批处理
        query = query.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        k = k.permute(0, 3, 1, 2, 4)  # [B, n_heads, H*W, L, d_head]
        v = v.permute(0, 3, 1, 2, 4)  # [B, n_heads, H*W, L, d_head]
        
        # 计算注意力权重
        # 扩展query以匹配k的维度
        query_expanded = query.unsqueeze(3)  # [B, n_heads, H*W, 1, d_head]
        
        # 计算注意力分数
        attn_scores = torch.einsum('bhnld,bhnmd->bhnl', k, query_expanded)  # [B, n_heads, H*W, L]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, H*W, L]
        
        # 应用注意力权重
        attn_output = torch.einsum('bhnl,bhnld->bnhd', attn_weights, v)  # [B, n_heads, H*W, d_head]
        
        # 转置回原始形状并合并头
        attn_output = attn_output.transpose(1, 2)  # [B, H*W, n_heads, d_head]
        attn_output = attn_output.reshape(B, HWC, -1)  # [B, H*W, C]
        
        return attn_output

class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=3, feature_channels=771, num_samples=8):
        super().__init__()
        self.grid_size = grid_size
        self.feature_channels = feature_channels
        self.num_samples = num_samples
        
        # 初始化三平面的query tensor，每个网格单元一个query
        self.query_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(grid_size, grid_size, feature_channels)) for _ in range(3)
        ])
        
        # 初始化注意力模块
        self.attentions_corss_view = TriplaneAttention(
                query_dim=feature_channels * 3, 
                context_dim=feature_channels, 
                n_heads=n_heads
        )

        
    def sample_points_along_axis(self, plane, scene_bounds):
        """
        沿着平面的垂直轴均匀采样点
        Args:
            plane: 'xy' | 'xz' | 'yz'
            scene_bounds: 场景边界
        Returns:
            sampled_points: [num_samples, H, W, 3] 采样点的3D坐标
        """
        H, W = self.grid_size, self.grid_size
        device = self.query_tensors[0].device
        
        # 生成网格坐标
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # 根据平面类型确定采样轴
        if plane == 'xy':
            # 在z轴上采样
            z_samples = torch.linspace(scene_bounds[4], scene_bounds[5], self.num_samples, device=device)
            sampled_points = torch.stack([
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples),
                z_samples.view(1, 1, -1).expand(H, W, -1)
            ], dim=-1)
        elif plane == 'xz':
            # 在y轴上采样
            y_samples = torch.linspace(scene_bounds[2], scene_bounds[3], self.num_samples, device=device)
            sampled_points = torch.stack([
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                y_samples.view(1, 1, -1).expand(H, W, -1),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples)
            ], dim=-1)
        else:  # 'yz'
            # 在x轴上采样
            x_samples = torch.linspace(scene_bounds[0], scene_bounds[1], self.num_samples, device=device)
            sampled_points = torch.stack([
                x_samples.view(1, 1, -1).expand(H, W, -1),
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples)
            ], dim=-1)
            
        return sampled_points
    
    def project_to_other_planes(self, points, plane):
        """
        将3D点投影到其他两个平面
        Args:
            points: [H, W, num_samples, 3] 3D点坐标
            plane: 当前平面
        Returns:
            other_plane_features: [H, W, num_samples*2, C] 其他两个平面的特征
        """
        H, W = points.shape[:2]
        features = []
        
        # 创建平面到索引的映射
        plane_to_idx = {'xy': 0, 'xz': 1, 'yz': 2}
        
        for other_plane in ['xy', 'xz', 'yz']:
            if other_plane == plane:
                continue
                
            # 获取投影坐标
            if other_plane == 'xy':
                coords = points[..., [0, 1]]
            elif other_plane == 'xz':
                coords = points[..., [0, 2]]
            else:  # 'yz'
                coords = points[..., [1, 2]]
                
            # 将坐标映射到网格索引
            grid_coords = (coords * 0.5 + 0.5) * (self.grid_size - 1)
            grid_coords = grid_coords.clamp(0, self.grid_size-1)
            
            # 获取特征
            plane_feat = self.query_tensors[plane_to_idx[other_plane]]  # [H, W, C]
            plane_feat = plane_feat.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            
            # 重塑grid_coords以适应grid_sample的要求
            grid_coords = grid_coords.view(1, H*W*self.num_samples, 1, 2)  # [1, H*W*num_samples, 1, 2]
            
            # 使用grid_sample采样特征
            sampled_feat = F.grid_sample(
                plane_feat,
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # [1, C, H*W*num_samples, 1]
            
            # 重塑采样后的特征
            sampled_feat = sampled_feat.squeeze(0).squeeze(-1)  # [C, H*W*num_samples]
            sampled_feat = sampled_feat.view(self.feature_channels, H, W, self.num_samples)  # [C, H, W, num_samples]
            sampled_feat = sampled_feat.permute(1, 2, 3, 0)  # [H, W, num_samples, C]
            features.append(sampled_feat)
            
        return torch.cat(features, dim=2)  # [H, W, num_samples*2, C]
    
    
    def forward(self, image_features, depths, c2w_cond, intrinsic_cond, pointclouds):
        B, Nv = image_features.shape[:2]
        assert image_features.shape[2] == self.feature_channels, \
            f"特征通道数不匹配: 期望 {self.feature_channels}, 实际 {image_features.shape[2]}"

        tri_planes = {
            'xy': torch.zeros((B, self.grid_size, self.grid_size, self.feature_channels)).cuda(),
            'xz': torch.zeros((B, self.grid_size, self.grid_size, self.feature_channels)).cuda(),
            'yz': torch.zeros((B, self.grid_size, self.grid_size, self.feature_channels)).cuda(),
        }
        
        bounds = torch.tensor([10,-10,10,-10,10,-10], dtype=float, device='cuda')
        
        bounds[0] = min(pointclouds[:,:, 0].min(), bounds[0])
        bounds[1] = max(pointclouds[:,:, 0].max(), bounds[1])
        bounds[2] = min(pointclouds[:,:, 1].min(), bounds[2])
        bounds[3] = max(pointclouds[:,:, 1].max(), bounds[3])
        bounds[4] = min(pointclouds[:,:, 2].min(), bounds[4])
        bounds[5] = max(pointclouds[:,:, 2].max(), bounds[5])

        padding = 0.05
        scene_bounds = (
            bounds[0] - padding * (bounds[1] - bounds[0]),
            bounds[1] + padding * (bounds[1] - bounds[0]),
            bounds[2] - padding * (bounds[3] - bounds[2]),
            bounds[3] + padding * (bounds[3] - bounds[2]),
            bounds[4] - padding * (bounds[5] - bounds[4]),
            bounds[5] + padding * (bounds[5] - bounds[4]),            
        )

        bounded_pointclouds = norm_points_bounds(pointclouds, scene_bounds)
        bounded_grid_coords, grid_indices = get_grid_coords(bounded_pointclouds, self.grid_size)
        grid_coords = denorm_points_bounds(bounded_grid_coords[0].unsqueeze(0), scene_bounds)
        grid_indice = grid_indices[0]

        query_xy = self.query_tensors[0][grid_indice[:,0], grid_indice[:,1]]
        query_xz = self.query_tensors[1][grid_indice[:,0], grid_indice[:,2]]
        query_yz = self.query_tensors[2][grid_indice[:,1], grid_indice[:,2]]
        query = torch.cat([query_xy, query_xz, query_yz], dim=-1)


        uv_coords = []
        for b in range(B):
            w2c = c2w_cond[b].inverse()
            intrinsic = intrinsic_cond[b]
            coords = grid_coords[b]
            uv_coord, depths, mask = project_3d_to_image(coords, w2c, intrinsic, image_features.shape[2], image_features.shape[3])
            features = sample_features_from_image(uv_coord, image_features[b])

            attn_output = self.attentions_corss_view(
                query.unsqueeze(0),  # [1, H*W, C]
                features.unsqueeze(0) # [1, H*W, C]
            )[0]  # [1, H*W, C]

            tri_planes['xy'][b][grid_indice[:,0], grid_indice[:,1]] = attn_output[:, :self.feature_channels]
            tri_planes['xz'][b][grid_indice[:,0], grid_indice[:,2]] = attn_output[:, self.feature_channels:2*self.feature_channels]
            tri_planes['yz'][b][grid_indice[:,1], grid_indice[:,2]] = attn_output[:, 2*self.feature_channels:]

        tri_planes_tokens = torch.cat([tri_planes['xy'].unsqueeze(1), tri_planes['xz'].unsqueeze(1), tri_planes['yz'].unsqueeze(1)], dim=1)
        tri_planes_tokens = rearrange(tri_planes_tokens, 'B N H W C -> B N C H W')
        return tri_planes_tokens, scene_bounds
    

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

def sample_features_from_image(uv_coords, image_features):
    """
    根据 uv_coords 从 image_features 中采样特征。

    参数:
        uv_coords (torch.Tensor): 点云在每个视图中的像素坐标，形状为 (N, M, 2)。
        image_features (torch.Tensor): 图像特征，形状为 (M, C, H, W)。

    返回:
        torch.Tensor: 采样得到的特征，形状为 (N, M, C)。
    """
    # 获取尺寸信息
    N, M, _ = uv_coords.shape  # 点的数量、视图数量
    _, C, H, W = image_features.shape  # 通道数、高度、宽度

    # 将 uv_coords 归一化到 [-1, 1] 范围，以便用于 grid_sample
    uv_coords_normalized = uv_coords.clone()
    uv_coords_normalized[..., 0] = 2 * (uv_coords[..., 0] / (W - 1)) - 1  # u 坐标
    uv_coords_normalized[..., 1] = 2 * (uv_coords[..., 1] / (H - 1)) - 1  # v 坐标

    # 调整 uv_coords_normalized 的形状为 (M, N, 1, 2)，以适应 grid_sample 的输入要求
    uv_coords_normalized = uv_coords_normalized.permute(1, 0, 2).unsqueeze(2)  # (M, N, 1, 2)

    # 使用 grid_sample 对每个视图的特征进行采样
    sampled_features = []
    for i in range(M):
        # 提取当前视图的特征 (C, H, W) 并添加 batch 维度
        feature_map = image_features[i].unsqueeze(0)  # (1, C, H, W)
        # 获取当前视图的网格坐标，并添加 batch 维度
        grid = uv_coords_normalized[i].unsqueeze(0)  # (1, N, 1, 2)
        # 采样特征
        sampled = torch.nn.functional.grid_sample(
            feature_map, grid, mode='bilinear', align_corners=True
        )  # 输出形状 (1, C, N, 1)
        # 调整形状为 (N, C)
        sampled = sampled.squeeze(0).squeeze(2).permute(1, 0)  # (N, C)
        sampled_features.append(sampled)

    # 将所有视图的采样结果堆叠为 (N, M, C)
    sampled_features = torch.stack(sampled_features, dim=1)  # (N, M, C)

    return sampled_features