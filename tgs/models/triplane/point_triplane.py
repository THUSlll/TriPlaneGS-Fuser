import torch
import torch.nn as nn
import pdb

# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[:, 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[:, 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[:, 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(1), normed_y.unsqueeze(1), normed_z.unsqueeze(1), world_points[:, 3:]], dim=-1)
    



class PointTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=4, feature_channels=196):
        super().__init__()
        self.grid_size = grid_size
                # 初始化三平面的query tensor，每个网格单元一个query

        # self.query_tensors = nn.ParameterList([
        #     nn.Parameter(torch.randn(grid_size, grid_size, feature_channels)) for _ in range(3)
        # ])
        
        # # 初始化注意力模块
        # self.attentions_corss_view = TriplaneAttention(
        #         query_dim=feature_channels, 
        #         context_dim=feature_channels, 
        #         n_heads=n_heads
        # )
        
    def forward(self, GS_feats,  scene_bounds):
        B, N = GS_feats.shape[:2]
        
        tri_planes = {
            'xy': torch.zeros((B, GS_feats.shape[2], self.grid_size, self.grid_size)).cuda(),
            'xz': torch.zeros((B, GS_feats.shape[2], self.grid_size, self.grid_size)).cuda(),
            'yz': torch.zeros((B, GS_feats.shape[2], self.grid_size, self.grid_size)).cuda(),
        }

        for b in range(B):

            bounded_GS_feats = norm_points_bounds(GS_feats[b], scene_bounds)
            project_points_to_planes(bounded_GS_feats, tri_planes, b)

        return torch.cat([tri_planes['xy'].unsqueeze(1), tri_planes['xz'].unsqueeze(1), tri_planes['yz'].unsqueeze(1)], dim=1)
    

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

def project_points_to_planes(bounded_GS_feats, tri_planes, b):
    """
    将归一化后的点云投影到三平面上，并考虑透明度权重
    
    参数:
        bounded_GS_feats: 归一化后的点云特征，形状为 [N, C]
                         其中前三位是坐标，第四位是透明度，后面是特征
        tri_planes: 已经初始化的三平面字典，包含'xy'、'xz'、'yz'三个键
        
    返回:
        直接修改传入的tri_planes
    """
    # 分离坐标、透明度和特征
    coords = bounded_GS_feats[:, :3]  # 前三位是坐标
    alpha = torch.sigmoid(bounded_GS_feats[:, 3])    # 第四位是透明度
    features = bounded_GS_feats # 特征
    
    # 对每个平面进行投影
    for plane in ['xy', 'xz', 'yz']:
        # 选择对应的坐标轴
        if plane == 'xy':
            plane_coords = coords[:, [0, 1]]
        elif plane == 'xz':
            plane_coords = coords[:, [0, 2]]
        else:  # yz
            plane_coords = coords[:, [1, 2]]
            
        # 将坐标映射到网格索引
        grid_size = tri_planes[plane][b].shape[-1]
        grid_coords = (plane_coords * 0.5 + 0.5) * (grid_size - 1)
        grid_coords = grid_coords.long().clamp(0, grid_size-1)
        
        # 展平坐标
        flat_coords = grid_coords.view(-1, 2)
        flat_features = features.view(features.shape[1], -1)
        
        # 创建扁平化的输出张量
        plane_feat_flat = torch.zeros((features.shape[1], grid_size * grid_size),
                                    device=features.device)
        
        # 构造1D索引
        index_1d = flat_coords[:, 0] * grid_size + flat_coords[:, 1]
        index = index_1d[None, :].expand(features.shape[1], -1)
        
        # 使用透明度作为权重进行散射操作
        weighted_features = flat_features * alpha[None, :]
        plane_feat_flat.scatter_reduce_(
            dim=1,
            index=index,
            src=weighted_features,
            reduce="sum",
            include_self=False
        )
        
        # 计算权重和
        weight_sum = torch.zeros(grid_size * grid_size, device=features.device)
        weight_sum.scatter_reduce_(
            dim=0,
            index=index_1d,
            src=alpha,
            reduce="sum",
            include_self=False
        )
        
        # 避免除以零
        weight_sum = weight_sum.clamp(min=1e-6)
        
        # 归一化特征
        plane_feat_flat = plane_feat_flat / weight_sum[None, :]
        
        # 重塑为2D平面并更新tri_planes
        tri_planes[plane][b] = plane_feat_flat.view(features.shape[1], grid_size, grid_size)
