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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb

# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[:, 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[:, 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[:, 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(1), normed_y.unsqueeze(1), normed_z.unsqueeze(1), world_points[:, 3:]], dim=-1)
    



class PointTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, feature_dim=256, n_heads=4):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        
        # 初始化注意力模块
        self.attention = PointAttention(feature_dim, grid_size, n_heads)
        
    def forward(self, GS_feats, scene_bounds):
        B, N = GS_feats.shape[:2]
        
        tri_planes = {
            'xy': torch.zeros((B, self.feature_dim, self.grid_size, self.grid_size)).cuda(),
            'xz': torch.zeros((B, self.feature_dim, self.grid_size, self.grid_size)).cuda(),
            'yz': torch.zeros((B, self.feature_dim, self.grid_size, self.grid_size)).cuda(),
        }

        for b in range(B):
            bounded_GS_feats = norm_points_bounds(GS_feats[b], scene_bounds)
            project_points_to_planes(bounded_GS_feats, tri_planes, b, self.attention)

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

def project_points_to_planes(bounded_GS_feats, tri_planes, b, attention):
    """
    将归一化后的点云投影到三平面上，使用注意力机制进行特征聚合
    
    参数:
        bounded_GS_feats: 归一化后的点云特征，形状为 [N, C]
        tri_planes: 已经初始化的三平面字典
        b: batch索引
        attention: 注意力模块
    """
    # 分离坐标和特征
    coords = bounded_GS_feats[:, :3]  # 前三位是坐标
    features = bounded_GS_feats  # 特征
    
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
        
        # 构造1D索引
        index_1d = grid_coords[:, 0] * grid_size + grid_coords[:, 1]

        # 计算每个网格的最大点数
        max_points_per_grid = 0
        for i in range(grid_size * grid_size):
            max_points_per_grid = max(max_points_per_grid, (index_1d == i).sum().item())
        pdb.set_trace()       
        # 使用注意力机制并行处理所有网格
        plane_feat_flat = attention(features, index_1d, max_points_per_grid, plane)
        
        # 重塑为2D平面并更新tri_planes
        tri_planes[plane][b] = plane_feat_flat.view(features.shape[1], grid_size, grid_size)

class PointAttention(nn.Module):
    def __init__(self, feature_dim, grid_size, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = feature_dim // n_heads
        self.grid_size = grid_size
        assert feature_dim % n_heads == 0, "feature_dim must be divisible by n_heads"
        
        # 为每个平面初始化可学习的query
        self.query_xy = nn.Parameter(torch.randn(feature_dim, grid_size, grid_size))
        self.query_xz = nn.Parameter(torch.randn(feature_dim, grid_size, grid_size))
        self.query_yz = nn.Parameter(torch.randn(feature_dim, grid_size, grid_size))
        
        # 初始化线性层
        self.to_k = nn.Linear(feature_dim, feature_dim)
        self.to_v = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features, index_1d, max_points_per_grid, plane):
        # features: [N, C] 其中N是点的数量，C是特征维度
        # index_1d: [N] 表示每个点属于哪个网格
        # max_points_per_grid: 每个网格最多包含的点数
        # plane: 'xy', 'xz', 或 'yz'
        
        N, C = features.shape
        grid_size = self.grid_size
        
        # 选择对应平面的query
        if plane == 'xy':
            query = self.query_xy
        elif plane == 'xz':
            query = self.query_xz
        else:  # yz
            query = self.query_yz
        
        # 归一化特征
        features = self.norm(features)
        
        # 计算K, V
        k = self.to_k(features)  # [N, C]
        v = self.to_v(features)  # [N, C]
        
        # 重塑为多头形式
        k = k.view(-1, self.n_heads, self.d_head)  # [N, n_heads, d_head]
        v = v.view(-1, self.n_heads, self.d_head)  # [N, n_heads, d_head]
        
        # 重塑query为多头形式
        q = query.view(C, -1).T  # [grid_size*grid_size, C]
        q = q.view(-1, self.n_heads, self.d_head)  # [grid_size*grid_size, n_heads, d_head]
        
        # 创建填充后的特征张量
        padded_k = torch.zeros((grid_size * grid_size, max_points_per_grid, self.n_heads, self.d_head), 
                             device=features.device)
        padded_v = torch.zeros_like(padded_k)
        mask = torch.zeros((grid_size * grid_size, max_points_per_grid), 
                          device=features.device, dtype=torch.bool)
        
        # 填充特征
        for i in range(grid_size * grid_size):
            grid_mask = (index_1d == i)
            if grid_mask.sum() > 0:
                n_points = min(grid_mask.sum(), max_points_per_grid)
                padded_k[i, :n_points] = k[grid_mask][:n_points]
                padded_v[i, :n_points] = v[grid_mask][:n_points]
                mask[i, :n_points] = True
        
        # 计算注意力分数
        attn_scores = torch.einsum('gnhd,ghd->gnh', padded_k, q)  # [grid_size*grid_size, max_points, n_heads]
        
        # 应用mask
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # [grid_size*grid_size, max_points, n_heads]
        
        # 应用注意力权重
        out = torch.einsum('gnh,gnhd->ghd', attn_weights, padded_v)  # [grid_size*grid_size, n_heads, d_head]
        out = out.view(grid_size * grid_size, -1)  # [grid_size*grid_size, C]
        
        return out
