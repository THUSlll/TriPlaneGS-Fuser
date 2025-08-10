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

import torch




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
        # context: [B, H*W, C]
        B, L, C = query.shape

        query = self.norm(query)
        
        # 计算K, V
        k = self.to_k(context)  # [B, H*W, C]
        v = self.to_v(context)  # [B, H*W, C]
        
        # 重塑为多头形式
        query = query.view(B, L, self.n_heads, self.d_head)  # [B, H*W, n_heads, d_head]
        k = k.view(B, L, self.n_heads, self.d_head)  # [B, H*W, n_heads, d_head]
        v = v.view(B, L, self.n_heads, self.d_head)  # [B, H*W, n_heads, d_head]
        
        # 转置以进行批处理
        query = query.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        k = k.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        v = v.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        
        # 计算每个头的注意力
        attn_score = query * k  # [B, n_heads, H*W, d_head]
        attn_score = attn_score / (self.d_head ** 0.5)
        
        # 转置回原始形状并合并头
        attn_score = attn_score.transpose(1, 2)  # [B, H*W, n_heads, d_head]
        attn_score = attn_score.reshape(B, L, C)  # [B, H*W, C]
        v = v.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        v = v.reshape(B, L, C)  # [B, H*W, C]
        
        return attn_score, v

class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=4, feature_channels=196, num_samples=8, num_layer=3):
        super().__init__()
        self.grid_size = grid_size
        self.feature_channels = feature_channels
        self.num_samples = num_samples
        self.num_layers = num_layer

        # 初始化三平面的query tensor
        self.query_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(grid_size, grid_size, feature_channels)) for _ in range(3)
        ])
        
        # 修改为 ModuleList
        self.attentions_corss_view = nn.ModuleList([
            TriplaneAttention(
                query_dim=feature_channels, 
                context_dim=feature_channels, 
                n_heads=n_heads
            ) for _ in range(num_layer)
        ])

        self.attentions_cross_plane = nn.ModuleList([
            TriplaneAttention(
                query_dim=feature_channels, 
                context_dim=feature_channels, 
                n_heads=n_heads
            ) for _ in range(num_layer)
        ])

        # 添加跨视图注意力的FFN和LayerNorm
        self.ffn_cross_view = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_channels, feature_channels * 4),
                nn.GELU(),
                nn.Linear(feature_channels * 4, feature_channels)
            ) for _ in range(num_layer)
        ])
        
        self.norm1_cross_view = nn.ModuleList([
            nn.LayerNorm(feature_channels) for _ in range(num_layer)
        ])
        
        self.norm2_cross_view = nn.ModuleList([
            nn.LayerNorm(feature_channels) for _ in range(num_layer)
        ])

        # 添加跨平面注意力的FFN和LayerNorm
        self.ffn_cross_plane = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_channels, feature_channels * 4),
                nn.GELU(),
                nn.Linear(feature_channels * 4, feature_channels)
            ) for _ in range(num_layer)
        ])
        
        self.norm1_cross_plane = nn.ModuleList([
            nn.LayerNorm(feature_channels) for _ in range(num_layer)
        ])
        
        self.norm2_cross_plane = nn.ModuleList([
            nn.LayerNorm(feature_channels) for _ in range(num_layer)
        ])

        self.tri_planes = None
        self.attn_value = None
        self.attn_score = None
        
    def sample_points_along_axis(self, plane, scene_bounds, query_tensor_input):
        """
        沿着平面的垂直轴均匀采样点
        Args:
            plane: 'xy' | 'xz' | 'yz'
            scene_bounds: 场景边界
        Returns:
            sampled_points: [num_samples, H, W, 3] 采样点的3D坐标
        """
        H, W = self.grid_size, self.grid_size
        device = query_tensor_input[0].device
        
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
    
    def project_to_other_planes(self, points, plane, tri_planes, b):
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
            plane_feat = tri_planes[other_plane][b] # [C, H, W]
            plane_feat = plane_feat.unsqueeze(0)  # [1, C, H, W]
            
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
    
    
    def forward(self, image_features, depth_cond, c2w_cond, intrinsic_cond, depth, c2w, intrinsic, scene_bounds=None):

        timers = {}
        def start_timer(name):
            
            timers[name] = {
                'start': torch.cuda.Event(enable_timing=True),
                'end': torch.cuda.Event(enable_timing=True)
            }
            timers[name]['start'].record()
                
        def end_timer(name):
            timers[name]['end'].record()
            torch.cuda.synchronize()
            elapsed_time = timers[name]['start'].elapsed_time(timers[name]['end'])
            print(f"{name} 耗时: {elapsed_time:.2f} ms")



        B, Nv = image_features.shape[:2]
        assert image_features.shape[2] == self.feature_channels, \
            f"特征通道数不匹配: 期望 {self.feature_channels}, 实际 {image_features.shape[2]}"  
          
        if(self.tri_planes == None):
            self.tri_planes = {
                'xy': torch.zeros((B, self.feature_channels, self.grid_size, self.grid_size), device=image_features.device),
                'xz': torch.zeros((B, self.feature_channels, self.grid_size, self.grid_size), device=image_features.device).cuda(),
                'yz': torch.zeros((B, self.feature_channels, self.grid_size, self.grid_size), device=image_features.device).cuda(),
            }
            self.attn_value = {
                'xy': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
                'xz': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
                'yz': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
            }     
            self.attn_score = {
                'xy': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
                'xz': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
                'yz': torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size)).cuda(),
            }  
        


        bounds = torch.tensor([10,-10,10,-10,10,-10], dtype=float, device='cuda')
        
        # 计算场景边界
        if scene_bounds == None:
            for b in range(B):
                for v in range(Nv):
                    world_points, mask = project_depth_to_3d(
                        depth_cond[b, v, 0], c2w_cond[b, v], intrinsic_cond[b, v]
                    )
                    
                    bounds[0] = min(world_points[mask][:, 0].min(), bounds[0])
                    bounds[1] = max(world_points[mask][:, 0].max(), bounds[1])
                    bounds[2] = min(world_points[mask][:, 1].min(), bounds[2])
                    bounds[3] = max(world_points[mask][:, 1].max(), bounds[3])
                    bounds[4] = min(world_points[mask][:, 2].min(), bounds[4])
                    bounds[5] = max(world_points[mask][:, 2].max(), bounds[5])

            padding = 0.05
            scene_bounds = (
                bounds[0] - padding * (bounds[1] - bounds[0]),
                bounds[1] + padding * (bounds[1] - bounds[0]),
                bounds[2] - padding * (bounds[3] - bounds[2]),
                bounds[3] + padding * (bounds[3] - bounds[2]),
                bounds[4] - padding * (bounds[5] - bounds[4]),
                bounds[5] + padding * (bounds[5] - bounds[4]),            
            )

        render_point, r_mask = project_depth_to_3d(
            depth[0,0,0], c2w[0,0], intrinsic[0][0]
        )
        query_tensor_input = self.query_tensors

        for n in range(0, self.num_layers):

            for plane_key in self.tri_planes:
                self.tri_planes[plane_key].zero_() 
            for plane_key in self.attn_value:
                self.attn_value[plane_key].zero_()
            for plane_key in self.attn_score:
                self.attn_score[plane_key].zero_()
  
            
            for b in range(B):
                start_timer('view_attention')
                for v in range(Nv):

                    world_points, mask = project_depth_to_3d(
                        depth_cond[b, v, 0], c2w_cond[b, v], intrinsic_cond[b, v]
                    )
                    bounded_points = norm_points_bounds(world_points, scene_bounds)

                    # 对每个平面进行处理
                    for i, plane in enumerate(['xy', 'xz', 'yz']):

                        plane_coords = get_plane_coords(bounded_points, mask, plane)
                        
                        # 将坐标映射到网格索引
                        grid_coords = (plane_coords * 0.5 + 0.5) * (self.grid_size - 1)
                        grid_coords = grid_coords.long().clamp(0, self.grid_size-1)

                        # 获取当前视角的特征
                        features = image_features[b, v, :, mask]  # [C, N]
                        
                        # 创建特征收集器
                        feature_collector = torch.zeros(
                            (self.grid_size * self.grid_size, self.feature_channels),
                            device=features.device,
                            dtype=features.dtype
                        )
                        count_collector = torch.zeros(
                            (self.grid_size * self.grid_size),
                            device=features.device,
                            dtype=features.dtype
                        )
                        
                        # 计算1D索引
                        index_1d = grid_coords[:, 0] * self.grid_size + grid_coords[:, 1]

                        # 使用scatter_reduce收集特征

                        feature_collector.scatter_reduce_(
                            dim=0,
                            index=index_1d.unsqueeze(1).expand(-1, self.feature_channels),
                            src=features.T,
                            reduce="sum",
                            include_self=False
                        )
                        
                        # 收集计数
                        count_collector.scatter_reduce_(
                            dim=0,
                            index=index_1d,
                            src=torch.ones_like(index_1d, dtype=features.dtype),
                            reduce="sum",
                            include_self=False
                        )

                        # 避免除以零
                        count_collector = count_collector.clamp(min=1e-6)
                        
                        # 计算平均特征
                        feature_collector = feature_collector / count_collector.unsqueeze(1)
                        
                        # 重塑为网格形状
                        feature_collector = feature_collector.view(
                            self.grid_size, self.grid_size, self.feature_channels
                        )
                        
                        # 获取所有query
                        queries = query_tensor_input[i]  # [H, W, C]
                        
                        queries = self.norm1_cross_view[i](queries)

                        # 重塑query和特征以进行批处理
                        queries = queries.reshape(-1, self.feature_channels)  # [H*W, C]
                        features_flat = feature_collector.reshape(-1, self.feature_channels)  # [H*W, C]
                        
                        # 对投影特征也进行归一化
                        features_flat = self.norm1_cross_view[i](features_flat)

                        # 计算注意力
                        
                        score, value = self.attentions_corss_view[n](
                            queries.unsqueeze(0),  # [1, H*W, C]
                            features_flat.unsqueeze(0) # [1, H*W, C]
                        )  # [1, H*W, C]
                        
                        # 重塑回平面特征
                        plane_feat = value.squeeze(0).reshape(self.grid_size, self.grid_size, self.feature_channels)
                        plane_feat = plane_feat.permute(2, 0, 1)  # [C, H, W]
                        self.attn_value[plane][b, v] = plane_feat
                        plane_score = score.squeeze(0).reshape(self.grid_size, self.grid_size, self.feature_channels)
                        plane_score = plane_score.permute(2, 0, 1)  
                        self.attn_score[plane][b, v] = plane_score
                        # 累积更新平面特征
                        # if v == 0:
                        #     tri_planes[plane][b] = plane_feat
                        # else:
                            #     tri_planes[plane][b] = (tri_planes[plane][b] * v + plane_feat) / (v + 1)
                for i, plane in enumerate(['xy', 'xz', 'yz']):
                    # 在视图维度上做softmax归一化
                    attn_weight = torch.softmax(self.attn_score[plane][b], dim=0)  # [Nv, C, H, W]
                    self.tri_planes[plane][b] = torch.sum(attn_weight * self.attn_value[plane][b], dim=0)

                    res_output = query_tensor_input[i] + self.tri_planes[plane][b].permute(1, 2, 0)

                    norm_output = self.norm2_cross_view[i](res_output)

                    FFN_output = self.ffn_cross_view[i](norm_output)

                    res_FFN_output = res_output + FFN_output
                    
                    self.tri_planes[plane][b] = res_FFN_output.permute(2, 0, 1)
                end_timer('view_attention')
                pdb.set_trace()
                start_timer('plane_attention')
                # 计算跨平面注意力
                for i, plane in enumerate(['xy', 'xz', 'yz']):
                    # 获取当前平面特征
                    plane_feat = self.tri_planes[plane][b]  # [C, H, W]
                    # 使用permute将C维度移到最后
                    plane_feat = plane_feat.permute(1, 2, 0)  # [H, W, C]
                    plane_feat = plane_feat.reshape(-1, self.feature_channels)  # [H*W, C]

                    plane_feat = self.norm1_cross_plane[i](plane_feat)
                    
                    # 采样点
                    sampled_points = self.sample_points_along_axis(plane, scene_bounds, query_tensor_input)
                    
                    # 投影到其他平面并获取特征
                    other_plane_features = self.project_to_other_planes(sampled_points, plane, self.tri_planes, b)
                    other_plane_features = self.norm1_cross_plane[i](other_plane_features)
                    # 计算跨平面注意力
                    values = []
                    scores = []
                    for j in range(0, 16):
                        other_plane_feature = other_plane_features[:, :, j, :].reshape(-1, self.feature_channels)
                        score, value = self.attentions_cross_plane[n](
                            plane_feat.unsqueeze(0),  # [1, H*W, C]
                            other_plane_feature.unsqueeze(0)  # [1, H*W, C]
                        )  # [1, H*W, C]
                        values.append(value)
                        scores.append(score)
                    
                    # 将所有注意力输出聚合
                    attended_value = torch.stack(values, dim=0)  # [16, H*W, C]
                    attended_score = torch.stack(scores, dim=0)  # [16, H*W, C]
                    attended_score = torch.softmax(attended_score, dim=0)
                    attended_feat = torch.sum(attended_score * attended_value, dim=0)  # [H*W, C]

                    updated_plane_data = attended_feat.reshape(self.grid_size, self.grid_size, self.feature_channels)# [H, W, C]

                    res_output2 = updated_plane_data + self.tri_planes[plane][b].permute(1, 2, 0)

                    norm_output2 = self.norm2_cross_plane[i](res_output2)

                    FFN_output2 = self.ffn_cross_plane[i](norm_output2)

                    res_FFN_output2 = FFN_output2 + res_output2

                    self.tri_planes[plane][b] = res_FFN_output2.permute(2, 0, 1)
                end_timer('plane_attention')
            query_tensor_input = torch.cat([self.tri_planes['xy'].permute(0, 2, 3, 1), 
                                            self.tri_planes['xz'].permute(0, 2, 3, 1), 
                                            self.tri_planes['yz'].permute(0, 2, 3, 1)], dim=0)
            
        query_tensor_input = query_tensor_input.permute(0, 3, 1, 2).unsqueeze(0)

        return render_point[r_mask], query_tensor_input, scene_bounds
    

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
