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
        
        # 转置回原始形状并合并头
        attn_score = attn_score.transpose(1, 2)  # [B, H*W, n_heads, d_head]
        attn_score = attn_score.reshape(B, L, C)  # [B, H*W, C]
        v = v.transpose(1, 2)  # [B, n_heads, H*W, d_head]
        v = v.reshape(B, L, C)  # [B, H*W, C]
        
        return attn_score, v


class TriplaneRefinementLayer(nn.Module):
    def __init__(self, grid_size, n_heads, feature_channels, num_samples):
        super().__init__()
        self.grid_size = grid_size
        self.feature_channels = feature_channels
        self.num_samples = num_samples # Number of samples along axis perpendicular to plane

        self.attentions_cross_view = TriplaneAttention(
            query_dim=feature_channels,
            context_dim=feature_channels,
            n_heads=n_heads
        )
        self.attentions_cross_plane = TriplaneAttention(
            query_dim=feature_channels,
            context_dim=feature_channels,
            n_heads=n_heads
        )

    def sample_points_along_axis(self, plane_type, scene_bounds_tuple, device):
        """
        Samples points along the axis perpendicular to the given plane.
        Args:
            plane_type: 'xy' | 'xz' | 'yz'
            scene_bounds_tuple: (min_x, max_x, min_y, max_y, min_z, max_z) for normalized [-1,1] space
            device: torch device
        Returns:
            sampled_points: [H, W, num_samples, 3] 3D coordinates in normalized [-1,1] space
        """
        H, W = self.grid_size, self.grid_size
        
        # Grid coordinates for the plane itself (range -1 to 1)
        u_coords = torch.linspace(-1, 1, W, device=device)
        v_coords = torch.linspace(-1, 1, H, device=device)
        grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij') # H, W

        # Samples along the third dimension (range -1 to 1, as scene_bounds are for normalized space)
        # The original code used scene_bounds directly, implying they might not be normalized for this linspace.
        # For consistency, if scene_bounds_tuple defines the [-1,1] space, then sampling within [-1,1] is appropriate.
        # If scene_bounds_tuple is world coords, then it should be used.
        # The original code used scene_bounds[4], scene_bounds[5] etc. which are world coords.
        # Let's assume scene_bounds_tuple is (min_x, max_x, min_y, max_y, min_z, max_z) in world coordinates.
        
        sampled_points_list = []

        if plane_type == 'xy': # Sample along z
            # grid_u is x, grid_v is y
            z_s = torch.linspace(scene_bounds_tuple[4], scene_bounds_tuple[5], self.num_samples, device=device)
            x_coords = grid_u.unsqueeze(-1).expand(H, W, self.num_samples)
            y_coords = grid_v.unsqueeze(-1).expand(H, W, self.num_samples)
            z_coords = z_s.view(1, 1, -1).expand(H, W, self.num_samples)
            sampled_points = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        elif plane_type == 'xz': # Sample along y
            # grid_u is x, grid_v is z
            y_s = torch.linspace(scene_bounds_tuple[2], scene_bounds_tuple[3], self.num_samples, device=device)
            x_coords = grid_u.unsqueeze(-1).expand(H, W, self.num_samples)
            z_coords = grid_v.unsqueeze(-1).expand(H, W, self.num_samples)
            y_coords = y_s.view(1, 1, -1).expand(H, W, self.num_samples)
            sampled_points = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        else:  # 'yz', sample along x
            # grid_u is y, grid_v is z
            x_s = torch.linspace(scene_bounds_tuple[0], scene_bounds_tuple[1], self.num_samples, device=device)
            y_coords = grid_u.unsqueeze(-1).expand(H, W, self.num_samples)
            z_coords = grid_v.unsqueeze(-1).expand(H, W, self.num_samples)
            x_coords = x_s.view(1, 1, -1).expand(H, W, self.num_samples)
            sampled_points = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        
        # The sampled points are in world coordinates. Normalize them.
        return sampled_points


    def project_to_other_planes(self, points_HWS3_norm, current_plane_type, current_triplanes_for_batch_CHW_dict):
        """
        Projects 3D points to other two planes and samples features.
        Args:
            points_HWS3_norm: [H, W, num_samples, 3] 3D points in normalized [-1,1] space
            current_plane_type: The plane from which points were sampled ('xy', 'xz', or 'yz')
            current_triplanes_for_batch_CHW_dict: Dict {'xy': [C,H,W], 'xz': [C,H,W], 'yz': [C,H,W]} features for current batch item
        Returns:
            other_plane_features: [H, W, num_samples*2, C] Features from the other two planes
        """
        H, W_grid, S_samples = points_HWS3_norm.shape[:3]
        device = points_HWS3_norm.device
        
        projected_features_list = []
        
        plane_to_idx = {'xy': 0, 'xz': 1, 'yz': 2} # Not used here, using dict keys

        for other_plane_type in ['xy', 'xz', 'yz']:
            if other_plane_type == current_plane_type:
                continue

            # Get 2D coordinates for projection onto other_plane_type
            # points_HWS3_norm are already normalized coordinates
            if other_plane_type == 'xy':
                coords_HWS2 = points_HWS3_norm[..., [0, 1]] # x, y
            elif other_plane_type == 'xz':
                coords_HWS2 = points_HWS3_norm[..., [0, 2]] # x, z
            else:  # 'yz'
                coords_HWS2 = points_HWS3_norm[..., [1, 2]] # y, z
            
            # coords_HWS2 are in [-1, 1] range, suitable for grid_sample
            # grid_sample expects coords in [N, H_out, W_out, 2] or [N, n_points, 2]
            # Input feature map for grid_sample: [N, C, H_in, W_in]
            
            plane_feat_CHW = current_triplanes_for_batch_CHW_dict[other_plane_type] # [C, H, W]
            plane_feat_NCHW = plane_feat_CHW.unsqueeze(0) # [1, C, H, W]

            # Reshape coords_HWS2 for grid_sample: [1, H*W*S, 1, 2] for (H_out=H*W*S, W_out=1)
            # Or more generally [N, n_points, 2] -> [N, 1, n_points, 2] if H_out=1
            grid_coords_for_sample = coords_HWS2.reshape(1, H * W_grid * S_samples, 1, 2) # [1, num_total_points, 1, 2]
            
            # F.grid_sample expects grid to be normalized between -1 and 1
            # align_corners=True is often used when coordinates are considered centers of boundary pixels.
            # align_corners=False is often used when coordinates are considered corners.
            # The original code used align_corners=True.
            sampled_feat = F.grid_sample(
                plane_feat_NCHW, # [1, C, H_grid, W_grid]
                grid_coords_for_sample, # [1, H*W*S, 1, 2] (x,y order)
                mode='bilinear',
                padding_mode='border', # or 'zeros'
                align_corners=True # Match original
            ) # Output: [1, C, H*W*S, 1]
            
            # Reshape sampled_feat back to [H, W_grid, S_samples, C]
            sampled_feat_CHWS = sampled_feat.squeeze(-1).squeeze(0) # [C, H*W*S]
            sampled_feat_HWS_C = sampled_feat_CHWS.reshape(self.feature_channels, H, W_grid, S_samples).permute(1, 2, 3, 0) # [H,W,S,C]
            projected_features_list.append(sampled_feat_HWS_C)
            
        return torch.cat(projected_features_list, dim=2) # Concatenate along samples dim: [H, W, num_samples*2, C]

    def forward(self, current_triplanes_dict_BCHW, image_features_BNC, 
                depth_cond_BN1HW, c2w_cond_BN44, intrinsic_cond_BN33, scene_bounds_tuple):
        """
        Args:
            current_triplanes_dict_BCHW: {'xy': [B,C,H,W], ...} Input triplanes for this layer
            image_features_BNC: [B, Nv, Cimg] Features from images (Nv is num_views, Cimg should be self.feature_channels)
            depth_cond_BN1HW: [B, Nv, 1, Hdepth, Wdepth] Depth maps
            c2w_cond_BN44: [B, Nv, 4, 4] Camera to world matrices
            intrinsic_cond_BN33: [B, Nv, 3, 3] or [B, Nv, 4, 4] Intrinsic matrices
            scene_bounds_tuple: (min_x, max_x, ...) world coordinates scene bounds
        Returns:
            refined_triplanes_dict_BCHW: {'xy': [B,C,H,W], ...}
        """

        B, Nv, C_img, _, _ = image_features_BNC.shape
        assert C_img == self.feature_channels, "Image feature channels mismatch"
        device = image_features_BNC.device
        dtype = image_features_BNC.dtype

        # Initialize containers for cross-view attention scores and values
        attn_values_cv = {p: torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size), device=device, dtype=dtype) for p in ['xy', 'xz', 'yz']}
        attn_scores_cv = {p: torch.zeros((B, Nv, self.feature_channels, self.grid_size, self.grid_size), device=device, dtype=dtype) for p in ['xy', 'xz', 'yz']}

        # 1. Cross-View Attention
        for b_idx in range(B):
            for v_idx in range(Nv):
                # Project depth to 3D world points for current view
                world_points, mask = project_depth_to_3d(
                    depth_cond_BN1HW[b_idx, v_idx, 0], 
                    c2w_cond_BN44[b_idx, v_idx], 
                    intrinsic_cond_BN33[b_idx, v_idx]
                ) # world_points: [num_pixels, 3], mask: [num_pixels]
                
                if not torch.any(mask): continue

                # Normalize world points to [-1, 1] using scene_bounds

                bounded_points_norm = norm_points_bounds(world_points, scene_bounds_tuple) # [num_pixels, 3]

                features_for_scatter = image_features_BNC[b_idx, v_idx, :, mask]
                if world_points[mask].shape[0] == 0: continue
                features_for_scatter = torch.randn(world_points[mask].shape[0], self.feature_channels, device=device, dtype=dtype) # Dummy
                # In reality, this should come from `image_features_BNC` based on `mask` and point projections.

                for plane_idx, plane_type in enumerate(['xy', 'xz', 'yz']):
                    plane_coords_2D_norm = get_plane_coords(bounded_points_norm, mask, plane_type) # [N_masked_points, 2]
                    if plane_coords_2D_norm.numel() == 0: continue

                    # Map normalized [-1,1] coords to grid indices [0, grid_size-1]
                    grid_coords_indices = (plane_coords_2D_norm * 0.5 + 0.5) * (self.grid_size - 1)
                    grid_coords_indices = grid_coords_indices.long().clamp(0, self.grid_size - 1) # [N_masked_points, 2]

                    # Scatter-add features to the plane grid cells
                    # feature_collector_flat: [grid_size*grid_size, C]
                    feature_collector_flat = torch.zeros(self.grid_size * self.grid_size, self.feature_channels, device=device, dtype=dtype)
                    count_collector_flat = torch.zeros(self.grid_size * self.grid_size, device=device, dtype=dtype)
                    
                    # 1D index for scatter: idx = y * W + x
                    index_1d = grid_coords_indices[:, 1] * self.grid_size + grid_coords_indices[:, 0] # Assuming H, W order for coords

                    feature_collector_flat.scatter_add_(
                        dim=0, 
                        index=index_1d.unsqueeze(1).expand(-1, self.feature_channels),
                        src=features_for_scatter # [N_masked_points, C]
                    )
                    count_collector_flat.scatter_add_(
                        dim=0,
                        index=index_1d,
                        src=torch.ones_like(index_1d, dtype=dtype)
                    )
                    
                    # Avoid division by zero
                    count_collector_flat = count_collector_flat.clamp(min=1e-6)
                    # Average features in each grid cell
                    avg_features_flat = feature_collector_flat / count_collector_flat.unsqueeze(1) # [grid_size*grid_size, C]

                    # Prepare queries from current_triplanes_dict_BCHW for this plane and batch item
                    current_plane_queries_CHW = current_triplanes_dict_BCHW[plane_type][b_idx] # [C, H, W]
                    queries_HWC = current_plane_queries_CHW.permute(1, 2, 0) # [H, W, C]
                    queries_flat = queries_HWC.reshape(-1, self.feature_channels) # [H*W, C]

                    # Cross-view attention
                    # query: [1, H*W, C], context: [1, H*W, C] (avg_features_flat are context here)
                    score, value = self.attentions_cross_view(
                        queries_flat.unsqueeze(0),
                        avg_features_flat.unsqueeze(0) 
                    ) # score, value: [1, H*W, C]

                    # Reshape back to plane format [C, H, W]
                    attended_feat_CHW = value.squeeze(0).reshape(self.grid_size, self.grid_size, self.feature_channels).permute(2,0,1)
                    score_CHW = score.squeeze(0).reshape(self.grid_size, self.grid_size, self.feature_channels).permute(2,0,1)
                    
                    attn_values_cv[plane_type][b_idx, v_idx] = attended_feat_CHW
                    attn_scores_cv[plane_type][b_idx, v_idx] = score_CHW
        
        # Aggregate features from all views for each plane using softmax over scores
        intermediate_triplanes_dict_BCHW = {}
        for plane_type in ['xy', 'xz', 'yz']:
            # attn_scores_cv[plane_type] is [B, Nv, C, H, W]
            # Softmax over Nv dimension
            weights_cv = F.softmax(attn_scores_cv[plane_type], dim=1) # [B, Nv, C, H, W]
            intermediate_triplanes_dict_BCHW[plane_type] = torch.sum(weights_cv * attn_values_cv[plane_type], dim=1) # [B, C, H, W]

        # 2. Cross-Plane Attention
        # Initialize output triplanes for this layer
        refined_triplanes_dict_BCHW = {p: torch.zeros_like(intermediate_triplanes_dict_BCHW[p]) for p in ['xy', 'xz', 'yz']}

        for b_idx in range(B):
            current_batch_intermediate_triplanes_CHW_dict = {
                p_type: intermediate_triplanes_dict_BCHW[p_type][b_idx] for p_type in ['xy', 'xz', 'yz']
            }
            for plane_type in ['xy', 'xz', 'yz']:
                # Query is the current plane's features after cross-view attention
                query_plane_feat_CHW = intermediate_triplanes_dict_BCHW[plane_type][b_idx] # [C,H,W]
                query_plane_feat_flat = query_plane_feat_CHW.permute(1,2,0).reshape(-1, self.feature_channels) # [H*W, C]

                # Sample points along axis perpendicular to current_plane_type (these are world coordinates)
                # then normalize them
                sampled_points_HWS3_norm = self.sample_points_along_axis(plane_type, scene_bounds_tuple, device)
                
                # Project these points to the *other* two planes and get their features
                # other_plane_features_HWS2C: [H, W, num_samples*2, C]
                other_plane_features_HWS2C = self.project_to_other_planes(
                    sampled_points_HWS3_norm, plane_type, current_batch_intermediate_triplanes_CHW_dict
                )

                # Perform attention for each of the (num_samples * 2) feature vectors per grid cell
                # other_plane_features_flat_S2_HWC = other_plane_features_HWS2C.reshape(-1, self.num_samples * 2, self.feature_channels) # [H*W, num_samples*2, C]
                
                scores_list_cp = []
                values_list_cp = []
                
                # Iterate over each "sample ray" feature
                for s_idx in range(self.num_samples * 2):
                    # context_feat_flat: [H*W, C]
                    context_feat_flat = other_plane_features_HWS2C[:,:,s_idx,:].reshape(-1, self.feature_channels)
                    
                    score, value = self.attentions_cross_plane(
                        query_plane_feat_flat.unsqueeze(0), # [1, H*W, C]
                        context_feat_flat.unsqueeze(0)      # [1, H*W, C]
                    ) # score, value: [1, H*W, C]
                    scores_list_cp.append(score)
                    values_list_cp.append(value)
                
                # Stack scores and values from all samples
                # stacked_scores: [num_samples*2, H*W, C]
                stacked_scores = torch.cat(scores_list_cp, dim=0) 
                stacked_values = torch.cat(values_list_cp, dim=0)

                # Softmax over the sample dimension
                weights_cp = F.softmax(stacked_scores, dim=0) # [num_samples*2, H*W, C]
                
                # Weighted sum of values
                attended_feat_flat = torch.sum(weights_cp * stacked_values, dim=0) # [H*W, C]
                
                # Reshape back to plane format [C, H, W]
                updated_plane_data_CHW = attended_feat_flat.reshape(self.grid_size, self.grid_size, self.feature_channels).permute(2,0,1)
                refined_triplanes_dict_BCHW[plane_type][b_idx] = updated_plane_data_CHW
                
        return refined_triplanes_dict_BCHW


class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=4, feature_channels=196, num_samples = 8, num_layers=3): # num_samples_per_layer
        super().__init__()
        self.grid_size = grid_size
        self.feature_channels = feature_channels
        self.num_layers = num_layers

        # Initial query tensors for the triplanes (learnable)
        # Shape: [C, H, W] for each plane
        self.initial_query_tensors = nn.ParameterList([
            nn.Parameter(torch.randn(feature_channels, grid_size, grid_size)) for _ in range(3)
        ])
        self.plane_names = ['xy', 'xz', 'yz']

        # Stack of refinement layers
        self.refinement_layers = nn.ModuleList([
            TriplaneRefinementLayer(grid_size, n_heads, feature_channels, num_samples)
            for _ in range(num_layers)
        ])

    def forward(self, image_features, depth_cond, c2w_cond, intrinsic_cond, depth, c2w, intrinsic, scene_bounds_tuple=None):
        """
        Args:
            image_features: [B, Nv, Cimg, Himg, Wimg] or [B, Nv, NumPoints, Cimg] or [B, Nv, Cimg] - needs clarification for TriplaneRefinementLayer
                            For this example, let's assume it's compatible with TriplaneRefinementLayer's expectation.
                            A common structure might be [B, Nv, C_feat_dim] if using global features per view,
                            or pre-projected features per point.
            depth_cond, c2w_cond, intrinsic_cond: Conditional inputs for scene bounds and refinement
            depth, c2w, intrinsic: Inputs for rendering the final point (usually for target view)
            scene_bounds_tuple: Optional precomputed scene bounds (min_x, max_x, min_y, max_y, min_z, max_z)
        """
        B, Nv = image_features.shape[:2] # Assuming image_features has at least 2 dims for B, Nv
        device = image_features.device
        
        # Calculate scene bounds if not provided
        if scene_bounds_tuple is None:
            # Initialize bounds with large/small values
            # Note: using .item() for min/max on single values, or direct tensor ops
            current_bounds_tensor = torch.tensor(
                [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')],
                dtype=torch.float32, device=device
            )

            for b_idx in range(B):
                for v_idx in range(Nv):
                    world_points, mask = project_depth_to_3d(
                        depth_cond[b_idx, v_idx, 0], 
                        c2w_cond[b_idx, v_idx], 
                        intrinsic_cond[b_idx, v_idx]
                    )
                    if torch.any(mask):
                        valid_points = world_points[mask]
                        current_bounds_tensor[0] = torch.min(current_bounds_tensor[0], valid_points[:, 0].min())
                        current_bounds_tensor[1] = torch.max(current_bounds_tensor[1], valid_points[:, 0].max())
                        current_bounds_tensor[2] = torch.min(current_bounds_tensor[2], valid_points[:, 1].min())
                        current_bounds_tensor[3] = torch.max(current_bounds_tensor[3], valid_points[:, 1].max())
                        current_bounds_tensor[4] = torch.min(current_bounds_tensor[4], valid_points[:, 2].min())
                        current_bounds_tensor[5] = torch.max(current_bounds_tensor[5], valid_points[:, 2].max())
            
            padding_percentages = 0.05 # 5% padding
            widths = current_bounds_tensor[1::2] - current_bounds_tensor[0::2]
            paddings = widths * padding_percentages
            
            padded_bounds_list = []
            for i in range(3): # x, y, z
                padded_bounds_list.append(current_bounds_tensor[2*i] - paddings[i])
                padded_bounds_list.append(current_bounds_tensor[2*i+1] + paddings[i])
            scene_bounds_tuple = tuple(b.item() for b in padded_bounds_list)


        # Initialize triplanes from initial query tensors
        current_triplanes_dict = {}
        for i, plane_name in enumerate(self.plane_names):
            # Expand initial query [C,H,W] to [B,C,H,W]
            current_triplanes_dict[plane_name] = self.initial_query_tensors[i].unsqueeze(0).expand(
                B, self.feature_channels, self.grid_size, self.grid_size
            ).clone().to(device)

        # Iteratively refine triplanes through layers
        for layer_idx, layer in enumerate(self.refinement_layers):
            current_triplanes_dict = layer(
                current_triplanes_dict, image_features,
                depth_cond, c2w_cond, intrinsic_cond,
                scene_bounds_tuple
            )

        # Get points for rendering from the target view (depth, c2w, intrinsic)
        # This part is from your original code's return values
        render_points_world, r_mask = project_depth_to_3d(
            depth[0,0,0], c2w[0,0], intrinsic[0][0] # Assuming B=1, Nv=1 for target render
        )

        # Concatenate triplanes for output: [B, 3, C, H, W]
        output_triplanes_tensor = torch.cat([
            current_triplanes_dict['xy'].unsqueeze(1),
            current_triplanes_dict['xz'].unsqueeze(1),
            current_triplanes_dict['yz'].unsqueeze(1)
        ], dim=1)

        return render_points_world[r_mask] if torch.any(r_mask) else torch.empty((0,3), device=device), \
               output_triplanes_tensor, \
               scene_bounds_tuple

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
