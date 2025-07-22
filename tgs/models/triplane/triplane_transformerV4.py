import torch
import torch.nn as nn
import pdb
from einops import rearrange
import torch.nn.functional as F
import diff_gaussian_rasterization_ch2280
import diff_gaussian_rasterization_ch1
import torchvision.utils as vutils
from gaussian.cameras import get_ortho_camera_params
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

def project_depth_to_3d(depth, c2w_cond, K):
    """
    depth: [B, H, W]
    c2w_cond: [B, 4, 4] (相机到世界矩阵)
    K: [B, 3, 3] (内参矩阵)
    返回: [B, H, W, 3] (世界坐标系下的3D点), [B, H, W] (有效深度mask)
    """
    B, H, W = depth.shape
    device = depth.device

    depth_mask = depth != 0.0
    depth = depth.clone()
    depth[depth_mask] += 0.01

    # 生成像素网格
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    uv_homogeneous = torch.stack([u.float(), v.float(), torch.ones_like(u).float()], dim=-1)  # [H, W, 3]
    uv_homogeneous = uv_homogeneous.unsqueeze(0).expand(B, H, W, 3)  # [B, H, W, 3]

    # 计算相机内参矩阵的逆
    K_inv = torch.inverse(K)  # [B, 3, 3]

    # 将像素坐标转换到相机坐标系下的方向向量
    uv_homogeneous_flat = uv_homogeneous.reshape(B, -1, 3)  # [B, H*W, 3]
    cam_coords_flat = torch.bmm(uv_homogeneous_flat, K_inv.transpose(1, 2))  # [B, H*W, 3]

    # 乘以深度得到相机坐标系下的 3D 点
    depth_flat = depth.reshape(B, -1, 1)  # [B, H*W, 1]
    points_cam_flat = cam_coords_flat * depth_flat  # [B, H*W, 3]

    # 齐次坐标
    ones = torch.ones((B, H*W, 1), device=device, dtype=points_cam_flat.dtype)
    points_cam_homogeneous_flat = torch.cat([points_cam_flat, ones], dim=-1)  # [B, H*W, 4]

    # 相机到世界
    points_world_homogeneous_flat = torch.bmm(points_cam_homogeneous_flat, c2w_cond.transpose(1, 2))  # [B, H*W, 4]

    # 取前三个分量
    points_world_flat = points_world_homogeneous_flat[:, :, :3]  # [B, H*W, 3]

    # reshape 回原始图像尺寸
    world_points = points_world_flat.reshape(B, H, W, 3)  # [B, H, W, 3]
    depth_mask = depth_mask  # [B, H, W]

    return world_points, depth_mask

import torch




# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[..., 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[..., 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[..., 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(3), normed_y.unsqueeze(3), normed_z.unsqueeze(3)], dim=-1)
    
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
        # context: [B, H*W, Nv, C]
        B, L, C = query.shape
        Nv = context.shape[2]

        query = self.norm(query)  # [B, H*W, C]
        k = self.to_k(context)    # [B, H*W, Nv, C]
        v = self.to_v(context)    # [B, H*W, Nv, C]

        # 多头拆分
        query = query.view(B, L, self.n_heads, self.d_head)  # [B, H*W, n_heads, d_head]
        k = k.view(B, L, Nv, self.n_heads, self.d_head)      # [B, H*W, Nv, n_heads, d_head]
        v = v.view(B, L, Nv, self.n_heads, self.d_head)      # [B, H*W, Nv, n_heads, d_head]

        # 交换head和view维度，方便后续计算
        query = query.permute(0, 2, 1, 3)   # [B, n_heads, H*W, d_head]
        k = k.permute(0, 3, 1, 2, 4)        # [B, n_heads, H*W, Nv, d_head]
        v = v.permute(0, 3, 1, 2, 4)        # [B, n_heads, H*W, Nv, d_head]

        # 逐点多头注意力: 每个空间点、每个head，query和Nv个key做点积
        # query: [B, n_heads, H*W, d_head]
        # k:     [B, n_heads, H*W, Nv, d_head]
        attn_score = (query.unsqueeze(3) * k).sum(-1)  # [B, n_heads, H*W, Nv]
        attn_score = attn_score / (self.d_head ** 0.5)
        attn_prob = torch.softmax(attn_score, dim=-1)  # [B, n_heads, H*W, Nv]

        # 加权value
        out = (attn_prob.unsqueeze(-1) * v).sum(3)  # [B, n_heads, H*W, d_head]

        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)  # [B, H*W, C]
        # attn_prob: [B, n_heads, H*W, Nv]
        return out, attn_prob

class ImageTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, n_heads=4, feature_channels=196, num_samples=4, num_layer=2):
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

        bg_color = torch.zeros(feature_channels, device="cuda")
        scene_range_min_np = np.array([-1.0, -1.0, -1.0])
        scene_range_max_np = np.array([1.0, 1.0, 1.0])

        # 获取 XY 平面的相机参数
        camera_params_xy = get_ortho_camera_params('xy', grid_size=self.grid_size, 
                                                scene_range_min_np=scene_range_min_np, 
                                                scene_range_max_np=scene_range_max_np)

        # 获取 XZ 平面的相机参数
        camera_params_xz = get_ortho_camera_params('xz', grid_size=self.grid_size, 
                                                scene_range_min_np=scene_range_min_np, 
                                                scene_range_max_np=scene_range_max_np)

        # 获取 YZ 平面的相机参数
        camera_params_yz = get_ortho_camera_params('yz', grid_size=self.grid_size, 
                                                scene_range_min_np=scene_range_min_np, 
                                                scene_range_max_np=scene_range_max_np)

        self.raster_settings_XY = diff_gaussian_rasterization_ch2280.GaussianRasterizationSettings(
            image_height=camera_params_xy['image_height'],
            image_width=camera_params_xy['image_width'],
            tanfovx=camera_params_xy['tanfovx'],
            tanfovy=camera_params_xy['tanfovy'],
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera_params_xy['viewmatrix'],
            projmatrix=camera_params_xy['projmatrix'], # 这是 full_proj_transform
            sh_degree=camera_params_xy['sh_degree'],
            campos=camera_params_xy['campos'],
            prefiltered=False
        )

        self.raster_settings_YZ = diff_gaussian_rasterization_ch2280.GaussianRasterizationSettings(
            image_height=camera_params_yz['image_height'],
            image_width=camera_params_yz['image_width'],
            tanfovx=camera_params_yz['tanfovx'],
            tanfovy=camera_params_yz['tanfovy'],
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera_params_yz['viewmatrix'],
            projmatrix=camera_params_yz['projmatrix'], # 这是 full_proj_transform
            sh_degree=camera_params_yz['sh_degree'],
            campos=camera_params_yz['campos'],
            prefiltered=False
        )
        
        self.raster_settings_XZ = diff_gaussian_rasterization_ch2280.GaussianRasterizationSettings(
            image_height=camera_params_xz['image_height'],
            image_width=camera_params_xz['image_width'],
            tanfovx=camera_params_xz['tanfovx'],
            tanfovy=camera_params_xz['tanfovy'],
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera_params_xz['viewmatrix'],
            projmatrix=camera_params_xz['projmatrix'], # 这是 full_proj_transform
            sh_degree=camera_params_xz['sh_degree'],
            campos=camera_params_xz['campos'],
            prefiltered=False
        )
        
        
    def sample_points_along_axis(self, plane, query_tensor_input):
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
            z_samples = torch.linspace(-1, 1, self.num_samples, device=device)
            sampled_points = torch.stack([
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples),
                z_samples.view(1, 1, -1).expand(H, W, -1)
            ], dim=-1)
        elif plane == 'xz':
            # 在y轴上采样
            y_samples = torch.linspace(-1, 1, self.num_samples, device=device)
            sampled_points = torch.stack([
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                y_samples.view(1, 1, -1).expand(H, W, -1),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples)
            ], dim=-1)
        else:  # 'yz'
            # 在x轴上采样
            x_samples = torch.linspace(-1, 1, self.num_samples, device=device)
            sampled_points = torch.stack([
                x_samples.view(1, 1, -1).expand(H, W, -1),
                grid_x.unsqueeze(-1).expand(-1, -1, self.num_samples),
                grid_y.unsqueeze(-1).expand(-1, -1, self.num_samples)
            ], dim=-1)
            
        return sampled_points
    
    def project_to_other_planes(self, points, plane, cross_view_plane_feats):
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
            plane_feat = cross_view_plane_feats[plane_to_idx[other_plane]] # [C, H, W]
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
    
    def Splatting_Points_to_Plane(self, feature, pts):
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
        coords = pts  # 前三位是坐标
        features = feature # 特征

        screenspace_points = torch.zeros_like(coords, dtype=coords.dtype, requires_grad=True, device=coords.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points

        shs = None
        
        colors_precomp =  rearrange(features,'C N -> N C')

        colors_precomp_count = torch.ones([colors_precomp.size(0), 1], dtype=torch.float32, device="cuda")

        cov3D_precomp = None
        scales = torch.full([colors_precomp.size(0), 3], 1/(self.grid_size * 3), dtype=torch.float32, device="cuda")
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        rotations = identity_quat.repeat(colors_precomp.size(0), 1)
        opacities = torch.full([colors_precomp.size(0), 1], 0.0067, dtype=torch.float32, device="cuda")



        # 对每个平面进行投影
        tri_planes = {}
        for plane in ['xy', 'xz', 'yz']:
            # 选择对应的坐标轴
            if plane == 'xy':
                rasterizer = diff_gaussian_rasterization_ch2280.GaussianRasterizer(self.raster_settings_XY)
                rasterizer_count = diff_gaussian_rasterization_ch1.GaussianRasterizer(self.raster_settings_XY)
            if plane == 'xz':
                rasterizer = diff_gaussian_rasterization_ch2280.GaussianRasterizer(self.raster_settings_XZ)
                rasterizer_count = diff_gaussian_rasterization_ch1.GaussianRasterizer(self.raster_settings_XZ)
            if plane == 'yz':
                rasterizer = diff_gaussian_rasterization_ch2280.GaussianRasterizer(self.raster_settings_YZ)       
                rasterizer_count = diff_gaussian_rasterization_ch1.GaussianRasterizer(self.raster_settings_YZ) 
  
            rendered_image, radii, depth = rasterizer(
                means3D = coords,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacities,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            count_rendered_image, radii, depth = rasterizer_count(
                means3D = coords,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp_count,
                opacities = opacities,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            
            count_rendered_image = count_rendered_image.clamp(min=1e-6)

            tri_planes[plane] = (rendered_image / count_rendered_image).unsqueeze(0)
        return tri_planes   
     
    def forward(self, image_features, depth_cond, c2w_cond, intrinsic_cond, depth, c2w, intrinsic, sample_grid, scene_bounds=None):

        # timers = {}
        # def start_timer(name):
            
        #     timers[name] = {
        #         'start': torch.cuda.Event(enable_timing=True),
        #         'end': torch.cuda.Event(enable_timing=True)
        #     }
        #     timers[name]['start'].record()
                
        # def end_timer(name):
        #     timers[name]['end'].record()
        #     torch.cuda.synchronize()
        #     elapsed_time = timers[name]['start'].elapsed_time(timers[name]['end'])
        #     print(f"{name} 耗时: {elapsed_time:.2f} ms")



        B, Nv = image_features.shape[:2]

        # Nv=1
        assert image_features.shape[2] == self.feature_channels, \
            f"特征通道数不匹配: 期望 {self.feature_channels}, 实际 {image_features.shape[2]}"  
          
        bounds = torch.tensor([10,-10,10,-10,10,-10], dtype=float, device='cuda')
        
        # 计算场景边界
        if scene_bounds == None:
            for b in range(B):
                world_points, mask = project_depth_to_3d(
                        depth_cond[b, :], c2w_cond[b, :], intrinsic_cond[b, :]
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
        # render_point, r_mask = project_depth_to_3d(
        #     depth[0,0], c2w[0], intrinsic[0]
        # )
        query_tensor_input = self.query_tensors



        for b in range(B):

            world_points, mask = project_depth_to_3d(
                    depth_cond[b, :], c2w_cond[b, :], intrinsic_cond[b, :]
            )
            # ds_mask = voxel_grid_downsample_mask(world_points, 0.015, 'mean')

            # mask = ds_mask & mask

            bounded_points = norm_points_bounds(world_points, scene_bounds)
                # 对每个平面进行处理

            # ft = image_features.permute(0, 1, 3, 4, 2)
            # save_xyz_tensor_as_ply(world_points[mask], ft[b, mask][:, 0:3], "pt.ply")
            splat_feats = []

            padding_features = []
            for v in range(Nv):
                feature = image_features[b, v, :, mask[v]]

                pts = bounded_points[v][mask[v]]

                padding_feature = torch.zeros([self.feature_channels * Nv, pts.size(0)], dtype=image_features.dtype, device=image_features.device)
                
                padding_feature[v * self.feature_channels:(v+1) * self.feature_channels] = feature
                padding_features.append(padding_feature)
               
            padding_features = torch.cat(padding_features, dim=1)

            tri_plane = self.Splatting_Points_to_Plane(padding_feature, pts)
            splat_feats = torch.cat([tri_plane['xy'].unsqueeze(1), tri_plane['xz'].unsqueeze(1), tri_plane['yz'].unsqueeze(1)], dim=1)

            splat_feats = splat_feats.view(1, 3, Nv, self.feature_channels, self.grid_size, self.grid_size)


            # save_feats = splat_feats.mean(dim=2)[0]
            # for i, plane in enumerate(['xy', 'xz', 'yz']):
            #     save_feat = save_feats[i,0:3]
            #     save_feat_img = save_feat.detach().cpu()
            #     save_feat_img = (save_feat_img - save_feat_img.min()) / (save_feat_img.max() - save_feat_img.min() + 1e-8)
            #     vutils.save_image(save_feat_img, f'save_feat_{plane}.png')
            # pdb.set_trace()

        for n in range(0, self.num_layers):

            cross_view_plane_feats = []
            for i, plane in enumerate(['xy', 'xz', 'yz']):

                queries = query_tensor_input[i]  # [H, W, C]
                
                queries = self.norm1_cross_view[n](queries)

                # 重塑query和特征以进行批处理
                queries = queries.view(-1, self.feature_channels)  # [H*W, C]

                plane_feat = splat_feats[0, i].permute(2, 3, 0, 1).view(-1, Nv, self.feature_channels)
                
                # 对投影特征也进行归一化
                plane_feat = self.norm1_cross_view[n](plane_feat)

                    # 计算注意力
                attn_output, prob = self.attentions_corss_view[n](
                    queries.unsqueeze(0),  # [1, H*W, C]
                    plane_feat.unsqueeze(0) # [1, H*W, Nv, C]
                )  # [1, H*W, C]
                
                # 重塑回平面特征
                plane_feat = attn_output.squeeze(0).view(self.grid_size, self.grid_size, self.feature_channels)
        
                # 在视图维度上做softmax归一化

                res_output = query_tensor_input[i] + plane_feat

                norm_output = self.norm2_cross_view[n](res_output)

                FFN_output = self.ffn_cross_view[n](norm_output)

                res_FFN_output = res_output + FFN_output
                
                cross_view_plane_feats.append(res_FFN_output.permute(2, 0, 1).unsqueeze(0)) 

            cross_view_plane_feats = torch.cat(cross_view_plane_feats, dim=0)

            cross_plane_plane_feats = []

            # 计算跨平面注意力
            for i, plane in enumerate(['xy', 'xz', 'yz']):
                # 获取当前平面特征
                plane_feat = cross_view_plane_feats[i]  # [C, H, W]
                # 使用permute将C维度移到最后
                plane_feat = plane_feat.permute(1, 2, 0)  # [H, W, C]
                plane_feat = plane_feat.view(-1, self.feature_channels)  # [H*W, C]

                plane_feat = self.norm1_cross_plane[n](plane_feat)
                
                # 采样点
                sampled_points = self.sample_points_along_axis(plane, query_tensor_input)
                
                # 投影到其他平面并获取特征
                other_plane_features = self.project_to_other_planes(sampled_points, plane, cross_view_plane_feats)
                other_plane_features = self.norm1_cross_plane[n](other_plane_features).view(-1, self.num_samples * 2, self.feature_channels)

                attn_output, prob = self.attentions_cross_plane[n](
                        plane_feat.unsqueeze(0),  # [1, H*W, C]
                        other_plane_features.unsqueeze(0)  # [1, H*W, self.num_samples * 2, C]
                    )  # [1, H*W, C]
        
                updated_plane_data = attn_output.view(self.grid_size, self.grid_size, self.feature_channels)# [H, W, C]

                res_output2 = updated_plane_data + cross_view_plane_feats[i].permute(1, 2, 0)

                norm_output2 = self.norm2_cross_plane[n](res_output2)

                FFN_output2 = self.ffn_cross_plane[n](norm_output2)

                res_FFN_output2 = FFN_output2 + res_output2

                cross_plane_plane_feats.append(res_FFN_output2.permute(2, 0, 1).unsqueeze(0)) 


            query_tensor_input = torch.cat([cross_plane_plane_feats[0].permute(0, 2, 3, 1), 
                                            cross_plane_plane_feats[1].permute(0, 2, 3, 1), 
                                            cross_plane_plane_feats[2].permute(0, 2, 3, 1)], dim=0)
            
        query_tensor_input = query_tensor_input.permute(0, 3, 1, 2).unsqueeze(0)

        return  query_tensor_input, scene_bounds

import numpy as np

def save_xyz_tensor_as_ply(tensor, tensor2, output_file):
    """
    将 (H, W, 3) 的张量（表示 XYZ 坐标）保存为 PLY 点云文件。
    
    参数:
        tensor (numpy.ndarray): 输入张量，形状为 (H, W, 3)，表示 XYZ 坐标。
        output_file (str): 输出的 PLY 文件路径。
    """

    
    # 展平张量为 (H*W, 3)
    point_cloud = tensor.reshape(-1, 3)
    color = tensor2.reshape(-1, 3).cpu().numpy()
    
    # 归一化颜色到 0-255，只做一次归一化
    color = ((color - color.min()) / (color.max() - color.min() + 1e-8) * 255).astype(np.uint8)
    
    # 写入 PLY 文件
    with open(output_file, 'w') as f:
        # 写入头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")    # 改为 uchar
        f.write("property uchar green\n")  # 改为 uchar
        f.write("property uchar blue\n")   # 改为 uchar
        f.write("end_header\n")
        
        # 写入点云数据
        for i in range(0, len(point_cloud)):
            x, y, z = point_cloud[i]
            r, g, b = color[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def voxel_grid_downsample_mask(
    points: torch.Tensor,
    voxel_size: float,
    reduction: str = 'first'
) -> list[torch.Tensor]:
    """
    对有组织的点云进行体素网格降采样，并返回一个布尔 mask。

    Args:
        points (torch.Tensor): 输入点云，维度为 (N, H, W, C)。
                                N 是批次大小，H, W 是图像尺寸，C 是坐标维度 (例如 3 表示 x, y, z)。
        voxel_size (float): 体素网格的大小。一个较大的值会导致更多的降采样。

    Returns:
        torch.Tensor: 一个布尔 mask，维度为 (N, H, W)。
                      如果 mask[n, h, w] 为 True，表示原始点 points[n, h, w] 被保留；
                      如果为 False，表示该点被降采样移除。
    """
    if points.dim() != 4:
        raise ValueError("输入点云的维度必须是 (N, H, W, C)。")
    
    N, H, W, C = points.shape
    device = points.device

    # 初始化一个与原始点云相同形状的布尔 mask，全部设为 False
    # 之后被保留的点将设为 True
    boolean_mask = torch.full((N, H, W), False, dtype=torch.bool, device=device)

    # 将点云展平以便进行通用点云操作 (N, H*W, C)
    flat_points_batch = points.view(N, H * W, C)

    for i in range(N):
        current_points = flat_points_batch[i] # 当前视图的展平点云 (H*W, C)
        
        # 如果当前点云为空，则跳过
        if current_points.numel() == 0:
            continue

        # 计算每个点所属的体素坐标
        # 将点坐标归一化到体素网格，然后取整得到体素索引
        min_coords = current_points.min(dim=0, keepdim=True).values
        min_coords = min_coords.to(current_points.dtype)

        # 将点云平移到正空间，再进行体素化
        shifted_points = current_points - min_coords
        voxel_coords = (shifted_points / voxel_size).floor().long()

        # 计算一维体素 ID
        max_voxel_coords = voxel_coords.max(dim=0, keepdim=True).values + 1
        
        # 处理特殊情况：如果点云太小或全部在一个体素内导致 max_voxel_coords 无效
        if max_voxel_coords.numel() == 0 or torch.any(max_voxel_coords <= 0):
            # 如果只有一个点，它当然被保留
            if current_points.shape[0] == 1:
                boolean_mask[i, 0, 0] = True # 假设这个点来自 (0,0) 位置
            # 对于其他无法体素化的情况，所有点都不保留（或根据需求决定）
            continue 

        voxel_ids = voxel_coords[:, 0] \
                    + voxel_coords[:, 1] * max_voxel_coords[0, 0] \
                    + voxel_coords[:, 2] * max_voxel_coords[0, 0] * max_voxel_coords[0, 1]

        # 核心逻辑：找出每个体素中的第一个点
        # 对 voxel_ids 进行排序，并记录原始索引
        sorted_voxel_ids, sort_indices = torch.sort(voxel_ids)
        
        # 找出每个唯一 voxel_id 第一次出现的位置
        # 第一个元素总是第一个出现，后面的元素如果与前一个不同，也是第一个出现
        first_occurrence_mask_1d = torch.cat([
            torch.tensor([True], device=device), # 第一个元素总是 True
            sorted_voxel_ids[1:] != sorted_voxel_ids[:-1] # 比较相邻元素是否不同
        ])
        
        # 获取被保留的点的原始展平索引
        # 这些索引指向的是 flat_points_batch[i] 中的位置
        preserved_flat_indices = sort_indices[first_occurrence_mask_1d]

        # 将这些展平索引转换回 (H, W) 格式，并设置布尔 mask 为 True
        # 原始索引 flat_idx = h * W + w
        # 那么 h = flat_idx // W
        # w = flat_idx % W
        
        preserved_h = preserved_flat_indices // W
        preserved_w = preserved_flat_indices % W
        
        # 将对应位置的布尔 mask 设为 True
        boolean_mask[i, preserved_h, preserved_w] = True

    return boolean_mask
