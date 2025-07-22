import torch
import torch.nn as nn
import pdb
from diff_gaussian_rasterization_ch384 import GaussianRasterizer, GaussianRasterizationSettings
from gaussian.cameras import get_ortho_camera_params
# ... 其他代码 ...
def norm_points_bounds(world_points: torch.Tensor, scene_bounds: tuple = (-1, 1, -1, 1,-1, 1)):
    normed_x = 2 * (world_points[:, 0] - scene_bounds[0]) / (scene_bounds[1] - scene_bounds[0]) - 1
    normed_y = 2 * (world_points[:, 1] - scene_bounds[2]) / (scene_bounds[3] - scene_bounds[2]) - 1
    normed_z = 2 * (world_points[:, 2] - scene_bounds[4]) / (scene_bounds[5] - scene_bounds[4]) - 1

    return torch.cat([normed_x.unsqueeze(1), normed_y.unsqueeze(1), normed_z.unsqueeze(1), world_points[:, 3:]], dim=-1)
    



class PointTriplaneGenerator(nn.Module):
    def __init__(self, grid_size=128, in_channels = 196, out_channels=196):
        super().__init__()
        self.grid_size = grid_size
        self.fuse_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
                # 初始化三平面的query tensor，每个网格单元一个query
        bg_color = torch.zeros(in_channels, device="cuda")
        # self.query_tensors = nn.ParameterList([
        #     nn.Parameter(torch.randn(grid_size, grid_size, feature_channels)) for _ in range(3)
        # ])
        
        # # 初始化注意力模块
        # self.attentions_corss_view = TriplaneAttention(
        #         query_dim=feature_channels, 
        #         context_dim=feature_channels, 
        #         n_heads=n_heads
        # )

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

        raster_settings_XY = GaussianRasterizationSettings(
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

        raster_settings_YZ = GaussianRasterizationSettings(
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
        
        raster_settings_XZ = GaussianRasterizationSettings(
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
        
        self.rasterizer_XY = GaussianRasterizer(raster_settings_XY)
        self.rasterizer_YZ = GaussianRasterizer(raster_settings_YZ)
        self.rasterizer_XZ = GaussianRasterizer(raster_settings_XZ)


    def forward(self, central_point_feature, dense_point_feature, weight, scene_bounds):
        B, N, C = dense_point_feature.shape
        
        for b in range(B):

            bounded_dense_point_feature = norm_points_bounds(dense_point_feature[b], scene_bounds)
            bounded_dense_point_feature = torch.cat([bounded_dense_point_feature[...,:3], weight[b].unsqueeze(-1), bounded_dense_point_feature[..., 3:]], dim=-1)
            tri_planes = self.Splatting_Points_to_Plane(bounded_dense_point_feature)

        plane_feats = torch.cat([tri_planes['xy'].unsqueeze(1), tri_planes['xz'].unsqueeze(1), tri_planes['yz'].unsqueeze(1)], dim=1)

        B, P, C, H, W = plane_feats.shape
        plane_feats = plane_feats.view(B * P, C, H, W)
        plane_feats = self.fuse_conv(plane_feats)  # [B*P, out_channels, H, W]

        plane_feats = plane_feats.view(B, P, -1, H, W)

        return plane_feats
    
    def Splatting_Points_to_Plane(self, bounded_dense_point_feature):
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
        coords = bounded_dense_point_feature[:, :3]  # 前三位是坐标
        weight = bounded_dense_point_feature[:, 3:4]     # 第四位是权重
        features = bounded_dense_point_feature[:, 4:] # 特征

        screenspace_points = torch.zeros_like(coords, dtype=coords.dtype, requires_grad=True, device=coords.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points

        shs = None

        colors_precomp = features * weight
        cov3D_precomp = None
        scales = torch.full([bounded_dense_point_feature.size(0), 3], 1/self.grid_size, dtype=torch.float32, device="cuda").requires_grad_(True)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        rotations = identity_quat.repeat(bounded_dense_point_feature.size(0), 1).requires_grad_(True)
        opacities = torch.full([bounded_dense_point_feature.size(0), 1], 0.0067, dtype=torch.float32, device="cuda").requires_grad_(True)
        self.opacities = opacities
        # 对每个平面进行投影
        tri_planes = {}
        for plane in ['xy', 'xz', 'yz']:
            # 选择对应的坐标轴
            if plane == 'xy':
                rasterizer = self.rasterizer_XY
            if plane == 'xz':
                rasterizer = self.rasterizer_XZ
            if plane == 'yz':
                rasterizer = self.rasterizer_YZ          

            rendered_image, radii, depth = rasterizer(
                means3D = coords,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacities,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            tri_planes[plane] = rendered_image.unsqueeze(0)
        return tri_planes
            
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

# def project_points_to_planes(bounded_dense_point_feature, tri_planes, b):
#     """
#     将归一化后的点云投影到三平面上，并考虑透明度权重
    
#     参数:
#         bounded_GS_feats: 归一化后的点云特征，形状为 [N, C]
#                          其中前三位是坐标，第四位是透明度，后面是特征
#         tri_planes: 已经初始化的三平面字典，包含'xy'、'xz'、'yz'三个键
        
#     返回:
#         直接修改传入的tri_planes
#     """
#     # 分离坐标、透明度和特征
#     coords = bounded_dense_point_feature[:, :3]  # 前三位是坐标
#     weight = bounded_dense_point_feature[:, 3]     # 第四位是权重
#     features = bounded_dense_point_feature[:, 4:] # 特征
    
#     # 对每个平面进行投影
#     for plane in ['xy', 'xz', 'yz']:
#         # 选择对应的坐标轴
#         if plane == 'xy':
#             plane_coords = coords[:, [0, 1]]
#         elif plane == 'xz':
#             plane_coords = coords[:, [0, 2]]
#         else:  # yz
#             plane_coords = coords[:, [1, 2]]
            
#         # 将坐标映射到网格索引
#         grid_size = tri_planes[plane][b].shape[-1]
#         grid_coords = (plane_coords * 0.5 + 0.5) * (grid_size - 1)
#         grid_coords = grid_coords.long().clamp(0, grid_size-1)
        
#         # 展平坐标
#         flat_coords = grid_coords.view(-1, 2)
#         flat_features = features.reshape(features.shape[1], -1)
        
#         # 创建扁平化的输出张量
#         plane_feat_flat = torch.zeros((features.shape[1], grid_size * grid_size),
#                                     device=features.device)
        
#         # 构造1D索引
#         index_1d = flat_coords[:, 0] * grid_size + flat_coords[:, 1]
#         index = index_1d[None, :].expand(features.shape[1], -1)
        
#         # 使用透明度作为权重进行散射操作
#         weighted_features = flat_features * weight[None, :]
#         plane_feat_flat.scatter_reduce_(
#             dim=1,
#             index=index,
#             src=weighted_features,
#             reduce="sum",
#             include_self=False
#         )
        
#         # 计算权重和
#         weight_sum = torch.zeros(grid_size * grid_size, device=features.device)
#         weight_sum.scatter_reduce_(
#             dim=0,
#             index=index_1d,
#             src=weight,
#             reduce="sum",
#             include_self=False
#         )
        
#         # 避免除以零
#         weight_sum = weight_sum.clamp(min=1e-6)
        
#         # 归一化特征
#         plane_feat_flat = plane_feat_flat / weight_sum[None, :]
        
#         # 重塑为2D平面并更新tri_planes
#         tri_planes[plane][b] = plane_feat_flat.view(features.shape[1], grid_size, grid_size)


