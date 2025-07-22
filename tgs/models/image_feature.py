from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from tgs.utils.base import BaseModule
from tgs.utils.ops import compute_distance_transform
from tgs.utils.typing import *
import pdb
class ImageFeature(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        use_rgb: bool = True
        use_feature: bool = True
        use_mask: bool = True
        feature_dim: int = 128
        out_dim: int = 133
        backbone: str = "default"
        freeze_backbone_params: bool = True

    cfg: Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # 添加特征降维卷积
        self.feature_reduce = nn.Sequential(
            nn.Conv2d(768, 384, 1),
            nn.ReLU(),
            nn.Conv2d(384, 192, 1)
        )

    def forward(self, rgb, mask=None, feature=None):
        B, Nv, C, H, W = rgb.size()

        rgb = rearrange(rgb, "B Nv C H W -> (B Nv) C H W")
        if mask is not None:
            mask = rearrange(mask, "B Nv C H W -> (B Nv) C H W")

        assert feature is not None
        feature = rearrange(feature, "B (Nv Nt) C -> (B Nv) Nt C", Nv=Nv)
        feature = feature[:, 1:].reshape(B * Nv, H // 14, W // 14, -1).permute(0, 3, 1, 2).contiguous()

        target_h, target_w = H // 1, W // 1
        feature = F.interpolate(feature, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # 在融合RGB之前进行特征降维
        # feature = self.feature_reduce(feature)  # [B*Nv, 192, H/4, W/4]

        if mask is not None and mask.is_floating_point():
            mask = mask > 0.5
            mask = F.interpolate(mask.float(), size=(target_h, target_w), mode='nearest')
            mask = mask > 0.5
        
        image_features = []
        if self.cfg.use_rgb:
            rgb = F.interpolate(rgb, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image_features.append(rgb)
        if self.cfg.use_feature:
            image_features.append(feature)
        if mask is not None:
            image_features += [mask, compute_distance_transform(mask)]

        image_features = torch.cat(image_features, dim=1)
        return rearrange(image_features, "(B Nv) C H W -> B Nv C H W", B=B, Nv=Nv).squeeze(1)
    
class ProgressiveFPN(nn.Module):
    def __init__(self, channels_list=[256, 128, 64, 32]):
        super().__init__()
        # 横向连接卷积（调整encoder特征通道）
        self.lat_convs = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1) for ch in channels_list
        ])
        
        # 自上而下的融合卷积
        self.top_down_convs = nn.ModuleList()
        for i in range(len(channels_list)-1):
            self.top_down_convs.append(
                nn.Conv2d(channels_list[i], channels_list[i+1], 3, padding=1)
            )
            
    def forward(self, encoder_feats, decoder_feats):
        """
        encoder_feats: 列表 [enc4, enc3, enc2, enc1]
        decoder_feats: 列表 [dec4, dec3, dec2, dec1]
        """
        # 自顶向下融合
        fused = []

        x = decoder_feats[0]  # 从dec4开始
        for i in range(len(self.top_down_convs)):
            # 与encoder特征融合
            lateral = self.lat_convs[i](encoder_feats[i])

            x = x + lateral
            
            # 上采样并与下一级decoder特征融合
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.top_down_convs[i](x)
            x = x + decoder_feats[i+1]
            
            fused.append(x)
        
        return fused[::-1]  # 返回[低层→高层]
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        # 编码器（下采样路径）[[2]][[6]]
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features*2, features*4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features*4, features*8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层（最低层）[[5]]
        self.bottleneck = self._block(features*8, features*16, name="bottleneck")

        # 解码器（上采样路径）[[7]]
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features*16, features*8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features*8, features*4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features*4, features*2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features*2, features, name="dec1")

        # 最终输出层[[4]]
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.fpn = ProgressiveFPN(
            channels_list=[init_features*8, init_features*4, init_features*2, init_features]
        )

    def forward(self, x):
        # 编码路径（保存所有层级特征）[[6]]
        enc1 = self.encoder1(x)          # [B, 32, H, W]
        enc2 = self.encoder2(self.pool1(enc1))  # [B, 64, H/2, W/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [B, 128, H/4, W/4]
        enc4 = self.encoder4(self.pool3(enc3))  # [B, 256, H/8, W/8]

        encoder_feats = [enc4, enc3, enc2, enc1]

        # 瓶颈层[[5]]         # [B, 512, H/16, W/16]
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 解码路径（保存上采样特征）[[7]]
        dec4 = self.upconv4(bottleneck)        # [B, 256, H/8, W/8]
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)             # [B, 256, H/8, W/8]

        dec3 = self.upconv3(dec4)              # [B, 128, H/4, W/4]
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)             # [B, 128, H/4, W/4]

        dec2 = self.upconv2(dec3)              # [B, 64, H/2, W/2]
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)             # [B, 64, H/2, W/2]

        dec1 = self.upconv1(dec2)              # [B, 32, H, W]
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)             # [B, 32, H, W]
        decoder_feats = [dec4, dec3, dec2, dec1]
        fused_features = self.fpn(encoder_feats, decoder_feats)
        # 收集所有层级特征[[2]][[7]]

        features = {
            'dec3': fused_features[2],    # 1/4分辨率
            'dec2': fused_features[1],    # 1/2分辨率
            'dec1': fused_features[0],    # 原始分辨率
            'fused': self.conv(fused_features[0]) # 最终输出
        }
        # features['fused'] = self.fusion(features)
      
        return features
    
    @staticmethod
    def _block(in_channels, features, name):
        # 基础卷积块（两次卷积）[[2]][[8]]
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
def add_positional_encoding(x):
    """
    在通道维度添加行、列坐标编码
    x: 输入张量，形状为 [B, C, H, W]
    返回: [B, C+2, H, W]
    """
    B, C, H, W = x.shape
    
    # 生成行/列坐标（归一化到 [-1, 1]）
    h_coords = torch.linspace(-1, 1, H, device=x.device)  # [H]
    w_coords = torch.linspace(-1, 1, W, device=x.device)  # [W]
    mesh_h, mesh_w = torch.meshgrid(h_coords, w_coords, indexing='ij')  # [H, W], [H, W]
    
    # 扩展维度以匹配输入张量
    mesh_h = mesh_h.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)  # [B, 1, H, W]
    mesh_w = mesh_w.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)  # [B, 1, H, W]
    
    # 拼接位置编码到通道维度 [[3]][[5]]
    x = torch.cat([x, mesh_h, mesh_w], dim=1)  # [B, C+2, H, W]
    return x

def flatten_spatial(x):
    """
    将 H 和 W 维度展平为 L=H*W
    x: 输入张量，形状为 [B, C, H, W]
    返回: [B, C, H*W]
    """
    return x.view(x.size(0), x.size(1), -1)  # [B, C, H*W] [[7]][[9]]

def add_plucker_channels(images: torch.Tensor, 
                        poses: torch.Tensor, 
                        intri: torch.Tensor) -> torch.Tensor:
    """
    在图像像素通道中添加普鲁克坐标
    
    Args:
        images (torch.Tensor): [B, H, W, 3] RGB图像
        poses (torch.Tensor): [B, 3, 4] 相机外参（R|t）
        intri (torch.Tensor): [B, 3, 3] 相机内参矩阵
    
    Returns:
        torch.Tensor: [B, H, W, 9] 增加普鲁克坐标后的图像
    """

    B, C, H, W = images.shape
    
    # 计算内参逆矩阵 [[1]]
    K_inv = torch.inverse(intri)  # [B,3,3]
    
    # 生成像素网格坐标 [[5]]
    u = torch.arange(W, device=images.device, dtype=images.dtype)
    v = torch.arange(H, device=images.device, dtype=images.dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    uv_hom = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)  # [H,W,3]
    uv_hom = uv_hom.view(1, H, W, 3, 1).expand(B, -1, -1, -1, -1)  # [B,H,W,3,1]
    
    # 计算相机坐标系下的方向向量 [[7]]
    d_cam = torch.matmul(K_inv.view(B,1,1,3,3), uv_hom).squeeze(-1)  # [B,H,W,3]
    
    # 提取外参的旋转和平移 [[6]]
    R = poses[:, :3, :3]  # [B,3,3] 取左上3×3旋转矩阵
    t = poses[:, :3, 3]   # [B,3] 取第四列前三行作为平移向量
    
    # 转换方向向量到世界坐标系 [[1]]
    R_T = R.transpose(-1, -2)  # [B,3,3]
    d_cam_flat = d_cam.view(B, 3, H*W)  # [B,3,HW]
    d_world_flat = torch.bmm(R_T, d_cam_flat)  # [B,3,HW]
    d_world = d_world_flat.view(B, H, W, 3)  # [B,H,W,3]
    
    # 计算矩向量（m = t × d_world） [[2]]
    t_expanded = t.view(B,1,1,3).expand(-1,H,W,-1)  # [B,H,W,3]
    m_world = torch.cross(t_expanded, d_world, dim=-1)  # [B,H,W,3]
    
    # 拼接普鲁克坐标通道 [[10]]
    plucker = torch.cat([d_world, m_world], dim=-1)  # [B,H,W,6]

    plucker = rearrange(plucker, 'B H W C -> B C H W')

    # 合并原始图像和普鲁克通道
    return torch.cat([images, plucker], dim=1)  # [B,9,H,W]

def create_plucker(images: torch.Tensor, 
                  poses: torch.Tensor, 
                  intri: torch.Tensor) -> torch.Tensor:
    """
    在图像像素通道中添加普鲁克坐标
    
    Args:
        images (torch.Tensor): [B, N, C, H, W] RGB图像
        poses (torch.Tensor): [B, N, 3, 4] 相机外参（R|t）
        intri (torch.Tensor): [B, N, 3, 3] 相机内参矩阵
    
    Returns:
        torch.Tensor: [B, N, 6, H, W] 增加普鲁克坐标后的张量
    """

    B, N, C, H, W = images.shape
    
    # 计算内参逆矩阵 [[1]]
    K_inv = torch.inverse(intri)  # [B,N,3,3]
    
    # 生成像素网格坐标 [[5]]
    u = torch.arange(W, device=images.device, dtype=images.dtype)
    v = torch.arange(H, device=images.device, dtype=images.dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # [H, W]
    uv_hom = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)  # [H,W,3]
    uv_hom = uv_hom.view(1, 1, H, W, 3, 1).expand(B, N, H, W, 3, 1)  # [B,N,H,W,3,1]
    
    # 计算相机坐标系下的方向向量 [[7]]
    K_inv = K_inv.view(B, N, 1, 1, 3, 3)  # [B,N,1,1,3,3]
    d_cam = torch.matmul(K_inv, uv_hom).squeeze(-1)  # [B,N,H,W,3]
    
    # 提取外参的旋转和平移 [[6]]
    R = poses[:, :, :3, :3]  # [B,N,3,3]
    t = poses[:, :, :3, 3]   # [B,N,3]
    
    # 转换方向向量到世界坐标系 [[1]]
    R_T = R.transpose(-1, -2)  # [B,N,3,3]
    d_cam_flat = d_cam.permute(0,1,4,2,3).contiguous().view(B, N, 3, H*W)  # [B,N,3,HW]
    d_world_flat = torch.matmul(R_T, d_cam_flat)  # [B,N,3,HW]
    d_world = d_world_flat.view(B, N, 3, H, W).permute(0,1,3,4,2)  # [B,N,H,W,3]
    
    # 计算矩向量（m = t × d_world） [[2]]
    t_expanded = t.view(B, N, 1, 1, 3).expand(-1, -1, H, W, -1)  # [B,N,H,W,3]
    m_world = torch.cross(t_expanded, d_world, dim=-1)  # [B,N,H,W,3]
    
    # 拼接普鲁克坐标通道 [[10]]
    plucker = torch.cat([d_world, m_world], dim=-1)  # [B,N,H,W,6]

    return plucker