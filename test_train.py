from tgs.models.triplane.triplane_splatting import PointTriplaneGenerator
from tgs.models.triplane.triplane_transformerV3 import ImageTriplaneGenerator
import torch
import torch.nn as nn
import torch.optim as optim

# 实例化模型
# pointTriplaneGenerator = PointTriplaneGenerator(196, 384, 228).cuda()

# # 随机输入
# B, N, C = 1, 132253, 387
# P, out_C, H, W = 3, 228, 196, 196


# gt = torch.randn(B, P, out_C, H, W, requires_grad=False, device='cuda')

# # 优化器（这里只优化输入，实际训练应优化模型参数）

# for step in range(3):
#     dense_point_feature = torch.randn(B, N, C, requires_grad=True, device='cuda')
#     weight = torch.randn(B, N, requires_grad=True, device='cuda')

#     padding = 0.05
#     scene_bounds = (
#                 dense_point_feature[:, :, 0].min() - padding * (dense_point_feature[:, :, 0].max() - dense_point_feature[:, :, 0].min()),
#                 dense_point_feature[:, :, 0].max() + padding * (dense_point_feature[:, :, 0].max() - dense_point_feature[:, :, 0].min()),
#                 dense_point_feature[:, :, 1].min() - padding * (dense_point_feature[:, :, 1].max() - dense_point_feature[:, :, 1].min()),
#                 dense_point_feature[:, :, 1].max() + padding * (dense_point_feature[:, :, 1].max() - dense_point_feature[:, :, 1].min()),
#                 dense_point_feature[:, :, 2].min() - padding * (dense_point_feature[:, :, 2].max() - dense_point_feature[:, :, 2].min()),
#                 dense_point_feature[:, :, 2].max() + padding * (dense_point_feature[:, :, 2].max() - dense_point_feature[:, :, 2].min()),
#             )

#     optimizer = optim.Adam([dense_point_feature, weight], lr=1e-3)
#     optimizer.zero_grad()
#     output = pointTriplaneGenerator(None, dense_point_feature, weight, scene_bounds)
#     loss = nn.functional.mse_loss(output, gt)
#     loss.backward()
#     print(...)
#     print(f"Step {step} | Loss: {loss.item()} | dense_point_feature.grad mean: {dense_point_feature.grad.mean().item()} | weight.grad mean: {weight.grad.mean().item()}")
#     optimizer.step()



triplaneGenerator = ImageTriplaneGenerator(128, feature_channels=228)

