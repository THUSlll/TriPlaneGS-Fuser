import torch

def grid_sample(xyz, npoint: int, scene_bounds, grid_size: int = 384):
    """
    网格采样 (优化版)
    xyz: [B, N, 3]
    npoint: 目标采样点数
    scene_bounds: tuple or list (x_min, x_max, y_min, y_max, z_min, z_max) 定义场景的边界
    grid_size: 网格大小
    返回: [B, npoint, 3] 采样后的点云
    """
    device = xyz.device
    B, N, C = xyz.shape
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    if N == 0 or npoint == 0:
        return torch.empty(B, npoint, C, dtype=xyz.dtype, device=device)

    # 1. 使用 scene_bounds 进行归一化
    # 将 scene_bounds 转换为 tensor 并调整维度以便广播
    scene_min = torch.tensor([x_min, y_min, z_min], dtype=xyz.dtype, device=device).view(1, 1, 3) # [1, 1, 3]
    scene_max = torch.tensor([x_max, y_max, z_max], dtype=xyz.dtype, device=device).view(1, 1, 3) # [1, 1, 3]

    # 计算场景范围
    scene_range = scene_max - scene_min
    # 避免除以零，对于范围为零的维度设置为1e-6
    scene_range[scene_range < 1e-6] = 1e-6

    # 使用场景边界进行归一化
    normalized_xyz = (xyz - scene_min) / scene_range # [B, N, 3]

    # 2. 计算网格索引并合并批次索引 (与原版相同)
    # 将归一化点映射到网格坐标 [0, grid_size-1]
    grid_indices = (normalized_xyz * grid_size).long()  # [B, N, 3]
    grid_indices = grid_indices.clamp(0, grid_size - 1)

    # 计算一维网格索引 (在 grid_size^3 范围内)
    grid_1d = grid_indices[..., 0] * grid_size * grid_size + \
              grid_indices[..., 1] * grid_size + \
              grid_indices[..., 2]  # [B, N]

    # 创建批次索引张量 [B, N]
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, N)

    # 合并批次索引和网格索引，形成全局唯一的索引 [B * N]
    # 每个批次的网格索引范围偏移 grid_size^3 * batch_id
    combined_grid_1d = batch_indices * (grid_size**3) + grid_1d
    combined_grid_1d_flat = combined_grid_1d.view(-1) # [B * N]

    # 将点云展平，方便后续排序和索引 [B * N, 3]
    xyz_flat = xyz.view(-1, C)

    # 3. 利用排序找到每个非空网格的代表点 (与原版相同)
    # 对合并索引进行排序，得到排序后的索引 permutation
    sort_indices = torch.argsort(combined_grid_1d_flat)

    # 按照排序后的索引重新排列点云和合并索引
    sorted_combined_grid_1d = combined_grid_1d_flat[sort_indices]
    sorted_xyz = xyz_flat[sort_indices]

    # 找到排序后索引中值发生变化的位置，这些位置对应每个非空网格的第一个点
    # prepend一个不可能的最小值，确保第一个网格也被检测到变化
    # 使用 unsqueeze(0) 确保prepend的值维度正确，与 sorted_combined_grid_1d 兼容
    change_mask = torch.diff(sorted_combined_grid_1d, prepend=torch.tensor([-1], device=device).unsqueeze(0)) != 0

    # 提取每个非空网格的第一个点 (代表点)
    unique_cell_points = sorted_xyz[change_mask] # [num_unique_cells, 3]

    # 提取这些代表点对应的原始合并索引，以便后续提取批次索引
    unique_combined_grid_1d = sorted_combined_grid_1d[change_mask]

    # 从合并索引中恢复批次索引
    unique_batch_indices = (unique_combined_grid_1d // (grid_size**3)).long() # [num_unique_cells]

    # 4. 基于目标点数 npoint 进行最终采样/填充 (仍然需要一个批次循环，但内部操作已大大简化) (与原版相同)
    sampled_xyz = torch.empty(B, npoint, C, dtype=xyz.dtype, device=device)

    for b in range(B):
        # 找到属于当前批次的代表点
        batch_unique_points_mask = unique_batch_indices == b
        batch_unique_points = unique_cell_points[batch_unique_points_mask] # [num_unique_cells_in_batch, 3]
        num_unique_in_batch = batch_unique_points.shape[0]

        if num_unique_in_batch >= npoint:
            # 如果当前批次的非空网格数量 >= 目标点数，则从这些代表点中随机采样 npoint 个
            choice = torch.randperm(num_unique_in_batch, device=device)[:npoint]
            sampled_xyz[b] = batch_unique_points[choice]
        elif num_unique_in_batch > 0:
            # 如果非空网格数量 < 目标点数，且 > 0，则选取所有代表点，然后随机重复填充至 npoint
            # 先选择所有 unique points
            sampled_points = batch_unique_points
            # 计算需要重复的点数
            num_to_repeat = npoint - num_unique_in_batch
            # 随机选择要重复的点的索引
            repeat_indices = torch.randint(0, num_unique_in_batch, (num_to_repeat,), device=device)
            # 重复点并拼接
            repeated_points = batch_unique_points[repeat_indices]
            sampled_xyz[b] = torch.cat([sampled_points, repeated_points], dim=0)
        else:
             # 如果当前批次没有点落入任何网格 (例如点云范围极小或网格太大)，或者原始点数为0
             # 回退到从原始点中随机采样 (如果N>0)，否则返回0点
             if N > 0:
                 # 随机从原始 N 个点中采样 npoint 个点 (可能重复)
                 # 需要处理原始点数 N < npoint 的情况，使用 replacement=True
                 choice = torch.randint(0, N, (npoint,), device=device)
                 sampled_xyz[b] = xyz[b, choice]
             else:
                 # 如果原始点数就是0，则填充0向量
                 sampled_xyz[b] = torch.zeros(npoint, C, dtype=xyz.dtype, device=device)

    return sampled_xyz
import torch

def grid_sample_centers(xyz, scene_bounds, grid_size: int = 384):
    """
    网格采样 (返回所有有点的网格中心)
    xyz: [B, N, 3]
    scene_bounds: tuple or list (x_min, x_max, y_min, y_max, z_min, z_max) 定义场景的边界
    grid_size: 网格大小
    返回: list of tensors, 列表长度为 B (批次大小)。
          list[i] 是一个形状为 [num_unique_in_batch_i, 3] 的 tensor，
          包含批次 i 中所有非空网格的中心点坐标。
    """
    device = xyz.device
    B, N, C = xyz.shape
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    # 如果输入点云为空，则返回一个包含 B 个空 tensor 的列表
    if N == 0:
        return [torch.empty(0, C, dtype=xyz.dtype, device=device) for _ in range(B)]

    # 1. 使用 scene_bounds 进行归一化
    # 将 scene_bounds 转换为 tensor 并调整维度以便广播
    # 修复: 将 .view(1, 1, 3) 改为 .view(1, 3)
    scene_min = torch.tensor([x_min, y_min, z_min], dtype=xyz.dtype, device=device).view(1, 3) # [1, 3]
    scene_max = torch.tensor([x_max, y_max, z_max], dtype=xyz.dtype, device=device).view(1, 3) # [1, 3]

    # 计算场景范围
    scene_range = scene_max - scene_min # [1, 3]
    # 避免除以零，对于范围为零的维度设置为一个很小的值
    scene_range_safe = scene_range.clone() # 避免修改原 tensor
    scene_range_safe[scene_range_safe < 1e-6] = 1e-6

    # 使用场景边界进行归一化，将点云映射到 [0, 1] 的范围
    # xyz [B, N, 3], scene_min [1, 3], scene_range_safe [1, 3] -> Broadcasting works: [B, N, 3]
    normalized_xyz = (xyz - scene_min) / scene_range_safe # [B, N, 3]

    # 2. 计算网格索引并合并批次索引
    # 将归一化点映射到网格坐标 [0, grid_size-1]
    # 注意：这里映射到的是网格的离散索引
    grid_indices = (normalized_xyz * grid_size).long()  # [B, N, 3]
    grid_indices = grid_indices.clamp(0, grid_size - 1) # 确保索引在有效范围内

    # 计算一维网格索引 (在 grid_size^3 范围内)，每个批次独立
    grid_1d = grid_indices[..., 0] * grid_size * grid_size + \
              grid_indices[..., 1] * grid_size + \
              grid_indices[..., 2]  # [B, N]

    # 创建批次索引张量 [B, N]
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, N)

    # 合并批次索引和网格索引，形成全局唯一的索引 [B * N]
    # 每个批次的网格索引范围偏移 grid_size^3 * batch_id
    combined_grid_1d = batch_indices * (grid_size**3) + grid_1d
    combined_grid_1d_flat = combined_grid_1d.view(-1) # [B * N]

    # 3. 利用排序找到每个非空网格的唯一一维索引
    # 对合并索引进行排序，得到排序后的索引 permutation
    sort_indices = torch.argsort(combined_grid_1d_flat)

    # 按照排序后的索引重新排列合并索引
    sorted_combined_grid_1d = combined_grid_1d_flat[sort_indices]

    # 找到排序后索引中值发生变化的位置，这些位置对应每个非空网格的第一个点
    # prepend一个不可能的最小值，确保第一个网格也被检测到变化
    change_mask = torch.diff(sorted_combined_grid_1d, prepend=torch.tensor([-1], device=device)) != 0

    # 提取每个非空网格对应的唯一的合并一维索引
    unique_combined_grid_1d = sorted_combined_grid_1d[change_mask] # [num_unique_cells]

    # 如果没有任何点落在任何网格内，unique_combined_grid_1d 可能为空
    if unique_combined_grid_1d.numel() == 0:
         return [torch.empty(0, C, dtype=xyz.dtype, device=device) for _ in range(B)]

    # 4. 从唯一合并索引中恢复网格的 [gx, gy, gz] 索引和批次索引
    # 恢复批次索引
    unique_batch_indices = (unique_combined_grid_1d // (grid_size**3)).long() # [num_unique_cells]

    # 恢复该批次内的网格一维索引
    unique_grid_1d_in_batch = unique_combined_grid_1d % (grid_size**3) # [num_unique_cells]

    # 从一维网格索引恢复三维网格索引 [gx, gy, gz]
    unique_gz = unique_grid_1d_in_batch % grid_size
    unique_gy = (unique_grid_1d_in_batch // grid_size) % grid_size
    unique_gx = unique_grid_1d_in_batch // (grid_size * grid_size)

    # 组合成三维网格索引张量 [num_unique_cells, 3]
    unique_grid_indices_3d = torch.stack([unique_gx, unique_gy, unique_gz], dim=-1)

    # 5. 计算这些网格中心的坐标
    # 网格中心在 [0, grid_size] 尺度下的坐标是 (gx + 0.5, gy + 0.5, gz + 0.5)
    unique_grid_centers_grid_scale = unique_grid_indices_3d.float() + 0.5 # [num_unique_cells, 3]

    # 将网格中心的坐标从 [0, grid_size] 尺度映射回原始场景坐标
    # original_center = (grid_center_in_grid_scale / grid_size) * scene_range + scene_min
    # unique_grid_centers_grid_scale [num_unique_cells, 3]
    # scene_range [1, 3], scene_min [1, 3]
    # Broadcasting works: [num_unique_cells, 3] * [1, 3] -> [num_unique_cells, 3]
    unique_cell_centers = (unique_grid_centers_grid_scale / grid_size) * scene_range + scene_min # [num_unique_cells, 3]

    # 6. 按批次分割唯一的网格中心
    sampled_centers_list = []
    for b in range(B):
        # 找到属于当前批次的唯一网格中心
        batch_unique_centers_mask = unique_batch_indices == b # [num_total_unique_cells]
        # unique_cell_centers [num_total_unique_cells, 3]
        # Indexing with [num_total_unique_cells] mask correctly selects rows
        batch_unique_centers = unique_cell_centers[batch_unique_centers_mask] # [num_unique_cells_in_batch, 3]
        sampled_centers_list.append(batch_unique_centers.unsqueeze(0))

    return sampled_centers_list

import math

def remove_outliers_knn(
    points: torch.Tensor,
    k: int = 100,
    std_ratio: float = 5.0,
    chunk_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KNN 孤点移除 (优化 V2: 批次并行 + 分块 - 速度和显存优化)。

    并行处理批次维度 (B) 以提高速度，同时沿点数维度 (N) 分块以控制显存。
    可选择使用 @torch.compile 进一步加速 (需要 PyTorch 2.x+)。

    Args:
        points (torch.Tensor): 输入点云 (B, N, 3)。
        k (int): 最近邻数量。
        std_ratio (float): 标准差阈值比例。
        chunk_size (int): 沿 N 维度的点块大小，用于显存控制。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - inlier_mask (torch.Tensor): 布尔掩码 (B, N), True 为内点。
            - avg_distances (torch.Tensor): 平均 k 邻居距离 (B, N)。
    """
    B, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    if N <= k:
        print(f"警告: N ({N}) <= k ({k})。将返回所有点。")
        return torch.ones((B, N), dtype=torch.bool, device=device), torch.zeros((B, N), device=device, dtype=dtype)

    # --- 核心计算: 批次并行 + N维度分块 ---
    # 预分配存储所有批次的平均距离
    all_avg_distances = torch.empty(B, N, device=device, dtype=dtype)

    # 沿 N 维度分块处理
    num_chunks = math.ceil(N / chunk_size)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, N)
        actual_chunk_size = end_idx - start_idx

        if actual_chunk_size == 0:
            continue

        # 获取所有批次的当前点块: Shape (B, actual_chunk_size, 3)
        points_chunk = points[:, start_idx:end_idx, :]

        # 批次化距离计算: (B, chunk, 3) vs (B, N, 3) -> (B, chunk, N)
        # cdist 自动处理批次维度，这是速度提升的关键
        chunk_dists = torch.cdist(points_chunk, points, p=2.0)

        # 批次化 Top K: 在每个批次的 N 维度上找 K+1 个最小距离
        # 输入: (B, actual_chunk_size, N), dim=2 是 N 维度
        # 输出: (B, actual_chunk_size, k+1)
        knn_dists_chunk, _ = torch.topk(chunk_dists, k + 1, dim=2, largest=False, sorted=True)

        # 批次化计算平均距离 (排除自身)
        # 输入: (B, actual_chunk_size, k+1), 切片后是 (B, actual_chunk_size, k)
        # 输出: (B, actual_chunk_size)
        # 使用 ... 省略号来保持批次和块维度
        avg_dist_chunk = torch.mean(knn_dists_chunk[..., 1:], dim=2)

        # 存储当前块的结果到所有批次
        all_avg_distances[:, start_idx:end_idx] = avg_dist_chunk

        # 清理不再需要的中间变量 (可选，torch 会自动管理，但显式删除可能有助于理解)
        del points_chunk, chunk_dists, knn_dists_chunk, avg_dist_chunk
        # if device.type == 'cuda': torch.cuda.empty_cache() # 谨慎使用

    # --- 所有块处理完毕 ---

    # 计算每个批次内部的均值和标准差 (沿 N 维度, dim=1)
    # keepdim=True 保留维度用于后续广播
    mean_dist = torch.mean(all_avg_distances, dim=1, keepdim=True) # Shape: (B, 1)
    std_dev = torch.std(all_avg_distances, dim=1, keepdim=True)   # Shape: (B, 1)

    # 计算每个批次的阈值
    threshold = mean_dist + std_ratio * std_dev # Shape: (B, 1)

    # 使用广播生成最终掩码
    # 比较 (B, N) <= (B, 1) -> 得到 (B, N)
    final_mask = all_avg_distances <= threshold

    return final_mask, all_avg_distances

def remove_outliers_radius_pytorch(
    points: torch.Tensor,
    radius: float,
    min_points: int,
    chunk_size: int = 1024 # 仍然保持分块以控制显存
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于 PyTorch 实现的半径离群点移除 (Radius Outlier Removal)。

    并行处理批次维度 (B)，沿点数维度 (N) 分块以控制显存。

    Args:
        points (torch.Tensor): 输入点云 (B, N, 3)。
        radius (float): 邻居搜索的半径。
        min_points (int): 在给定半径内，一个点被认为是内点所需的最小邻居数量。
        chunk_size (int): 沿 N 维度的点块大小，用于显存控制。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - inlier_mask (torch.Tensor): 布尔掩码 (B, N), True 为内点。
            - neighbor_counts (torch.Tensor): 每个点在半径内的邻居数量 (B, N)。
    """
    B, N, _ = points.shape
    device = points.device

    if N <= min_points:
        print(f"警告: N ({N}) <= min_points ({min_points})。将返回所有点。")
        return torch.ones((B, N), dtype=torch.bool, device=device), \
               torch.full((B, N), N, dtype=torch.long, device=device) # 所有点都有 N 个邻居

    all_neighbor_counts = torch.empty(B, N, dtype=torch.long, device=device)

    num_chunks = math.ceil(N / chunk_size)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, N)
        actual_chunk_size = end_idx - start_idx

        if actual_chunk_size == 0:
            continue

        points_chunk = points[:, start_idx:end_idx, :] # (B, actual_chunk_size, 3)

        # 批次化距离计算: (B, chunk, 3) vs (B, N, 3) -> (B, chunk, N)
        chunk_dists = torch.cdist(points_chunk, points, p=2.0)

        # 统计在半径内的邻居数量 (包括点本身)
        # (B, actual_chunk_size, N) -> (B, actual_chunk_size)
        neighbor_counts_chunk = torch.sum(chunk_dists <= radius, dim=2)

        # 存储当前块的结果
        all_neighbor_counts[:, start_idx:end_idx] = neighbor_counts_chunk

        del points_chunk, chunk_dists, neighbor_counts_chunk

    # 创建内点掩码
    # 注意：这里默认将点本身也算作一个邻居。如果不想算自己，可以减1，但通常ROR计算是包含自身的。
    inlier_mask = all_neighbor_counts >= min_points

    return inlier_mask, all_neighbor_counts


def farthest_point_sample(xyz, npoint):
    """
    优化的最远点采样
    xyz: [B, N, 3]
    npoint: 采样点数
    返回: [B, npoint, 3] 采样后的点云
    """
    # 如果点数太多，使用网格采样
    if xyz.shape[1] > 100000:
        return grid_sample(xyz, npoint)
        
    device = xyz.device
    B, N, C = xyz.shape
    
    # 使用随机初始点
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    # 预分配距离矩阵
    distance = torch.ones(B, N).to(device) * 1e10
    
    # 使用向量化操作
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        # 使用向量化计算距离
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        # 使用掩码更新距离
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # 使用argmax找到最远点
        farthest = torch.max(distance, -1)[1]
    
    # 使用高级索引获取采样点
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1).expand(-1, npoint)
    sampled_xyz = xyz[batch_indices, centroids]
    
    return sampled_xyz
