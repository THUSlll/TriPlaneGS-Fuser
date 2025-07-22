import json
import math
from dataclasses import dataclass, field

import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from tgs.utils.config import parse_structured

from tgs.utils.typing import *

from tgs.utils.camera_utils import JSON_to_camera, cameraList_from_camInfos
from gaussian.gaussian_pcd import load_gaussian_ply
from tgs.models.pointclouds.samplefarthest import remove_outliers_knn, remove_outliers_radius_pytorch

import pdb
from tqdm import tqdm

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


@dataclass
class CustomDatasetConfig:
    image_list: Any = ""
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )

    relative_pose: bool = False
    cond_height: int = 512
    cond_width: int = 512
    cond_camera_distance: float = 1.6
    cond_fovy_deg: float = 40.0
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 0.0
    eval_camera_distance: float = 1.6
    eval_fovy_deg: float = 40.0
    n_test_views: int = 120
    num_views_output: int = 120
    only_3dgs: bool = False

    source_path: str = ""
    resolution: int = 1
    data_device: str = "cuda"


class CustomGaussianFuseDataset(Dataset):
    def __init__(self, istest, cfg: Any) -> None:
        super().__init__()
        self.istest = istest
        self.cfg: CustomDatasetConfig = parse_structured(CustomDatasetConfig, cfg)
        if istest:
            self.source_path = os.path.join(self.cfg.source_path, "test")
        else:
            self.source_path = os.path.join(self.cfg.source_path, "train")
        self.scene_paths = [
                os.path.join(self.source_path, p)
                for p in os.listdir(self.source_path)
                if os.path.isdir(os.path.join(self.source_path, p))
            ]
            
        for i, scene_path in enumerate(self.scene_paths):
            
            self.scene_paths[i] = os.path.join(self.source_path, scene_path)

        self.all_scenes = []

        print("Loading Training Data")
        too_large = []
        self.train_camera_items = []
        for i, scene_path in enumerate(tqdm(self.scene_paths, desc="加载训练数据")):
            tqdm.write(f"Loading Training {scene_path}, No.{i}")
            scnens_parts = []
            scnens_parts_paths = os.listdir(scene_path)
            
            for j, part_path in enumerate(scnens_parts_paths):
                scnens_parts_paths[j] = os.path.join(scene_path, part_path)
            visible_caled = True

            for j, part_path in enumerate(scnens_parts_paths):
                if not os.path.isdir(part_path):
                    continue

                image_path = None

                # if os.path.exists(os.path.join(scene_path, "seg_images")):
                #     image_path = os.path.join(scene_path, "seg_images")
                # else: 
                #     image_path = os.path.join(scene_path, "images")
                image_path = os.path.join(part_path, "images")


                train_cameras_list = []
                with open(os.path.join(part_path, "output", "cameras.json"), 'r') as file:
                    data = json.load(file)
                    for camera_entry in data:
                        camera = JSON_to_camera(camera_entry, image_path)
                        train_cameras_list.append(camera)
                
                train_cameras = cameraList_from_camInfos(train_cameras_list, 2, self.cfg, is_path=True)
                #cond_cameras = cameraList_from_camInfos(cond_cameras_list, 1.0, self.cfg)

                # gaussian_ply = load_gaussian_ply(os.path.join(part_path, "output", "point_cloud", "iteration_10000", "point_cloud.ply"))


                # point_cloud = gaussian_ply.xyz
                gaussian_ply_path = os.path.join(part_path, "output", "point_cloud", "transformed", "point_cloud.ply")
                score_path = os.path.join(part_path, "output", "point_cloud", "transformed", "score_data.npz")
                mask_path = os.path.join(part_path, "output", "point_cloud", "transformed", "mask_data.npz")
                if not os.path.exists(score_path) or not os.path.exists(mask_path):
                    visible_caled = False
                # else:
                #     score_data = np.load(score_path)
                #     mask_data = np.load(mask_path)
                #     try:
                #         score_cat = torch.tensor(score_data['score_data'], device=self.cfg.data_device).float()
                #         a = score_cat.size(1)
                #         mask = torch.tensor(mask_data['mask_data'], device=self.cfg.data_device).bool()
                #         mask = torch.tensor(mask_data['mask_data'], device=self.cfg.data_device)
                #         maskk = mask==1
                #         if (mask[maskk].size(0) > 500000):
                #             too_large.append(scene_path)
                #         pdb.set_trace()
                #     except:
                #         pdb.set_trace()
                # mask, dist = remove_outliers_knn(point_cloud.unsqueeze(0))
                # mask = mask.cpu()
                # opt_point_cloud = point_cloud.unsqueeze(0)[mask].to(self.cfg.data_device)
                ###
                opt_point_cloud = gaussian_ply_path
                # mask = torch.full([point_cloud.size(0)], True, dtype=torch.bool)
                ###
                focalY_length = train_cameras[0].image_height * 0.5 / np.tan(0.5 *train_cameras[0].FoVy)
                focalX_length = train_cameras[0].image_width * 0.5 / np.tan(0.5 *train_cameras[0].FoVx)
                
                intrinsic = torch.eye(3, dtype=torch.float32)
                intrinsic[0, 0] = focalX_length
                intrinsic[1, 1] = focalY_length
                intrinsic[0, 2] = train_cameras[0].image_width * 0.5
                intrinsic[1, 2] = train_cameras[0].image_height * 0.5


                c2w_cond = []
                depth_cond = []
                rgb_cond = []
                proj_cond = []
                cam_center_cond = []
                intrinsic_cond = []
                opacity_mask = []

                for cam in train_cameras:
                    c2w_cond.append(cam.world_view_transform.inverse().transpose(0, 1))
                    depth_cond.append(cam.fixed_depth_path)
                    rgb_cond.append(cam.image_path)
                    proj_cond.append(cam.full_proj_transform)
                    cam_center_cond.append(cam.camera_center)
                    intrinsic_cond.append(intrinsic)
                    # if cam.original_image.size(0) == 4:
                    #     opacity_mask.append(cam.original_image[3,:,:])

                self.sample_p = int(len(train_cameras)/9)
                # if len(opacity_mask) != 0:
                #     opacity_mask = torch.stack(opacity_mask, dim=0)
                
                c2w_cond = torch.stack(c2w_cond, dim=0)
                # depth_cond = torch.stack(depth_cond, dim=0)
                # rgb_cond = torch.stack(rgb_cond, dim=0)
                proj_cond = torch.stack(proj_cond, dim=0)
                cam_center_cond = torch.stack(cam_center_cond, dim=0)
                intrinsic_cond = torch.stack(intrinsic_cond, dim=0)
                
                scene = {
                    "scene_path": part_path,
                    "image_path": image_path,
                    "train_cameras": train_cameras,
                    "scene_name": part_path.split("/")[-1],
                    "gaussian_ply_path": gaussian_ply_path,
                    "score_path": os.path.join(gaussian_ply_path[:-15], "score_data.npz"),
                    "mask_path": os.path.join(gaussian_ply_path[:-15], "mask_data.npz"), 
                    "opt_point_cloud": opt_point_cloud,
                    "intrinsic": intrinsic,
                    "c2w_cond": c2w_cond,
                    "depth_cond": depth_cond,
                    "rgb_cond": rgb_cond,
                    "proj_cond": proj_cond,
                    "opacity_mask": opacity_mask,
                    "cam_center_cond": cam_center_cond,
                    "intrinsic_cond": intrinsic_cond,
                    "sample_p": self.sample_p,
                    }
                if istest:

                    # first_camera = train_cameras[0]
                    # last_camera = train_cameras[-1]

                    # train_cameras = [first_camera, last_camera]
                    train_cameras = train_cameras[0::10]
                for camera in train_cameras:
                    self.train_camera_items.append({"Camera": camera, "Scene_id": i, "Part_id": j})   

                scnens_parts.append(scene)

            if not visible_caled:
                self.visible_score_filter(scene_part=scnens_parts)

            self.all_scenes.append(scnens_parts)
        # 排除too_large中重复的元素
        too_large = list(set(too_large))
        
        # 将过大的场景信息存储到文件
        if len(too_large) > 0:
            output_file = os.path.join(self.source_path, "too_large_scenes.txt")
            with open(output_file, 'w') as f:
                f.write(f"过大的场景数量: {len(too_large)}\n")
                f.write("过大的场景路径:\n")
                for path in too_large:
                    f.write(f"  {path}\n")
            print(f"过大的场景信息已保存到: {output_file}")

        self.sample_offset = 0



    def offset_update(self):
        self.sample_offset += 1
        if self.sample_offset >= self.sample_p:
            self.sample_offset = 0

    def __len__(self):
        return len(self.train_camera_items) 

    def image_feat_set(self, image_feat):
        ind = 0
        for scenes_parts in self.all_scenes:
            for scene in scenes_parts:
                len = scene["rgb_cond"].size(0)
                scene["image_feat"] = torch.tensor(image_feat[ind:ind + len])
                ind += len

    def __getitem__(self, index):
        # 判断是训练相机还是测试相机
        camera_item = self.train_camera_items[index]
        rgb = camera_item["Camera"].read_pic()
        depth = camera_item["Camera"].read_depth()

        scene_parts = self.all_scenes[camera_item["Scene_id"]]
        # plys, depths, rgbs = self.visible_score_filter(scene_part=scene_parts)

        # 提取场景相关的字段
        image_feat_list = []

        c2w_cond_list = []
        rgb_cond_list = []
        depth_cond_path_list = []
        rgb_cond_path_list = []
        proj_cond_list = []
        cam_center_cond_list = []
        intrinsic_cond_list = []
        depth_cond_list = []
        xyz_list = []
        rots_list = []
        scales_list = []
        opacities_list = []
        features_dc_list = []
        features_extra_list = []
        opt_point_cloud_list = []
        opt_mask_list = []
        intrinsic_list = []
        ply_score_list = []
        avg_ply_score_list = []
        camera_cond_list = []

        try:
            for scene in scene_parts:
                image_feat_list.append(scene["image_feat"][self.sample_offset::scene["sample_p"]])
        except:
            pass
        
        part_num = 0
        for i, scene in enumerate(scene_parts):
            cam_num = scene_parts[i]["c2w_cond"].size(0)
            ply = load_gaussian_ply(scene_parts[i]["gaussian_ply_path"])
        
            score_data = np.load(scene_parts[i]["score_path"])
            score_cat = torch.tensor(score_data['score_data'], device=self.cfg.data_device).float()
            mask_data = np.load(scene_parts[i]["mask_path"])
            mask = torch.tensor(mask_data['mask_data'], device=self.cfg.data_device).bool()
            opt_point_cloud = score_cat[:, :3]
            ply_score = score_cat[:, 3:3+cam_num]
            avg_ply_score = score_cat[:, 3+cam_num:]
            opt_mask = mask

            c2w_cond_list.append(scene["c2w_cond"][self.sample_offset::scene["sample_p"]])
            rgb_cond_path_list.append(scene["rgb_cond"][self.sample_offset::scene["sample_p"]])
            proj_cond_list.append(scene["proj_cond"][self.sample_offset::scene["sample_p"]])
            cam_center_cond_list.append(scene["cam_center_cond"][self.sample_offset::scene["sample_p"]])
            intrinsic_cond_list.append(scene["intrinsic_cond"][self.sample_offset::scene["sample_p"]])
            depth_cond_path_list.append(scene["depth_cond"][self.sample_offset::scene["sample_p"]])
            camera_cond_list.append(scene["train_cameras"][self.sample_offset::scene["sample_p"]])

            xyz_list.append(ply.xyz)
            rots_list.append(ply.rots)
            scales_list.append(ply.scales)
            opacities_list.append(ply.opacities)
            features_dc_list.append(ply.features_dc)
            features_extra_list.append(ply.features_extra)
            opt_point_cloud_list.append(opt_point_cloud)
            opt_mask_list.append(opt_mask)
            intrinsic_list.append(scene["intrinsic"])
            avg_ply_score_list.append(avg_ply_score)
            ply_score_list.append(ply_score[:, self.sample_offset::scene["sample_p"]])
            part_num += 1
        
        for i in range(0, len(camera_cond_list)):   
            rgb_cond_list_single = []
            depth_cond_list_single = []

            for j in range(0, len(camera_cond_list[i])):
                rgb_cond = camera_cond_list[i][j].read_pic()
                depth_cond = camera_cond_list[i][j].read_depth()
                rgb_cond_list_single.append(rgb_cond.unsqueeze(0))
                depth_cond_list_single.append(depth_cond)

            rgb_cond_list_single = torch.cat(rgb_cond_list_single, dim=0)
            depth_cond_list_single = torch.cat(depth_cond_list_single, dim=0)
            rgb_cond_list.append(rgb_cond_list_single)
            depth_cond_list.append(depth_cond_list_single)

        return {
                "rgb": rgb[:3,:,:].unsqueeze(0),
                "mask" : rgb[3:4,:,:].unsqueeze(0),
                "c2w" : camera_item["Camera"].world_view_transform.inverse().transpose(0, 1).unsqueeze(0),#
                "proj" : camera_item["Camera"].full_proj_transform.unsqueeze(0),#
                "cam_center": camera_item["Camera"].camera_center.unsqueeze(0),#
                "depth": depth.unsqueeze(0),

                "c2w_cond": c2w_cond_list, #
                "rgb_cond": rgb_cond_list,
                "proj_cond": proj_cond_list, #
                "cam_center_cond": cam_center_cond_list,#
                "intrinsic_cond": intrinsic_cond_list,#
                "depth_cond": depth_cond_list,
                "image_feat_cond": image_feat_list,

                "xyz": xyz_list,
                "rots": rots_list,
                "scales": scales_list,
                "opacities": opacities_list,
                "features_dc": features_dc_list,
                "features_extra": features_extra_list,
                "opt_point_cloud": opt_point_cloud_list,
                "opt_mask" : opt_mask_list,
                "intrinsic": intrinsic_list, #
                "ply_score": ply_score_list, 
                "avg_ply_score": avg_ply_score_list,
                "scene_path": scene_parts[0]['scene_path'],
                "part_num": part_num,
                "is_test": index >= len(self.train_camera_items)  # 添加标志表示是否为测试数据
            }

    def collate(self, batch):
    # 确保 batch 是一个列表
        if not isinstance(batch, list):
            raise TypeError(f"Expected batch to be a list, but got {type(batch)}")
        
        # 初始化返回值
        result = {}

        # 遍历第一个样本的所有键
        for key in batch[0]:
            if isinstance(batch[0][key], list):  # 如果字段是列表
                # 尝试对列表中的每个元素进行堆叠
                try:
                    result[key] = [
                        torch.stack([sample[key][i] for sample in batch], dim=0)
                        for i in range(len(batch[0][key]))
                    ]
                except RuntimeError:  # 如果无法堆叠，保持为列表
                    result[key] = [sample[key] for sample in batch]
            elif isinstance(batch[0][key], torch.Tensor):  # 如果字段是张量
                try:
                    result[key] = torch.stack([sample[key] for sample in batch], dim=0)
                except RuntimeError:  # 如果无法堆叠，保持为列表
                    result[key] = [sample[key] for sample in batch]
            else:  # 其他类型，保持为列表
                result[key] = [sample[key] for sample in batch]

        # 添加额外信息
        if "intrinsic" in result and isinstance(result["intrinsic"], list) and len(result["intrinsic"]) > 0:
            result.update({
                "height": result["intrinsic"][0][0][1, 2] * 2,
                "width": result["intrinsic"][0][0][0, 2] * 2
            })

        return result

    def visible_score_filter(self, scene_part):        
        N = len(scene_part)
        exist = True

        for i in range(0, N):
            if not os.path.exists(scene_part[i]["score_path"]):
                exist = False

            if not os.path.exists(scene_part[i]["mask_path"]):
                exist = False

        if exist:
            return
        
        pts = []
        c2ws = []
        intris = []
        pt_num=[]
        Nv_num=[]
        opacity_masks = []
        pt_num.append(0)
        recent_pt_num = 0
        Nv_num.append(0)
        recent_Nv_num = 0
        real_depths_list = []
        images_list = []
        plys = []

        for i in range(0, N):
            real_depths = []
            images = []
            for j in range(0, len(scene_part[i]["train_cameras"])):
                real_depth=scene_part[i]["train_cameras"][j].read_depth()
                real_depths.append(real_depth)
                image = scene_part[i]["train_cameras"][j].read_pic()
                images.append(image.unsqueeze(0))
            real_depths = torch.cat(real_depths, dim=0)
            images = torch.cat(images, dim=0)
            real_depths_list.append(real_depths)
            images_list.append(images)

            c2w = scene_part[i]["c2w_cond"]
            intri = scene_part[i]["intrinsic_cond"]
            opacity_mask = scene_part[i]["opacity_mask"]
            ply = load_gaussian_ply(scene_part[i]["gaussian_ply_path"])
            plys.append(ply)
            pt = ply.xyz
            ones = torch.ones((pt.size(0), 1), device=pt.device)
            pt = torch.cat([pt, ones], dim=-1)
            pts.append(pt)
            c2ws.append(c2w)
            intris.append(intri)
            opacity_masks.append(opacity_mask)
            recent_pt_num = recent_pt_num + pt.size(0)
            pt_num.append(recent_pt_num)
            recent_Nv_num = recent_Nv_num + c2w.size(0)
            Nv_num.append(recent_Nv_num)

        pts = torch.cat(pts, dim=0)
        c2ws = torch.cat(c2ws, dim=0)
        intris = torch.cat(intris, dim=0)
        real_depths = torch.cat(real_depths_list, dim=0)  # 转换为float32

        opacity_used = True

        if not isinstance(opacity_masks[0], list):
            opacity_masks = torch.cat(opacity_masks, dim=0)
        else: opacity_used = False
        #################################################################################################################
        #高斯模糊避免硬边缘 2025/6/16
        # 添加高斯模糊处理
        # kernel_size = 5
        # sigma = 1.0
        # # 创建高斯核
        # x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=opacity_masks.device)
        # y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=opacity_masks.device)
        # x, y = torch.meshgrid(x, y, indexing='ij')
        # gaussian_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        # gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # # 将kernel扩展为4D张量 [out_channels, in_channels, height, width]
        # gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # # 对每个通道分别进行卷积
        # opacity_masks = opacity_masks.unsqueeze(1)  # 添加通道维度 [B, 1, H, W]
        # opacity_masks = F.conv2d(opacity_masks, gaussian_kernel, padding=kernel_size//2)
        # opacity_masks = opacity_masks.squeeze(1)  # 移除通道维度 [B, H, W]
        ###################################################################################################################

        pts_depth, uv_coords, mask = self.proj_cal(pts, c2ws, intris, real_depth.size(1), real_depth.size(2))

        pts_real_depth = self.sample_depths_from_views(real_depths, uv_coords, real_depth.size(1), real_depth.size(2), 'zeros')
        if opacity_used:
            pts_pixel_opacity = self.sample_depths_from_views(opacity_masks, uv_coords, opacity_mask.size(1), opacity_mask.size(2), 'ones')
            
            # 创建一个掩码，表示每行只要有一个元素为0就为True
            front_mask = torch.any(pts_pixel_opacity == 0, dim=1)  # 返回一个大小为N的布尔张量
        else:
            front_mask = torch.full((pts_depth.size(0),), False, dtype=torch.bool, device=pts_depth.device)
            
        mask = mask & (pts_real_depth > 0)
        score = 1.0 - torch.abs(pts_real_depth - pts_depth) / pts_depth
        score[~mask] = 0
        
        avg_score = []
        ply_score_list = []
        avg_ply_score_list = []
        for i in range(0, N):
            scene_score = score[:, Nv_num[i]:Nv_num[i+1]]
            scene_score = torch.mean(scene_score, dim=-1)
            avg_score.append(scene_score.unsqueeze(-1))
        for i in range(0, N):
            ply_score = score[pt_num[i]:pt_num[i+1], Nv_num[i]:Nv_num[i+1]]
            
            # scene_part[i]["ply_score"] = ply_score.cpu()
            ply_score_list.append(ply_score.cpu())

            # scene_part[i]["front_mask"] = front_mask[pt_num[i]:pt_num[i+1]].cpu()

            # scene_part[i]["avg_ply_score"] = torch.cat(avg_score, dim=1)[pt_num[i]:pt_num[i+1], :].cpu()
            avg_ply_score_list.append(torch.cat(avg_score, dim=1)[pt_num[i]:pt_num[i+1], :].cpu())

        avg_score = torch.cat(avg_score, dim = -1)
        avg_score = (avg_score - avg_score.min())/(avg_score.max() - avg_score.min())
        # pdb.set_trace()
        for i in range(0, N):
            scene_score = avg_score[pt_num[i]:pt_num[i+1], i]
            best_score_values, _ = torch.max(avg_score[pt_num[i]:pt_num[i+1], :], dim=-1)

            # for j in range(0, 3):
            #     scene_part[i]["gaussian_ply"].features_dc[:, j, :] = 0

            # scene_part[i]["gaussian_ply"].features_dc[:, i, :] = scene_score.unsqueeze(-1)

            del_pts_mask = (best_score_values - scene_score) > 0.0
            del_pts_mask = del_pts_mask | front_mask[pt_num[i]:pt_num[i+1]]

            del_pts_mask = del_pts_mask.cpu()

            opt_point_cloud_data = plys[i].xyz[~del_pts_mask]
            ply_score_data = ply_score_list[i][~del_pts_mask]
            avg_ply_score_data = avg_ply_score_list[i][~del_pts_mask]
            # temp_mask = torch.full([plys[i].xyz.size(0)], False, dtype=torch.bool)
            # temp_mask[scene_part[i]["opt_mask"]] = ~del_pts_mask
            opt_mask_data = ~del_pts_mask
            score_data = torch.cat([opt_point_cloud_data, ply_score_data, avg_ply_score_data], dim = -1).numpy()
            mask_data = opt_mask_data.int().numpy()
            np.savez_compressed(scene_part[i]["score_path"], score_data=score_data)
            np.savez_compressed(scene_part[i]["mask_path"], mask_data=mask_data)


        return plys, real_depths_list, images_list

    def proj_cal(self, pts, c2ws, intris, H, W):

        cam_points_homogeneous = torch.einsum(
            'nl,mkl->nmk',
            pts, # Shape: [N, 4]
            c2ws.inverse()                     # Shape: [M, 4, 4]
        )

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
            intris           # Shape: [M, 3, 3]
        )

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
        
        return depths, uv_coords, mask
    
    def sample_depths_from_views(self, depth_maps, uv_coords, H, W, padding_mode='zeros'):
        """
        从一系列深度图中采样每个点的深度。

        Args:
            depth_maps (torch.Tensor): 深度图张量，形状为 [V, H, W]
            uv_coords (torch.Tensor): uv坐标张量，形状为 [Np, V, 2]
            H (int): 深度图的高度
            W (int): 深度图的宽度
            padding_mode (str): 填充模式，可选值为 'zeros', 'border', 'reflection'

        Returns:
            torch.Tensor: 每个点在每个视图的采样深度，形状为 [Np, V]
        """
        V = depth_maps.shape[0] # 视图数量
        Np = uv_coords.shape[0] # 点的数量

        # 1. 调整深度图的形状为 (N, C, H_in, W_in)
        # N=V, C=1
        depth_maps_input = depth_maps.unsqueeze(1) # 形状变为 [V, 1, H, W]

        # 2. 归一化 uv_coords 到 [-1, 1] 范围
        # 注意：这里假设uv_coords是像素坐标 [0, W-1] 和 [0, H-1]
        # 需要将其转换为grid_sample所需的[-1, 1]范围
        
        # 将 uv_coords 从 [Np, V, 2] 转换为 [V, Np, 1, 2] 以便批处理
        # 并且交换 u,v 顺序，因为 grid_sample 期望 (x, y) 即 (W, H)
        # 而我们的 uv_coords 是 (u, v) 即 (W, H)
        
        # 归一化 u 坐标
        u_normalized = 2 * (uv_coords[..., 0] / (W - 1)) - 1
        # 归一化 v 坐标
        v_normalized = 2 * (uv_coords[..., 1] / (H - 1)) - 1

        # 重新堆叠为 (x_norm, y_norm) 顺序，即 (u_norm, v_norm)
        # 形状为 [Np, V, 2]
        normalized_uv_coords = torch.stack([u_normalized, v_normalized], dim=-1)

        # 调整 normalized_uv_coords 的形状以匹配 grid_sample 的 grid 参数
        # grid_sample 期望 grid 的形状为 (N, H_out, W_out, 2)
        # N 是批次大小，这里是 V (视图数量)
        # H_out, W_out 是输出的采样网格大小，这里我们把每个点当做一个 H_out=Np, W_out=1 的网格
        
        # 将 [Np, V, 2] 转换为 [V, Np, 1, 2]
        # 首先交换 Np 和 V 的维度，变为 [V, Np, 2]
        grid = normalized_uv_coords.permute(1, 0, 2)
        # 然后添加一个维度，变为 [V, Np, 1, 2]
        grid = grid.unsqueeze(2) # 形状变为 [V, Np, 1, 2]

        # 3. 使用 grid_sample 进行采样
        # align_corners=True 对于图像像素的精确采样通常是推荐的。
        # interpolation_mode='bilinear' 是默认值，表示双线性插值。
        # padding_mode='zeros' 是默认值，对于超出范围的采样点返回0。
        # 你可能需要根据实际情况调整 padding_mode，例如 'border' 或 'reflection'
        
        # output_depths 形状为 [V, 1, Np, 1]

        if padding_mode == "zeros":
            sampled_depths = F.grid_sample(
                depth_maps_input, # [V, 1, H, W]
                grid,             # [V, Np, 1, 2]
                mode='bilinear',
                padding_mode=padding_mode, # 使用ones填充模式
                align_corners=True
            )
        elif padding_mode == "ones":
            ones = torch.ones_like(depth_maps_input)
            ones_sample= F.grid_sample(
                ones, # [V, 1, H, W]
                grid,             # [V, Np, 1, 2]
                mode='bilinear',
                padding_mode='zeros', # 使用ones填充模式
                align_corners=True
            )
            mask = (ones_sample == 0)
            sampled_depths = F.grid_sample(
                depth_maps_input, # [V, 1, H, W]
                grid,             # [V, Np, 1, 2]
                mode='bilinear',
                padding_mode='border', # 使用ones填充模式
                align_corners=True
            )
            sampled_depths[mask] = 1

        # 4. 调整输出形状
        # sampled_depths 当前是 [V, 1, Np, 1]
        # 我们想要 [Np, V]
        sampled_depths = sampled_depths.squeeze() # 变为 [V, Np]
        sampled_depths = sampled_depths.permute(1, 0) # 变为 [Np, V]

        return sampled_depths