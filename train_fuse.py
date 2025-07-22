import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import tgs
from tgs.models.image_feature import UNet, add_positional_encoding, flatten_spatial, create_plucker
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection, points_projection_with_score_chunked, find_nearest_neighbors_with_distance_and_indices_pytorch3d, points_projection_with_score_optimized
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *
from tgs.utils.loss_def import l1_loss, ssim
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pdb
from tgs.models.triplane.triplane_transformerV4 import ImageTriplaneGenerator
from tgs.models.triplane.triplane_splatting import PointTriplaneGenerator
from tgs.models.Fuse_Attention_origin import FuseNetwork
from tgs.models.pointclouds.samplefarthest import grid_sample, grid_sample_centers, remove_outliers_knn
from transformers import AutoModel
from timm.data.transforms_factory import create_transform
import torchvision.transforms.functional as TF # 用于可能的张量变换
from tgs.models.pointclouds.pointnet2 import Pointnet2
from tgs.models.networks import MLP
import torch
from pointMamba.models import build_model_from_cfg
from pointMamba.utils.config import cfg_from_yaml_file
from lpipsPyTorch import lpips
import gc
from torch.utils.tensorboard import SummaryWriter
def psnr(img1, img2):
    # 确保输入张量是连续的
    img1 = img1.contiguous()
    img2 = img2.contiguous()
    # 使用reshape替代view
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

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

def save_checkpoint(model, save_dir, epoch=None, step=None):
    """
    Save the model's state_dict to a file.
    
    Args:
        model (torch.nn.Module): The model to save.
        save_dir (str): Directory where the checkpoint will be saved.
        epoch (int, optional): Current epoch number.
        step (int, optional): Current training step.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    checkpoint_name = f"model_checkpoint"
    if epoch is not None:
        checkpoint_name += f"_epoch_{epoch}"
    if step is not None:
        checkpoint_name += f"_step_{step}"
    checkpoint_name += ".pth"

    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'cfg': model.cfg  # Optionally save the configuration for reproducibility
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)



        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

        fuser_cls: str = ""
        fuser: dict = field(default_factory=dict)
    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )

        model_state_dict = self.state_dict()

        # 过滤掉不以 'image_tokenizer' 开头的参数
        # filtered_state_dict = {k: v for k, v in state_dict.items() if (k.startswith('image_tokenizer') or k.startswith('camera_embedder')) and k in model_state_dict and v.size() == model_state_dict[k].size()}

        self.load_state_dict(model_state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        # parameters loaded:
            # image_tokenizer
            # camera_embedder
            # tokenizer
            # backbone
            # post_processor
            # renderer
            # pointcloud_generator
            # pointcloud_encoder      


        # self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
        #     self.cfg.image_tokenizer
        # )


        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        # self.image_feature = ImageFeature(self.cfg.image_feature)

        # self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        # self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)


        # self.point_encoder = tgs.find(self.cfg.pointcloud_encoder_cls)(self.cfg.pointcloud_encoder)

        # self.pose_pos_embed = torch.tensor([[0, 0]], device = 'cuda')
        # load checkpoint
        # if self.cfg.weights is not None:
        #     self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
        self.FuseNet = tgs.find(self.cfg.fuser_cls)(self.cfg.fuser)

        # self.UnetFeature = UNet(9, 128)
        self.triplaneGenerator = ImageTriplaneGenerator(grid_size=128, feature_channels=228)
        self.pointTriplaneGenerator = PointTriplaneGenerator(128, in_channels=384, out_channels=228)
        
        # 冻结除了FuseNet之外的所有网络参数
        # for name, param in self.named_parameters():
        #     if not name.startswith('FuseNet'):
        #         param.requires_grad = False

        self.mambavision_backbone = AutoModel.from_pretrained(
            "nvidia/MambaVision-L2-512-21K",
            trust_remote_code=True
        )

        for param in self.mambavision_backbone.parameters():
            param.requires_grad = False

        self.mambavision_config = self.mambavision_backbone.config
        self.mambavision_input_resolution = (3, 512, 512)
        
        self.mambavision_transform = create_transform(
            input_size=self.mambavision_input_resolution[1:], # 只需 H, W
            is_training=False, # 通常在推理或特征提取时设为 False
            mean=self.mambavision_config.mean,
            std=self.mambavision_config.std,
            crop_mode=self.mambavision_config.crop_mode,
            crop_pct=self.mambavision_config.crop_pct
        )

        self.pointNet = Pointnet2()

        self.camera_norm = torch.nn.LayerNorm(32)

        self.GS_embedder = MLP(dim_in=56, dim_out=64, n_neurons=128, n_hidden_layers=2)

        self.triplane_gate = MLP(dim_in=228 * 2, dim_out=1, n_neurons=228, n_hidden_layers=2, output_activation="sigmoid")

        config = cfg_from_yaml_file('pointMamba/cfgs/test.yaml')

        model_cfg = config.model

        self.PM_model = build_model_from_cfg(model_cfg)



    def forward(self, batch: Dict[str, Any], shs_mask:Optional[int] = 0) -> Dict[str, Any]:
        batch_size = batch['xyz'][0].size(0)
        scene_codes = []
        local_point_features = []
        pointclouds = []
        project_feats = []
        GS_feats = []
        padding = 0.05  # 扩展5%的范围
        fuse_point = torch.cat(batch["opt_point_cloud"], dim = 1)
        # fuse_mask = torch.cat(batch["opt_mask"], dim = -1)


        # fuse_mask = batch["opt_mask"][0]
        # fuse_point= batch["opt_point_cloud"][0]
        scene_bounds = (
            fuse_point[:, :, 0].min() - padding * (fuse_point[:, :, 0].max() - fuse_point[:, :, 0].min()),
            fuse_point[:, :, 0].max() + padding * (fuse_point[:, :, 0].max() - fuse_point[:, :, 0].min()),
            fuse_point[:, :, 1].min() - padding * (fuse_point[:, :, 1].max() - fuse_point[:, :, 1].min()),
            fuse_point[:, :, 1].max() + padding * (fuse_point[:, :, 1].max() - fuse_point[:, :, 1].min()),
            fuse_point[:, :, 2].min() - padding * (fuse_point[:, :, 2].max() - fuse_point[:, :, 2].min()),
            fuse_point[:, :, 2].max() + padding * (fuse_point[:, :, 2].max() - fuse_point[:, :, 2].min()),
        )

        for id in range(batch['part_num'][0]):
            scene_code, local_point_feature, pointcloud, project_feat, GS_feat = self.forward_single_part(id, batch, scene_bounds, shs_mask)
            scene_codes.append(scene_code)
            local_point_features.append(local_point_feature)
            pointclouds.append(pointcloud)
            project_feats.append(project_feat)          # 无梯度
            GS_feats.append(GS_feat[batch["opt_mask"][id]])

        for i in range(len(scene_codes)):
            scene_codes[i] = scene_codes[i].unsqueeze(1)

        # 对输入场景编码进行归一化
        scene_codes = torch.cat(scene_codes, dim=1) #B Nt Np C Hp Wp

        omega, fused_scene_code = self.FuseNet(scene_codes) ## 有梯度
 
        # print(f"{omegas[0].item()}, {omegas[1].item()}")
        # fused_scene_code = scene_codes[0]
        #pdb.set_trace()
        # GS_feats = torch.cat(GS_feats, dim=1).unsqueeze(0)[fuse_mask].unsqueeze(0)

        # GS_feats = GS_feats[0]
        # pointclouds = torch.cat(pointclouds, dim=1)
        # local_point_features = local_point_features[0]
        # proj_feat = torch.cat(project_feats, dim = 1)

        recent_ply = batch["opt_point_cloud"][0]
        recent_ply_count = torch.ones([recent_ply.size(1)], device = 'cuda')
        recent_ply_feat = project_feats[0][0]
        recent_GS = GS_feats[0]
        recent_score = batch["avg_ply_score"][0][0].squeeze(-1)
        threshold = torch.exp(torch.cat(batch["scales"], dim = 1)).mean()

        for i in range(1, batch['part_num'][0]):
            another_ply = batch["opt_point_cloud"][i]
            another_feats = project_feats[i]
            another_GS = GS_feats[i]
            another_mask = None
            another_score = batch["avg_ply_score"][i][0].squeeze(-1)

            try:
                for j in range(0, 1):
                    nearest_neighbor_indices, nearest_neighbor_distances = find_nearest_neighbors_with_distance_and_indices_pytorch3d(recent_ply, another_ply)

                    mask = nearest_neighbor_distances < threshold * 2
                    
                    score = another_score[nearest_neighbor_indices]

                    score_mask = torch.abs(recent_score[:, i] - score[:, i]) < 0.05

                    mask = mask & score_mask

                    masked_indices = nearest_neighbor_indices[mask]
                    feat = another_feats[0][masked_indices]
                    another_mask = torch.full([another_ply.size(1)], False, dtype=torch.bool)
                    another_mask[masked_indices] = True
                    recent_ply_feat[mask] += feat
                    recent_ply_count[mask] += 1

                    another_ply = another_ply[0][~another_mask].unsqueeze(0)
                    another_feats = another_feats[0][~another_mask].unsqueeze(0)
                    another_GS = another_GS[~another_mask]
                    another_score = another_score[~another_mask]
            except:
                pdb.set_trace()

            recent_ply = torch.cat([recent_ply, another_ply], dim = 1)
            recent_ply_feat = torch.cat([recent_ply_feat,  another_feats[0]], dim = 0)
            ones = torch.ones([another_feats[0].size(0)], device='cuda')
            recent_ply_count = torch.cat([recent_ply_count, ones], dim = 0)
            recent_GS = torch.cat([recent_GS, another_GS], dim = 0)

        recent_ply_feat = recent_ply_feat / recent_ply_count.unsqueeze(-1)
        GS_embedding = self.GS_embedder(recent_GS)
        recent_ply_feat = torch.cat([recent_ply_feat, GS_embedding], dim=-1)     #有梯度

        # recent_ply = torch.cat(batch['xyz'], dim=1)
        # recent_ply_feat = torch.zeros([recent_ply.size(1), 292], dtype=torch.float32, device=recent_ply.device)
        # opacities = torch.cat(batch['opacities'], dim=1)
        # rots = torch.cat(batch['rots'], dim=1)
        # scales = torch.cat(batch['scales'], dim=1)
        # features_dc = torch.cat(batch['features_dc'], dim=1)
        # features_extra = torch.cat(batch['features_extra'], dim=1)
        # rgb_feats = torch.cat([features_dc,features_extra],dim=-1).transpose(2,3).flatten(2)
        # recent_GS = torch.cat([opacities, rots, scales, rgb_feats], dim=-1).squeeze(0)

        rend_out = self.renderer(fused_scene_code,
                                query_points=recent_ply,
                                additional_features = recent_ply_feat.unsqueeze(0),
                                GS_feats=recent_GS.unsqueeze(0),
                                shs_mask = shs_mask,
                                scene_bounds = scene_bounds,
                                **batch)
        # 3GB

        return rend_out
    
    def forward_single_part(self, id, batch: Dict[str, Any], scene_bounds, shs_mask:Optional[int] = 0) -> Tensor:
        rgb_feats = torch.cat([batch["features_dc"][id], batch["features_extra"][id]], dim=-1).transpose(2, 3).flatten(2)
        GS_feats = torch.cat([batch["opacities"][id], batch["rots"][id], batch["scales"][id], rgb_feats], dim=-1)

        pointclouds = batch["xyz"][id]
        batch_size, n_input_views = batch["rgb_cond"][id].shape[:2]

        # Camera modulation
        camera_extri = batch["c2w_cond"][id]
        camera_intri = batch["intrinsic_cond"][id]
        c2s_cond = batch["c2w_cond"][id].view(*batch["c2w_cond"][id].shape[:-2], -1)
        intri_cond = batch["intrinsic_cond"][id].view(*batch["intrinsic_cond"][id].shape[:-2], -1)

        camera_feats = torch.cat([intri_cond, c2s_cond], dim=-1)

        if len(batch["image_feat_cond"]) != 0:
            image_features = batch["image_feat_cond"][id]
        else:
            
            rgb_cond_flat = rearrange(batch["rgb_cond"][id], 'b nv c h w -> (b nv) c h w')
            rgb_cond_flat =rgb_cond_flat[:, 0:3]
            resized_inputs = TF.resize(rgb_cond_flat, self.mambavision_input_resolution[1:], antialias=True)
            normalized_inputs = TF.normalize(resized_inputs, mean=self.mambavision_config.mean, std=self.mambavision_config.std)
            inputs_for_mamba = normalized_inputs.cuda() # 确保在 GPU 上
            self.mambavision_backbone.eval() # 或者确保在训练循环外设置
            # pdb.set_trace()
            # with torch.no_grad() if self.mambavision_backbone.parameters().__next__().requires_grad == False else torch.enable_grad():
            with torch.no_grad():
                #model 输出: _, features (包含 4 个 stage 特征的列表)
                _, intermediate_features = self.mambavision_backbone(inputs_for_mamba)
        
                # image_feature: Float[Tensor, "(B Nv) Cit H W"] = self.UnetFeature(input_images)['fused']
                
                # image_feature = rearrange(image_feature, "(B Nv) Cit H W -> B Nv Cit H W" , B = batch_size)
                # image_feature_withpos = add_positional_encoding(image_feature)

                # input_image_tokens = flatten_spatial(image_feature_withpos)

                #input_image_tokens = torch.cat([input_image_tokens, camera_feats], dim=-1)
                selected_features = intermediate_features[0] # Shape: [(B*Nv), C3, H3, W3]

                del intermediate_features
                image_features = rearrange(selected_features, "(B Nv) C H W -> B Nv C H W", B = batch_size)

                original_size = batch["rgb_cond"][id].shape[-2:]  # 获取原始图像的高度和宽度

                image_features = torch.nn.functional.interpolate(
                    image_features.flatten(0, 1),  # 将B和Nv维度合并
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                ).view(batch_size, n_input_views, -1, *original_size)  # 恢复B和Nv维度
            
            #get image features for projection

            torch.cuda.empty_cache()  # 清理缓存

        pluker = create_plucker(image_features, camera_extri, camera_intri)

        pluker_embedding = self.camera_embedder(pluker)

        pluker_embedding = self.camera_norm(pluker_embedding)
        # expanded_camera_feat = pluker_embedding.view(batch_size, n_input_views, -1, 1, 1).expand(-1, -1, -1, original_size[0], original_size[1])

        # image_features = image_features + expanded_camera_feat
        pluker_embedding = rearrange(pluker_embedding, "B Nv H W C -> B Nv C H W")

        image_features = torch.cat([image_features, pluker_embedding], dim=2)
        # #only support number of input view is one
        # c2w_cond = batch["c2w_cond"][id]
        # intrinsic_cond = batch["intrinsic_cond"][id]
        depth_cond = batch['depth_cond'][id]

        # # print(pointclouds.size())
        # print(c2w_cond.size())
        # print(intrinsic_cond.size())
        # # print(image_features.size())
        proj_feats = None

        proj_feats = points_projection_with_score_chunked(batch["opt_point_cloud"][id], camera_extri, camera_intri, image_features, batch["ply_score"][id])

        # del image_features  # 释放投影前的特征显存
        # torch.cuda.empty_cache()  # 再次清理缓存
        # # 5GB
        # print(pointclouds.size())
        # print(proj_feats.size())

        #point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats, GS_feats], dim=-1))
        # point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, GS_feats], dim=-1))
        # 7GB
        #print(point_cond_embeddings)
        # tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)

        # tokens = self.backbone(
        #     tokens,
        #     encoder_hidden_states=input_image_tokens,
        #     modulation_cond=None,
        # )

        # scene_codes = self.post_processor(self.tokenizer.detokenize(tokens)) #转换为三平面并上采样
        # rend_out = self.renderer(scene_codes,
        #                         query_points=pointclouds,
        #                         additional_features=torch.cat([proj_feats, GS_feats], dim=-1),
        #                         **batch)
        # padding = 0.05  # 扩展5%的范围
        # scene_bounds = (
        #     batch['xyz'].min() - padding * (batch['xyz'].max() - batch['xyz'].min()),
        #     batch['xyz'].max() + padding * (batch['xyz'].max() - batch['xyz'].min())
        # )

        opt_GS_feats = GS_feats[batch['opt_mask'][id]].unsqueeze(0)
        point_feats = torch.cat([batch['opt_point_cloud'][id], opt_GS_feats], dim=-1).requires_grad_(True)

        # start_timer("point_feat")
        # local_point_feature = None       
        central_point_feature, dense_point_feature, weight = self.PM_model(point_feats)

        # end_timer("point_feat")
        depth = batch["depth"]
        c2w = batch["c2w"]
        intrinsic = batch["intrinsic"]
        
        sample_grid = torch.exp(batch["scales"][id].mean())

###############################################################
        # padding = torch.zeros([1, 10, 225, 242, 324]).cuda()
        # image_features = torch.cat([batch['rgb_cond'][id], padding], dim=2)

        image_scene_codes, scene_bounds = self.triplaneGenerator(image_features, depth_cond, camera_extri, camera_intri, depth, c2w, intrinsic, sample_grid, scene_bounds)

###############################################################
        # local_point_feature = rearrange(local_point_feature, 'B C N -> B N C')

        point_scene_codes = self.pointTriplaneGenerator(central_point_feature, dense_point_feature, weight, scene_bounds)

        scene_codes = torch.cat([image_scene_codes, point_scene_codes], dim=2)

        gate_weight = self.triplane_gate(rearrange(scene_codes, 'B P C H W -> B (P H W) C'))
        
        gate_weight = rearrange(gate_weight, 'B (P H W) C -> B P C H W', H = 128, W=128, P=3)

        scene_codes = image_scene_codes * gate_weight + point_scene_codes * (1-gate_weight)
        # scene_codes = point_scene_codes
        # scene_codes = image_scene_codes
        # scene_codes = image_scene_codes
        # proj_feats = torch.cat([proj_feats, local_point_feature[batch['opt_mask'][id][0]].unsqueeze(0)], dim = 2)

        return scene_codes, None, pointclouds, proj_feats, GS_feats


    def image_feat_precal(self, batch: Dict[str, Any]):

        batch_size = batch["rgb_cond"][0].shape[0]

        rgb_cond_flat = rearrange(batch["rgb"], 'b nv c h w -> (b nv) c h w')
        resized_inputs = TF.resize(rgb_cond_flat, self.mambavision_input_resolution[1:], antialias=True)
        normalized_inputs = TF.normalize(resized_inputs, mean=self.mambavision_config.mean, std=self.mambavision_config.std)
        inputs_for_mamba = normalized_inputs.cuda() # 确保在 GPU 上
        self.mambavision_backbone.eval() # 或者确保在训练循环外设置
        # pdb.set_trace()
        with torch.no_grad() if self.mambavision_backbone.parameters().__next__().requires_grad == False else torch.enable_grad():
            #model 输出: _, features (包含 4 个 stage 特征的列表)
            _, intermediate_features = self.mambavision_backbone(inputs_for_mamba)
        
            # image_feature: Float[Tensor, "(B Nv) Cit H W"] = self.UnetFeature(input_images)['fused']
            
            # image_feature = rearrange(image_feature, "(B Nv) Cit H W -> B Nv Cit H W" , B = batch_size)
            # image_feature_withpos = add_positional_encoding(image_feature)

            # input_image_tokens = flatten_spatial(image_feature_withpos)

            #input_image_tokens = torch.cat([input_image_tokens, camera_feats], dim=-1)
            selected_features = intermediate_features[0] # Shape: [(B*Nv), C3, H3, W3]

            image_features = rearrange(selected_features, "(B Nv) C H W -> B Nv C H W", B = batch_size)

            original_size = batch["rgb"].shape[-2:]  # 获取原始图像的高度和宽度
            image_features = torch.nn.functional.interpolate(
                image_features.flatten(0, 1),  # 将B和Nv维度合并
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).view(batch_size, 1, -1, *original_size)  # 恢复B和Nv维度
        
        #get image features for projection

        torch.cuda.empty_cache()  # 清理缓存

        return image_features
        
if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.data_Fuse_Scan import CustomGaussianFuseDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    parser.add_argument("--exp_name", default='test', type=str, help="exp name, used to save argument and result")
    args, extras = parser.parse_known_args()
    # pdb.set_trace
    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)


    from huggingface_hub import hf_hub_download
    # model_path = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
    # model_path = "checkpoints/model_lvis_rel.ckpt"
    model_path = "exp/test_2025_7_17_full/checkpoint/model_checkpoint_epoch_5_step_0.pth"

    #pdb.set_trace()
    test_dataset = CustomGaussianFuseDataset(istest=True, cfg=cfg.data)
    train_dataset = CustomGaussianFuseDataset(istest=False, cfg=cfg.data)
    train_dataloader = DataLoader(train_dataset,
            batch_size=cfg.data.eval_batch_size, 
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=train_dataset.collate,
        )
    
    
    test_dataloader = DataLoader(test_dataset,
            batch_size=cfg.data.eval_batch_size, 
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=test_dataset.collate,
        )
    data_size = train_dataset.__len__()
    cfg.system.weights=model_path
    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)

    # checkpoint = torch.load(model_path, map_location="cpu")

    # filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()}
    # model.load_state_dict(filtered_state_dict, strict=False)

    model.train()
    print("load model ckpt done.")



    # precalDataloader = DataLoader(dataset,
    #                     batch_size=cfg.data.eval_batch_size, 
    #                     num_workers=cfg.data.num_workers,
    #                     shuffle=False,
    #                     collate_fn=dataset.collate,
    #                 )
    print('load Training data done.')
    criterion = torch.nn.MSELoss()  # 假设使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    save_dir = os.path.join("exp", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    pic_dir = os.path.join(save_dir, "render")
    check_dir = os.path.join(save_dir, "checkpoint")
    GS_dir = os.path.join(save_dir, "gs")
    os.makedirs(pic_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(GS_dir, exist_ok=True)



    num_epochs = 150
    print("train start")
    iteration = 0
    scaler = GradScaler()
    shs_mask = 3

    ## LOG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_md = os.path.join(save_dir, f'psnr_log_{timestamp}.md')  # Markdown格式
    with open(log_file_md, 'w') as f:
        f.write("# PSNR Training Log\n")
        f.write("| Epoch | PSNR  |\n")
        f.write("|-------|-------|\n")
    # image_feat =[]
    # for batch_idx, batch in enumerate(tqdm(precalDataloader, desc=f"feat precal")):
    #     model.eval()
    #     batch = todevice(batch)
    #     with torch.no_grad():
    #         outputs = model.image_feat_precal(batch)
    #         image_feat.append(outputs[0].cpu())

    # image_feat = np.concatenate(image_feat, axis=0)
    # dataset.image_feat_set(image_feat)
    # del precalDataloader


     # 初始化TensorBoard日志
    # writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

 
    for epoch in tqdm(range(0, num_epochs), desc="Total Epochs"):
        #训练阶段
        train_iter = iter(train_dataloader)

        # for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):
        for batch_idx in tqdm(range(0, 2500), desc=f"Training Epoch {epoch}/{num_epochs}"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            optimizer.zero_grad()
            torch.cuda.empty_cache()          

            gc.collect()  
            batch = todevice(batch)
            with autocast(dtype=torch.float32):
                tqdm.write(f"scene:{batch['scene_path']}")
                outputs = model(batch, shs_mask)
                alpha_mask = batch['mask'].permute(0, 1, 3, 4, 2).squeeze(1)

                # overflow_loss = torch.mean(torch.nn.functional.relu(outputs['comp_rgb'][0] - 1.0))
                # underflow_loss = torch.mean(torch.nn.functional.relu(-outputs['comp_rgb'][0]))

                rendered_image = outputs['comp_rgb'][0]  * alpha_mask

                gt_image = batch['rgb'].permute(0, 1, 3, 4, 2).squeeze(1)  * alpha_mask


                L1 = l1_loss(rendered_image, gt_image)
                ssim_value = ssim(rendered_image, gt_image)

                loss = L1 + 0.2 * (1.0 - ssim_value)

                val = psnr(gt_image, rendered_image)
            start_timer("backward")
            scaler.scale(loss).backward()        # 反向传播自动转为float32
            end_timer("backward")
            scaler.step(optimizer)
            scaler.update()

            # 使用tqdm.write替代print <button class="citation-flag" data-index="8">
            tqdm.write(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{2500}], L1 Loss: {L1.item()}, SSIM Loss: {(1.0 - ssim_value).item()}, Scene: {batch['scene_path'][0].split('/')[-2]}, PSNR: {val.item()}, shs_mask:{shs_mask}")
            iteration += 1
            img = torch.concat([gt_image, rendered_image], dim = 2)
            img = img[0]
            img_np = (img.detach().cpu().numpy() * 255).astype('uint8')
            img_save = Image.fromarray(img_np)
            img_path = os.path.join(pic_dir, f'train.png')
            img_save.save(img_path)
            

        if epoch > 0 and epoch % 5 == 0:
            save_checkpoint(model, save_dir=check_dir, epoch=epoch, step=0)

        # outputs["3dgs"][0].save_ply(f"pcd.ply")
        with torch.no_grad():
            model.eval()
            # 验证阶段进度条
            vali_idx = 0
            psnrs = []
            ssims = []
            lpipss = []
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Validation Epoch {epoch}")):  # [[新增]]
                torch.cuda.empty_cache()  

                batch = todevice(batch)
                with autocast(dtype=torch.float32):
                    outputs = model(batch, shs_mask)
        
                alpha_mask = batch['mask'].permute(0, 1, 3, 4, 2).squeeze(1, 4)

                gt = batch['rgb'][0][0] * alpha_mask
                render = rearrange(outputs['comp_rgb'][0][0], 'H W C -> C H W') * alpha_mask
                gt = gt.to(torch.float32)
                render = render.to(torch.float32)
                psnrs.append(psnr(gt, render).cpu())
                ssims.append(ssim(gt.unsqueeze(0), render.unsqueeze(0)).cpu())
                lpipss.append(lpips(gt.unsqueeze(0), render.unsqueeze(0), net_type='vgg').cpu())
                if epoch % 5 == 0:
                    img = torch.concat([gt, render], dim = 2)
                    img = rearrange(img, 'C H W -> H W C') 
                    img_np = (img.detach().cpu().numpy() * 255).astype('uint8')
                    #pdb.set_trace()
                    img_save = Image.fromarray(img_np)
                    # alpha_mask_save = Image.fromarray(alpha_mask_np)
                    img_path = os.path.join(pic_dir, f'epoch_{epoch}_{batch_idx+1}_{batch["scene_path"][0].split("/")[-2]}.png')
                    # alpha_mask_path = os.path.join(pic_dir, f'epoch_{epoch}_batch_{batch_idx+1}mask.png')
                    img_save.save(img_path)

                    if  vali_idx == 0:
                        outputs["3dgs"][0].save_ply(os.path.join(GS_dir, f"epoch_{epoch}_batch_{batch_idx+1}.ply"))
                        #tqdm.write(f"Saved image to {img_path}")  # 替代print <button class="citation-flag" data-index="8">
                vali_idx += 1

            avg_psnr = np.array(psnrs).mean()
            avg_ssim = np.array(ssims).mean()
            avg_lpips = np.array(lpipss).mean()
            tqdm.write(f"psnr: {avg_psnr}, ssim: {avg_ssim}, lpips: {avg_lpips}")
            with open(log_file_md, 'a') as f:
                f.write(f"| {epoch} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")


            model.train()

            if epoch > 0 and epoch % 10 == 0:
                shs_mask+=1
                if shs_mask >= 3:
                    shs_mask = 3

    # writer.close()
        


    # model.save_img_sequences(
    #     "video",
    #     "(\d+)\.png",
    #     save_format="mp4",
    #     fps=30,
    #     delete=True,
    # )