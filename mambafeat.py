import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import tgs
from tgs.models.image_feature import UNet, add_positional_encoding, flatten_spatial, add_plucker_channels
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *
from tgs.utils.loss_def import l1_loss, ssim
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pdb
from TriplaneGaussian.tgs.models.triplane.triplane_transformer import ImageTriplaneGenerator
from tgs.models.triplane.point_triplane import PointTriplaneGenerator
from TriplaneGaussian.tgs.models.Fuse_Attention_origin import FuseNetwork
from tgs.models.pointclouds.samplefarthest import grid_sample, grid_sample_centers, remove_outliers_knn
from transformers import AutoModel
from timm.data.transforms_factory import create_transform
import torchvision.transforms.functional as TF # 用于可能的张量变换
from tgs.models.pointclouds.pointnet2 import Pointnet2
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


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

        # fuser_cls: str = ""
        # fuser: dict = field(default_factory=dict)
    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )

        model_state_dict = self.state_dict()

        # 过滤掉不以 'image_tokenizer' 开头的参数
        filtered_state_dict = {k: v for k, v in state_dict.items() if (k.startswith('image_tokenizer') or k.startswith('camera_embedder')) and k in model_state_dict and v.size() == model_state_dict[k].size()}
        # pdb.set_trace()
        self.load_state_dict(filtered_state_dict, strict=False)

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


        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )


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

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)


        self.point_encoder = tgs.find(self.cfg.pointcloud_encoder_cls)(self.cfg.pointcloud_encoder)

        self.pose_pos_embed = torch.tensor([[0, 0]], device = 'cuda')
        # load checkpoint
        # if self.cfg.weights is not None:
        #     self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)


        # self.UnetFeature = UNet(9, 128)

        self.triplaneGenerator = ImageTriplaneGenerator(128)
        self.pointTriplaneGenerator = PointTriplaneGenerator(128)

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

    def forward(self, batch: Dict[str, Any], shs_mask:Optional[int] = 0) -> Dict[str, Any]:

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        rgb_cond_flat = rearrange(batch["rgb"], 'b nv c h w -> (b nv) c h w')
        resized_inputs = TF.resize(rgb_cond_flat, self.mambavision_input_resolution[1:], antialias=True)
        normalized_inputs = TF.normalize(resized_inputs, mean=self.mambavision_config.mean, std=self.mambavision_config.std)
        inputs_for_mamba = normalized_inputs.cuda() # 确保在 GPU 上
        self.mambavision_backbone.eval() # 或者确保在训练循环外设置

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

        original_size = batch["rgb_cond"].shape[-2:]  # 获取原始图像的高度和宽度
        image_features = torch.nn.functional.interpolate(
            image_features.flatten(0, 1),  # 将B和Nv维度合并
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).view(batch_size, 1, -1, *original_size)  # 恢复B和Nv维度
        
        #get image features for projection

        torch.cuda.empty_cache()  # 清理缓存

        # 3GB
        return image_features



if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.data_GS import CustomGaussianTransDataset
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
    model_path = "exp/test_2025_5_12_2/checkpoint/model_checkpoint_epoch_50_step_0.pth"

    #pdb.set_trace()
    cfg.system.weights=model_path
    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)

    # checkpoint = torch.load(model_path, map_location="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.train()
    print("load model ckpt done.")

    dataset = CustomGaussianTransDataset(cfg.data)

    dataloader = DataLoader(dataset,
                        batch_size=cfg.data.eval_batch_size, 
                        num_workers=cfg.data.num_workers,
                        shuffle=True,
                        collate_fn=dataset.collate,
                    )
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


    data_size = dataset.__len__()
    num_epochs = 150
    print("train start")
    iteration = 0
    
    # 移除重复的环境变量设置
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
    
    scaler = GradScaler()
    shs_mask = 0

    ## LOG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_md = os.path.join(save_dir, f'psnr_log_{timestamp}.md')
    with open(log_file_md, 'w') as f:
        f.write("# PSNR Training Log\n")
        f.write("| Epoch | PSNR  |\n")
        f.write("|-------|-------|\n")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training")):  # [[新增]]
        optimizer.zero_grad()
        torch.cuda.empty_cache()            
        batch = todevice(batch)
        with autocast(dtype=torch.float32):
            outputs = model(batch, shs_mask)
            outputs = outputs.detach().cpu().numpy()
            np.save(os.path.join(batch["image_path"][0], batch['image_name'][0] + 'feat.npy'), outputs[0, 0])

        iteration += 1

    # model.save_img_sequences(
    #     "video",
    #     "(\d+)\.png",
    #     save_format="mp4",
    #     fps=30,
    #     delete=True,
    # )