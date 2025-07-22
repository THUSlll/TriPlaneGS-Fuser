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

from TriplaneGaussian.tgs.models.Fuse_Attention_origin import FuseNetwork
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
        # self.FuseNet = FuseNetwork(self.cfg.fuser)

        self.UnetFeature = UNet(9, 128)





    def forward(self, batch: Dict[str, Any], shs_mask:Optional[int] = 0) -> Dict[str, Any]:
        # generate point cloud
        # batch_size = batch["rgb"].shape[0]

        rgb_feats = torch.cat([batch["features_dc"], batch["features_extra"]], dim=-1).flatten(2)
        GS_feats = torch.cat([batch["rots"], batch["scales"], batch["opacities"], rgb_feats], dim=-1)

        pointclouds = batch["xyz"]
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        # camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        # camera_intri = batch["intrinsic_cond"].view(*batch["intrinsic_cond"].shape[:-2], -1)
        camera_extri = batch["c2w_cond"]
        camera_intri = batch["intrinsic_cond"]
        
        #camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        #pdb.set_trace()

        # with torch.no_grad():
        #camera_feats = self.camera_embedder(camera_feats)

        #camera_feats = torch.cat([camera_feats, self.pose_pos_embed.expand(1, batch_size * 20, 2)], dim = 2)
        
        #camera_feats = rearrange(camera_feats, 'L B_Nv C -> B_Nv C L')
        #pdb.set_trace()

        #print(batch["rgb_cond"].size())
        input_images = rearrange(batch["rgb_cond"], 'B Nv C H W -> (B Nv) C H W')
        camera_extri = rearrange(camera_extri, 'B Nv H W -> (B Nv) H W')
        camera_intri = rearrange(camera_intri, 'B Nv H W -> (B Nv) H W')

        input_images = add_plucker_channels(input_images, camera_extri, camera_intri)

        # input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
        #         batch["rgb_cond"],
        #         modulation_cond=camera_feats,
        #     )#15GB
        # del camera_feats
        # input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)


        image_feature: Float[Tensor, "(B Nv) Cit H W"] = self.UnetFeature(input_images)['dec3']
      
        image_feature_withpos = add_positional_encoding(image_feature)

        input_image_tokens = flatten_spatial(image_feature_withpos)

        #input_image_tokens = torch.cat([input_image_tokens, camera_feats], dim=-1)
        #pdb.set_trace()
        #get image features for projection
        # image_features = self.image_feature(
        #     rgb = batch["rgb_cond"],
        #     mask = batch.get("mask_cond", None),
        #     feature = input_image_tokens
        # )
        # # torch.cuda.empty_cache()  # 清理缓存
        # # #only support number of input view is one
        # c2w_cond = batch["c2w_cond"]
        # intrinsic_cond = batch["intrinsic_cond"]

        # # print(pointclouds.size())
        # # print(c2w_cond.size())
        # # print(intrinsic_cond.size())
        # # print(image_features.size())

        # proj_feats = points_projection(pointclouds, c2w_cond, intrinsic_cond, image_features)
        # del image_features  # 释放投影前的特征显存
        # torch.cuda.empty_cache()  # 再次清理缓存

        # print(pointclouds.size())
        # print(proj_feats.size())
        #pdb.set_trace()
        # point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats, GS_feats], dim=-1))
        point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, GS_feats], dim=-1))

        #print(point_cond_embeddings)
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)
        input_image_tokens = rearrange(input_image_tokens, '(B Nv) C N -> B (Nv N) C', B = batch_size)
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens)) #转换为三平面并上采样
        # rend_out = self.renderer(scene_codes,
        #                         query_points=pointclouds,
        #                         additional_features=torch.cat([proj_feats, GS_feats], dim=-1),
        #                         **batch)
        #pdb.set_trace()

        rend_out = self.renderer(scene_codes,
                                query_points=pointclouds,
                                # additional_features = GS_feats,
                                shs_mask = shs_mask,
                                **batch)

        return rend_out
    
        

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
    model_path = "exp/test_2025_4_4/checkpoint/model_checkpoint_epoch_50_step_0.pth"

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
    scaler = GradScaler()
    shs_mask = 0

    ## LOG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_md = os.path.join(save_dir, f'psnr_log_{timestamp}.md')  # Markdown格式
    with open(log_file_md, 'w') as f:
        f.write("# PSNR Training Log\n")
        f.write("| Epoch | PSNR  |\n")
        f.write("|-------|-------|\n")

    for epoch in tqdm(range(num_epochs), desc="Total Epochs"):  # 主进度条（Epoch级别）[[新增]]

        # #训练阶段进度条
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):  # [[新增]]
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            batch = todevice(batch)
            with autocast(dtype=torch.float32):
                outputs = model(batch, shs_mask)
                alpha_mask = batch['mask'].permute(0, 1, 3, 4, 2).squeeze(1)

                # overflow_loss = torch.mean(torch.nn.functional.relu(outputs['comp_rgb'][0] - 1.0))
                # underflow_loss = torch.mean(torch.nn.functional.relu(-outputs['comp_rgb'][0]))

                rendered_image = outputs['comp_rgb'][0] * alpha_mask


                gt_image = batch['rgb'].permute(0, 1, 3, 4, 2).squeeze(1) * alpha_mask
                L1 = l1_loss(rendered_image, gt_image)
                ssim_value = ssim(rendered_image, gt_image)

                loss = L1 + 0.2 * (1.0 - ssim_value)

            scaler.scale(loss).backward()        # 反向传播自动转为float32
            scaler.step(optimizer)
            scaler.update()

            # 使用tqdm.write替代print <button class="citation-flag" data-index="8">
            tqdm.write(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{data_size}], L1 Loss: {L1.item()}, SSIM Loss: {(1.0 - ssim_value).item()}, shs_mask:{shs_mask}")
            
            iteration += 1

        # outputs["3dgs"][0].save_ply(f"pcd.ply")
        if epoch >= 0:
            with torch.no_grad():
                model.eval()
                # 验证阶段进度条
                vali_idx = 0
                psnrs = []
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch}")):  # [[新增]]
                    if vali_idx % 50 == 0:

                        batch = todevice(batch)
                        with autocast(dtype=torch.float32):
                            outputs = model(batch, shs_mask)
                        
                        
                        gt = batch['rgb'][0][0]
                        render = rearrange(outputs['comp_rgb'][0][0], 'H W C -> C H W')
                        # render = torch.clamp(render, max=1.0, min=0.0)
                        psnrs.append(psnr(gt, render).cpu())


                        img = torch.concat([gt, render], dim = 2)


                        img = rearrange(img, 'C H W -> H W C') 

                        img_np = (img.detach().cpu().numpy() * 255).astype('uint8')

                        #pdb.set_trace()
                        img_save = Image.fromarray(img_np)
                        img_path = os.path.join(pic_dir, f'epoch_{epoch}_batch_{batch_idx+1}.png')
                        img_save.save(img_path)
                            #使用tqdm.write避免干扰进度条显示 <button class="citation-flag" data-index="8">

                        ######################  颜色溢出监控
                        img_overflow = torch.nn.functional.relu(img - 1.0) + torch.nn.functional.relu(-img)
                        img_overflow_np = (img_overflow.detach().cpu().numpy() * 255).astype('uint8')
                        img_overflow_save = Image.fromarray(img_overflow_np)
                        img_overflow_path = os.path.join(pic_dir, f'epoch_{epoch}_batch_{batch_idx+1}_overflow.png')
                        img_overflow_save.save(img_overflow_path)

                        if  vali_idx % 200 == 0 and vali_idx > 1500:
                            outputs["3dgs"][0].save_ply(os.path.join(GS_dir, f"epoch_{epoch}_batch_{batch_idx+1}.ply"))
                            #tqdm.write(f"Saved image to {img_path}")  # 替代print <button class="citation-flag" data-index="8">
                    vali_idx += 1

                psnrs = np.array(psnrs)
                avg_psnr = psnrs.mean()
                tqdm.write(f"psnr: {avg_psnr}")  # 替代print <button class="citation-flag" data-index="8">
                with open(log_file_md, 'a') as f:
                    f.write(f"| {epoch} | {avg_psnr:.4f} |\n")


                model.train()
                if epoch % 50 == 0:
                    save_checkpoint(model, save_dir=check_dir, epoch=epoch, step=0)
                if epoch % 10 == 0:
                    shs_mask+=1
                    if shs_mask >= 3:
                        shs_mask = 3

        


    # model.save_img_sequences(
    #     "video",
    #     "(\d+)\.png",
    #     save_format="mp4",
    #     fps=30,
    #     delete=True,
    # )