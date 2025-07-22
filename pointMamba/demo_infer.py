import torch
from models import build_model_from_cfg
from utils.config import cfg_from_yaml_file
import pdb
# 1. 读取yaml配置，提取model字段
config = cfg_from_yaml_file('cfgs/test.yaml')
model_cfg = config.model

# 2. 实例化模型
model = build_model_from_cfg(model_cfg)
model.eval()

# 3. 加载权重（假设权重文件为 'your_ckpt.pth'，请替换为实际路径）

# ckpt = torch.load('checkpoints/pretrain.pth', map_location='cpu')
# # 兼容不同权重结构
# if 'model' in ckpt:
#     model.load_state_dict(ckpt['model'], strict=False)
# elif 'base_model' in ckpt:
#     model.load_state_dict(ckpt['base_model'], strict=False)
# else:
#     model.load_state_dict(ckpt, strict=False)

# # 4. 冻结参数
# for param in model.parameters():
#     param.requires_grad = False

# 5. 放到GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 6. 构造随机输入（假设输入为 [B, N, 3]，B=2, N=1024）
pts = torch.randn(1, 321942, 56).to(device)

# 7. 前向推理
with torch.no_grad():
    output = model(pts)
pdb.set_trace()
print("输出结果 shape:", output)