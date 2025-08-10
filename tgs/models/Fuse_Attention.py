# Copyright 2024 VAST AI Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Bo Liu, 2025
# - Added adaptive density control
# - Improved rendering stability

from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from tgs.utils.typing import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from diffusers.utils.torch_utils import maybe_allow_in_graph
import pdb
# from flash_attn.flash_attn_interface import flash_attn_func
from tgs.utils.base import BaseModule
from dataclasses import dataclass
from einops import rearrange

class FuseAttention(nn.Module):
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
        # query: [B, 3*H*W, C]
        # context: [B, 3*H*W, Nt, C]
        B, L, C = query.shape
        Nv = context.shape[2]

        query = self.norm(query)  # [B, 3*H*W, C]
        k = self.to_k(context)    # [B, 3*H*W, Nv, C]
        v = self.to_v(context)    # [B, 3*H*W, Nv, C]

        # 多头拆分
        query = query.view(B, L, self.n_heads, self.d_head)  # [B, 3*H*W, n_heads, d_head]
        k = k.view(B, L, Nv, self.n_heads, self.d_head)      # [B, 3*H*W, Nv, n_heads, d_head]
        v = v.view(B, L, Nv, self.n_heads, self.d_head)      # [B, 3*H*W, Nv, n_heads, d_head]

        # 交换head和view维度，方便后续计算
        query = query.permute(0, 2, 1, 3)   # [B, n_heads, 3*H*W, d_head]
        k = k.permute(0, 3, 1, 2, 4)        # [B, n_heads, 3*H*W, Nv, d_head]
        v = v.permute(0, 3, 1, 2, 4)        # [B, n_heads, 3*H*W, Nv, d_head]

        # 逐点多头注意力: 每个空间点、每个head，query和Nv个key做点积
        # query: [B, n_heads, 3*H*W, d_head]
        # k:     [B, n_heads, 3*H*W, Nv, d_head]
        attn_score = (query.unsqueeze(3) * k).sum(-1)  # [B, n_heads, 3*H*W, Nv]
        attn_score = attn_score / (self.d_head ** 0.5)
        attn_prob = torch.softmax(attn_score, dim=-1)  # [B, n_heads, 3*H*W, Nv]

        # 加权value
        out = (attn_prob.unsqueeze(-1) * v).sum(3)  # [B, n_heads, 3*H*W, d_head]

        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)  # [B, 3*H*W, C]
        # attn_prob: [B, n_heads, 3*H*W, Nv]
        return out, attn_prob

class FuseNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int  = 128
        inner_div: int = 395
        out_div: int = 395
        n_heads: int = 8
        d_head: int = 64

    cfg: Config
    def configure(self) -> None:
        super().configure()
        self.fuse_tokens = torch.nn.Parameter(
            torch.randn((3 * self.cfg.plane_size * self.cfg.plane_size , self.cfg.inner_div), dtype=torch.float32, device='cuda')
            )
        
        self.attn = FuseAttention(query_dim=self.cfg.inner_div, context_dim=self.cfg.inner_div, n_heads=self.cfg.n_heads)
        self.norm = nn.LayerNorm(self.cfg.inner_div)
    def forward(
        self,
        scene_codes: Tensor,
    ):

        scene_codes = rearrange(scene_codes, 'B Nt Np C Hp Wp -> B (Np Hp Wp) Nt C')

        scene_codes = self.norm(scene_codes)

        query = self.fuse_tokens.unsqueeze(0)

        attn_out, prob = self.attn(query, scene_codes)

        attn_out = rearrange(attn_out, 'B (Np Hp Wp) C -> B Np C Hp Wp', Np = 3, Hp = self.cfg.plane_size, Wp = self.cfg.plane_size)

        return prob, attn_out