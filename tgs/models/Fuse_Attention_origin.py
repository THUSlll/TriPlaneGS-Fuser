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

class FuseProcessor(AttnProcessor2_0):
    r"""
    Processor for Multi-Latent Attention (MLA) mechanism
    """
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: "FuseAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # 原始Processor逻辑
    
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 生成QKV
        query = attn.to_q(hidden_states)
        
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        # compressed_value = attn.to_v(compressed_encoder_hidden_states_KV)

        # MLA核心：低秩压缩与解压


        # # 处理RoPE（如果存在）
        # if hasattr(attn, "rotary_emb"):
        #     decompressed_key = attn.rotary_emb(decompressed_key, seq_len=decompressed_key.shape[1])
        #     query = attn.rotary_emb(query, seq_len=query.shape[1])

        # 重塑维度
        # batch_size, _, _ = query.shape
        # head_dim = query.shape[-1] // attn.heads
        
        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # compressed_value = compressed_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        batch_size = query.size(0)
        query = query.view(-1, query.shape[-1])  # 调整为 [batch_size, dim]
        key = key.view(-1, key.shape[-1])   
        omega = torch.sum(query * key, dim = -1)

        return omega
    
@maybe_allow_in_graph
class FuseAttention(Attention):
    def __init__(self, inner_div, out_div, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 低秩压缩层


        self.to_k = nn.Linear(inner_div, out_div, bias=self.use_bias)

        self.to_q = nn.Linear(inner_div, out_div, bias=self.use_bias)
        # 低秩解压层

        # 替换Processor
        self.set_processor(FuseProcessor())
        

    def forward(self, *args, **kwargs):
        # 保留原始forward逻辑，通过processor实现MLA

        return super().forward(*args, **kwargs)
    

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
        
        self.attn = FuseAttention(query_dim=self.cfg.inner_div, inner_div=self.cfg.inner_div,  out_div = self.cfg.out_div, heads=self.cfg.n_heads, dim_head=self.cfg.d_head)

    def forward(
        self,
        scene_codes: list,
    ):
        omegas = []
        omegas_sum = torch.zeros([1, 3, 1, self.cfg.plane_size , self.cfg.plane_size], device='cuda')
        # 对输入场景编码进行归一化
        normalized_scene_codes = []
        for tokens in scene_codes:
            # 使用均值和标准差进行归一化
            mean = tokens.mean(dim=2, keepdim=True)
            std = tokens.std(dim=2, keepdim=True).clamp(min=1e-6)
            normalized_tokens = (tokens - mean) / std
            normalized_scene_codes.append(normalized_tokens)
        
        for id, tokens in enumerate(normalized_scene_codes):
            rearranged_tokens = rearrange(tokens, 'B Np C Hp Wp -> B (Np Hp Wp) C')

            omega = self.attn(self.fuse_tokens.unsqueeze(0), rearranged_tokens).unsqueeze(0).unsqueeze(2)

            omega = rearrange(omega, 'B (Np Hp Wp) C -> B Np C Hp Wp', Np = 3, Hp = self.cfg.plane_size, Wp = self.cfg.plane_size)
            omegas.append(omega)
            omegas_sum += omega
        
        fused_scene_code = torch.zeros_like(scene_codes[0], device='cuda')
        
        for i in range(len(omegas)):
            omegas[i] = omegas[i] / omegas_sum
            fused_scene_code += scene_codes[i] * omegas[i]
        
        # 对融合后的场景编码进行归一化
        mean = fused_scene_code.mean(dim=2, keepdim=True)
        std = fused_scene_code.std(dim=2, keepdim=True).clamp(min=1e-6)
        fused_scene_code = (fused_scene_code - mean) / std
        
        return omegas, fused_scene_code