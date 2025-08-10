# Copyright 2025 Your Name
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

# This file is part of a project based on:
#   Original project: https://github.com/vast-engineering/xxx
#   Copyright 2024 VAST AI Research, licensed under Apache-2.0


from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from tgs.utils.typing import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from diffusers.utils.torch_utils import maybe_allow_in_graph
import pdb
# from flash_attn.flash_attn_interface import flash_attn_func

class MLAProcessor(AttnProcessor2_0):
    r"""
    Processor for Multi-Latent Attention (MLA) mechanism
    """
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: "MLAAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # 原始Processor逻辑
        residual = hidden_states
        #pdb.set_trace()
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 生成QKV
        compressed_hidden_states = attn.compressor_Q(hidden_states)
        query = attn.to_q(compressed_hidden_states)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        compressed_encoder_hidden_states_KV = attn.compressor_KV(encoder_hidden_states)
        compressed_key = attn.to_k(compressed_encoder_hidden_states_KV)
        compressed_value = attn.to_v(compressed_encoder_hidden_states_KV)

        # MLA核心：低秩压缩与解压


        # # 处理RoPE（如果存在）
        # if hasattr(attn, "rotary_emb"):
        #     decompressed_key = attn.rotary_emb(decompressed_key, seq_len=decompressed_key.shape[1])
        #     query = attn.rotary_emb(query, seq_len=query.shape[1])

        # 重塑维度
        batch_size, _, _ = query.shape
        head_dim = query.shape[-1] // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        compressed_key = compressed_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        compressed_value = compressed_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            compressed_key = attn.norm_k(compressed_key)

        #注意力计算
        hidden_states = F.scaled_dot_product_attention(
            query, compressed_key, compressed_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attn.is_causal
        )

        # hidden_states = flash_attn_func(
        #         query.contiguous(),  # 确保连续内存布局
        #         compressed_key.contiguous(),
        #         compressed_value.contiguous(),
        #         dropout_p=0.0,
        #         softmax_scale=None,
        #         causal=attn.is_causal
        #     )
        #pdb.set_trace()
        # 后处理
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
@maybe_allow_in_graph
class MLAAttention(Attention):
    def __init__(self, latent_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 低秩压缩层
        # pdb.set_trace()
        self.latent_dim = latent_dim
        self.compressor_KV = nn.Linear(self.cross_attention_dim, latent_dim, bias=False)
        self.compressor_Q = nn.Linear(self.inner_kv_dim, latent_dim, bias=False)

        self.to_k = nn.Linear(self.latent_dim, self.inner_kv_dim, bias=self.use_bias)
        self.to_v = nn.Linear(self.latent_dim, self.inner_kv_dim, bias=self.use_bias)
        self.to_q = nn.Linear(self.latent_dim, self.inner_dim, bias=self.use_bias)
        # 低秩解压层

        # 替换Processor
        self.set_processor(MLAProcessor())
        

    def forward(self, *args, **kwargs):
        # 保留原始forward逻辑，通过processor实现MLA

        return super().forward(*args, **kwargs)