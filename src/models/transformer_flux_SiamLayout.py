# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from src.models.attention_processor_flux_SiamLayout import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


    

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    return emb


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TextBoundingboxProjection(nn.Module):
    def __init__(self, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy #64

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]


        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))

        
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        boxes,#[B,10,4]
        masks,#[B,10]
        positive_embeddings, #torch.Size([B, 10, 512,1536])
    ):
        
        B,max_box,num_token,dim = positive_embeddings.shape
        
        masks = masks.unsqueeze(-1) #torch.Size([2, 10, 1])

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # B*N*4 -> B*N*C #torch.Size([2, 10, 64])

        # learnable null embedding
        xyxy_null = self.null_position_feature.view(1, 1, -1) #torch.Size([1, 1, 64])

        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null #torch.Size([2, 10, 64])

     

        # 增加一个维度
        xyxy_embedding_unsqueezed = torch.unsqueeze(xyxy_embedding, 2)  # shape变为[2, 10, 1, 64]

        # 然后扩展到新的大小
        xyxy_embedding_expanded = xyxy_embedding_unsqueezed.expand(-1, -1, num_token, -1)  # torch.Size([2, 10, 30, 64])


       

        masks = masks.unsqueeze(-1) #torch.Size([2, 10, 1, 1])
        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, 1, -1) #从[1536]变到[1,1,1,1536]

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null #torch.Size([2, 10, 30, 1536])



        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding_expanded], dim=-1)) # torch.Size([2, 10,30, 1536+64]) ->torch.Size([2, 10,30,1536])

        objs = objs.view(B, max_box*num_token, -1)


        return objs #[B,300,1536]
    
class CustomIdentity(nn.Module):
    def __init__(self):
        super(CustomIdentity, self).__init__()

    def forward(self, img_bbox, vec=None, pe=None):
        return img_bbox

@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0,attention_type="layout"):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )
        self.attention_type = attention_type
        if self.attention_type == "layout":
            self.bbox_norm = AdaLayerNormZeroSingle(dim)
            self.bbox_proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
            self.bbox_act_mlp = nn.GELU(approximate="tanh")
            self.bbox_proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
            self.bbox_attn= Attention(
                    query_dim=dim,
                    cross_attention_dim=None,
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    out_dim=dim,
                    bias=True,
                    processor=processor,
                    qk_norm="rms_norm",
                    eps=1e-6,
                    pre_only=True,
                )
            self.bbox_forward = zero_module(nn.Linear(dim, dim))
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        bbox_hidden_states=None,
        temb: torch.FloatTensor=None,
        image_rotary_emb=None,
        image_rotary_emb_for_bbox=None,
        bbox_scale=1.0,
        bbox_end_index = 300,
        txt_end_index = 77,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        residual_bbox_hidden_states = bbox_hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        #print(attn_output.shape, mlp_hidden_states.shape)#torch.Size([1, 1101, 3072]) torch.Size([1, 1101, 12288])
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)

        # layout
        if self.attention_type == "layout" and bbox_hidden_states!=None and bbox_scale!=0.0:
            
            #hidden_states_for_img_and_bbox = torch.cat([bbox_hidden_states, hidden_states[:, txt_end_index :, ...]], dim=1)
            hidden_states_for_img_and_bbox = torch.cat([bbox_hidden_states, residual[:, txt_end_index :, ...]], dim=1)

            norm_hidden_states_for_img_and_bbox, gate_bbox = self.bbox_norm(hidden_states_for_img_and_bbox, emb=temb)
            mlp_hidden_states_for_img_and_bbox = self.bbox_act_mlp(self.bbox_proj_mlp(norm_hidden_states_for_img_and_bbox))
            attn_output_bbox = self.bbox_attn(
                hidden_states=norm_hidden_states_for_img_and_bbox,
                image_rotary_emb=image_rotary_emb_for_bbox,
                **joint_attention_kwargs,
            )
            hidden_states_for_img_and_bbox = torch.cat([attn_output_bbox, mlp_hidden_states_for_img_and_bbox], dim=2)
            gate_bbox = gate_bbox.unsqueeze(1)
            hidden_states_for_img_and_bbox = gate_bbox * self.bbox_proj_out(hidden_states_for_img_and_bbox)

            # update imgs in hidden_states
            hidden_states_bbox_img_residual = self.bbox_forward(hidden_states_for_img_and_bbox[:, bbox_end_index :, ...])
            hidden_states[:, txt_end_index :, ...] += bbox_scale * hidden_states_bbox_img_residual
            # update bbox_hidden_states 
            bbox_hidden_states = residual_bbox_hidden_states + hidden_states_for_img_and_bbox[:, :bbox_end_index, ...]

            if bbox_hidden_states.dtype == torch.float16:
                bbox_hidden_states = bbox_hidden_states.clip(-65504, 65504)
            
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states,bbox_hidden_states
      
@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6,attention_type="default"):
        super().__init__()

        self.attention_type = attention_type

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        #layout
        if self.attention_type == "layout":
            self.norm1_bbox = AdaLayerNormZero(dim)
            self.bbox_attn = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=False,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=eps,
            )
            self.norm2_bbox = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_bbox = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            self.bbox_forward = zero_module(nn.Linear(dim, dim))




    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        bbox_hidden_states=None,
        image_rotary_emb_for_bbox=None,
        bbox_scale=1.0,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        
        # img-txt Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output


        # img-bbox Attention
        # #layout. after gate_msa
        if self.attention_type == "layout" and bbox_scale!=0.0:
            norm_bbox_hidden_states, bbox_gate_msa, bbox_shift_mlp, bbox_scale_mlp, bbox_gate_mlp = self.norm1_bbox(
                bbox_hidden_states, emb=temb
            )
        
            attn_output_from_bbox, bbox_attn_output = self.bbox_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_bbox_hidden_states,
                image_rotary_emb=image_rotary_emb_for_bbox,
                **joint_attention_kwargs,
            )

            attn_output = attn_output + bbox_scale*self.bbox_forward(attn_output_from_bbox) # zero module

            
        hidden_states = hidden_states + attn_output

        
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # Process attention outputs for the `bbox_hidden_states`.
        if self.attention_type == "layout" and bbox_scale!=0.0:
            bbox_attn_output = bbox_gate_msa.unsqueeze(1) * bbox_attn_output
            bbox_hidden_states = bbox_hidden_states + bbox_attn_output
            norm_bbox_hidden_states = self.norm2_bbox(bbox_hidden_states)
            norm_bbox_hidden_states = norm_bbox_hidden_states * (1 + bbox_scale_mlp[:, None]) + bbox_shift_mlp[:, None]
            bbox_ff_output = self.ff_bbox(norm_bbox_hidden_states)
            bbox_hidden_states = bbox_hidden_states + bbox_gate_mlp.unsqueeze(1) * bbox_ff_output
        
        
        return encoder_hidden_states, hidden_states,bbox_hidden_states


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        attention_type="layout",
        single_blocks_index= [],
        double_blocks_index=[],
        is_add=True,
        gradient_checkpointing=False,
        max_boxes_token_length=30,
        fix_bbox_ids = True,
    ):
        super().__init__()
        #layout
        self.attention_type = attention_type
        self.single_blocks_index=single_blocks_index
        self.double_blocks_index=double_blocks_index
        self.is_add = is_add
        self.max_boxes_token_length = max_boxes_token_length
        self.fix_bbox_ids = fix_bbox_ids

        self.in_channels = in_channels
        self.num_layers= num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim=attention_head_dim
        self.num_attention_heads=num_attention_heads
        self.joint_attention_dim=joint_attention_dim
        self.pooled_projection_dim=pooled_projection_dim


        self.out_channels = out_channels or in_channels
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.pooled_projection_dim
        )
        print(f"FluxTransforer2DModel joint_attention_dim: {self.joint_attention_dim}, inner_dim: {self.inner_dim}")
        self.context_embedder = nn.Linear(self.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    attention_type=self.attention_type if i in self.double_blocks_index else "default",
                )
                for i in range(self.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    attention_type=self.attention_type if i in self.single_blocks_index else "default",
                )
                for i in range(self.num_single_layers)
            ]
        )

       

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = gradient_checkpointing
        
        if self.attention_type =="layout":
            self.position_net = TextBoundingboxProjection(
                positive_len=self.inner_dim, out_dim=self.inner_dim
            )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        layout_kwargs: dict | None = None,
        bbox_scale =1.0,
        bbox_ids: torch.Tensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

   
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        print("hidden_states type",hidden_states.dtype)
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # pulid input
        if 'id_embeddings' in joint_attention_kwargs and joint_attention_kwargs['id_embeddings'] is not None:
            id_embeddings = joint_attention_kwargs['id_embeddings']
            id_masks = []
            for id_mask in joint_attention_kwargs['id_masks']:
                id_masks.append(id_mask[None, :, None].repeat(hidden_states.shape[0], 1, hidden_states.shape[-1]).to(
                    device=hidden_states.device, dtype=hidden_states.dtype))
            id_weights = joint_attention_kwargs['id_weights']
            ca_idx = 0
        else:
            id_embeddings = None
            id_masks = None
            id_weights = 0.0
            ca_idx = 0

        if txt_ids.ndim == 3:
            # logger.warning(
            #     "Passing `txt_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            # logger.warning(
            #     "Passing `img_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            img_ids = img_ids[0]
        

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        #layout
        N = hidden_states.shape[0]
        if self.attention_type=="layout" and layout_kwargs is not None and layout_kwargs.get("layout", None) is not None:
            layout_args = layout_kwargs["layout"]
            bbox_raw = layout_args["boxes"].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,10,4]
            bbox_text_embeddings = layout_args["positive_embeddings"].to(dtype=hidden_states.dtype, device=hidden_states.device) #[B,10,77,4096]
            print(f"bbox_text_embeddings: {bbox_text_embeddings.shape}")
            bbox_text_embeddings = self.context_embedder(bbox_text_embeddings) # [B,10,77,3072]
            print(f"bbox_text_embeddings: {bbox_text_embeddings.shape}")
            bbox_text_embeddings = bbox_text_embeddings[:,:,:self.max_boxes_token_length,:]# [B,10,30,1536]
            bbox_masks = layout_args["bbox_masks"].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,10]
            bbox_hidden_states = self.position_net(boxes=bbox_raw,masks=bbox_masks,positive_embeddings=bbox_text_embeddings) # "bbox": torch.Size([B, 300, 1536])
            print(f"bbox_hidden_states: {bbox_hidden_states.shape},bbox_masks: {bbox_masks.shape}")

            #bbox_ids固定为0
            if self.fix_bbox_ids:
                #bbox_ids = torch.zeros(bbox_hidden_states.shape[0], bbox_hidden_states.shape[1], 3).to(device=bbox_hidden_states.device, dtype=bbox_hidden_states.dtype)
                bbox_ids = -1 * torch.ones(bbox_hidden_states.shape[0], bbox_hidden_states.shape[1], 3).to(device=bbox_hidden_states.device, dtype=bbox_hidden_states.dtype)
            else:
                # bbox_ids与其他ids不一样
                bbox_ids = torch.zeros(bbox_hidden_states.shape[1], 3).to(device=bbox_hidden_states.device, dtype=bbox_hidden_states.dtype)
                max_img_id = img_ids.max()
                #print(f"max_img_id: {max_img_id}")
                bbox_ids_bs = bbox_hidden_states.shape[1] // self.max_boxes_token_length
                # 按批次填充bbox_ids
                for i in range(bbox_ids_bs):
                    start_idx = i * self.max_boxes_token_length
                    end_idx = (i + 1) * self.max_boxes_token_length
                    bbox_ids[start_idx:end_idx] = max_img_id + 1 + i
                #print("bbox_ids:", bbox_ids,bbox_ids.shape)

            if bbox_ids.ndim == 3:
                # logger.warning(
                #     "Passing `bbox_ids` 3d torch.Tensor is deprecated."
                #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
                # )
                bbox_ids = bbox_ids[0] #[300,3]

            ids_for_bbox = torch.cat((bbox_ids, img_ids), dim=0)
            image_rotary_emb_for_bbox = self.pos_embed(ids_for_bbox)

        else:
            bbox_hidden_states = None
            bbox_masks = None
            image_rotary_emb_for_bbox = None

        
        #double
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states, bbox_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    bbox_hidden_states,
                    image_rotary_emb_for_bbox,
                    bbox_scale,
                    **ckpt_kwargs
                )

            else:
                print(
                    f"double transformer input hidden_states: {hidden_states.shape} bbox_hidden_states: {bbox_hidden_states.shape}")
                encoder_hidden_states, hidden_states, bbox_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    bbox_hidden_states=bbox_hidden_states,
                    image_rotary_emb_for_bbox=image_rotary_emb_for_bbox,
                    bbox_scale=bbox_scale,
                    joint_attention_kwargs=joint_attention_kwargs
                )
                print(f"double transformer output hidden_states: {hidden_states.shape} bbox_hidden_states: {bbox_hidden_states.shape}")

                print(f"id_weights",id_weights)
                if id_embeddings is not None and index_block % self.pulid_double_interval == 0:
                    for id_weight, id_embedding, id_mask in zip(id_weights, id_embeddings, id_masks):
                        print(f"----------pulid ca id_weight:{id_weight}, id_mask:{type(id_mask)}")
                        print(f"id_embedding shape:{id_embedding.shape} hidden_states shape:{hidden_states.shape}, id_mask shape:{id_mask.shape}")
                        
                        pulid_states = id_weight[0] * self.pulid_ca[ca_idx](id_embedding, hidden_states) * id_mask
                        print(f"double transformer pulid_states shape:{pulid_states.shape}")
                        hidden_states = hidden_states + pulid_states
                    ca_idx += 1


            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        #single
        bbox_end_index = bbox_hidden_states.shape[1]
        txt_end_index = encoder_hidden_states.shape[1]
        print(f"cat encoder_hidden_states before hidden_states: {hidden_states.shape} ")
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        print(f"cat encoder_hidden_states after hidden_states: {hidden_states.shape} ")
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states,bbox_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    bbox_hidden_states,
                    temb,
                    image_rotary_emb,
                    image_rotary_emb_for_bbox,
                    bbox_scale,
                    bbox_end_index,
                    txt_end_index,
                    **ckpt_kwargs,
                )
            else:
                print(
                    f"single transformer input hidden_states: {hidden_states.shape} bbox_hidden_states: {bbox_hidden_states.shape}")
                hidden_states,bbox_hidden_states = block(
                    hidden_states=hidden_states,
                    bbox_hidden_states = bbox_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_rotary_emb_for_bbox =image_rotary_emb_for_bbox,
                    bbox_scale=bbox_scale,
                    bbox_end_index = bbox_end_index,
                    txt_end_index = txt_end_index,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                print(f"single transformer output hidden_states: {hidden_states.shape} bbox_hidden_states: {bbox_hidden_states.shape}")

                # pulid compute
                # split
                encoder_hidden_states, hidden_states = hidden_states[:, :encoder_hidden_states.shape[1],
                                                       ...], hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                if id_embeddings is not None and index_block % self.pulid_single_interval == 0:
                    for id_weight, id_embedding, id_mask in zip(id_weights, id_embeddings, id_masks):
                        mask_pulid_states = id_weight[0] * self.pulid_ca[ca_idx](id_embedding, hidden_states) * id_mask
                        print(f"single transformer mask_pulid_states shape:{mask_pulid_states.shape}")
                        hidden_states = hidden_states + mask_pulid_states
                        print(
                            f"id_embedding shape:{id_embedding.shape} hidden_states shape:{hidden_states.shape}, id_mask shape:{id_mask.shape}")
                    ca_idx += 1

                # merge
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)


        # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
