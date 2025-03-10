from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm


logger = logging.get_logger(__name__)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

@maybe_allow_in_graph
class SiamLayoutJointTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False,attention_type="default",bbox_pre_only=True,bbox_with_temb = False):
        super().__init__()

        # text
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        # bbox 
        self.bbox_pre_only = bbox_pre_only 

        if bbox_pre_only:
            if bbox_with_temb:
                bbox_norm_type = "ada_norm_continous" 
            else:
                bbox_norm_type = "LayerNorm"
        else:
            bbox_norm_type = "ada_norm_zero" 

        self.bbox_norm_type = bbox_norm_type

        # img
        self.norm1 = AdaLayerNormZero(dim)

        # text
        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
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
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.attention_type = attention_type
        if self.attention_type == "layout":
            self.bbox_fuser_block = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=bbox_pre_only,
                bias=True,
                processor=processor,
            )

            self.bbox_forward = zero_module(nn.Linear(dim, dim))

            self.bbox_pre_only = bbox_pre_only
            
            
            if self.bbox_norm_type == "ada_norm_continous":
                self.norm1_bbox = AdaLayerNormContinuous(
                    dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
                )
            elif self.bbox_norm_type == "LayerNorm":
                self.norm1_bbox = nn.LayerNorm(dim)
            elif self.bbox_norm_type == "ada_norm_zero":
                self.norm1_bbox = AdaLayerNormZero(dim)
           
            
            if not self.bbox_pre_only:
                self.norm2_bbox = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                self.ff_bbox = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            else:
                self.norm2_bbox = None
                self.ff_bbox = None
        
    

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor,bbox_hidden_states=None,bbox_scale=1.0
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # img-txt MM-Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output  #gate_msa

        # Layout
        if self.attention_type == "layout" and bbox_scale!=0.0:

            if self.bbox_pre_only:
                norm_bbox_hidden_states = self.norm1_bbox(bbox_hidden_states, temb)
            else:
                norm_bbox_hidden_states, bbox_gate_msa, bbox_shift_mlp, bbox_scale_mlp, bbox_gate_mlp = self.norm1_bbox(
                    bbox_hidden_states, emb=temb
                )
            # img-bbox MM-Attention.
            img_attn_output, bbox_attn_output = self.bbox_fuser_block(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_bbox_hidden_states
            )

            attn_output = attn_output + bbox_scale*self.bbox_forward(img_attn_output) 

            if self.bbox_pre_only:
                bbox_hidden_states = None
            else:
                bbox_attn_output = bbox_gate_msa.unsqueeze(1) * bbox_attn_output
                bbox_hidden_states = bbox_hidden_states + bbox_attn_output

                norm_bbox_hidden_states = self.norm2_bbox(bbox_hidden_states)
                norm_bbox_hidden_states = norm_bbox_hidden_states * (1 + bbox_scale_mlp[:, None]) + bbox_shift_mlp[:, None]
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    bbox_ff_output = _chunked_feed_forward(
                        self.ff_bbox, norm_bbox_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    bbox_ff_output = self.ff_bbox(norm_bbox_hidden_states)
                bbox_hidden_states = bbox_hidden_states + bbox_gate_mlp.unsqueeze(1) * bbox_ff_output

        
        # Process attention outputs for the `hidden_states`.
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states,bbox_hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states