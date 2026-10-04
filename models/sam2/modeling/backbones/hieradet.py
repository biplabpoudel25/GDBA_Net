from functools import partial
from typing import List, Tuple, Union
from itertools import repeat
import pywt
import math
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

from ...modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from ...modeling.sam2_utils import DropPath, MLP

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x
        
class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class LearnableGaborBank(nn.Module):
    """
    Learnable multi-orientation, multi-scale Gabor filter bank.
    Produces oriented magnitude maps useful for histopathology textures.
    """
    def __init__(self, in_channels=3, kernel_size=13,
                 orientations=8, scales=(2.0, 3.0, 4.5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.orientations = orientations
        self.scales = scales
        self.in_channels = in_channels
        self.out_channels = len(scales) * orientations * in_channels

        # learnable parameters
        self.freq = nn.Parameter(torch.randn(len(scales), orientations) * 0.1 + 0.25)
        self.theta = nn.Parameter(
            torch.linspace(0, math.pi, orientations).repeat(len(scales), 1)
        )
        self.sigma = nn.Parameter(torch.ones(len(scales), orientations) * 3.0)
        self.gamma = nn.Parameter(torch.ones(len(scales), orientations) * 0.5)

        # projection and normalization
        self.proj = nn.Conv2d(self.out_channels,
                              self.out_channels // 2,
                              kernel_size=1)
        self.norm = nn.LayerNorm(self.out_channels // 2)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        device = x.device
        filters = []
        k = self.kernel_size // 2
        y, xg = torch.meshgrid(
            torch.arange(-k, k + 1, device=device, dtype=torch.float32),
            torch.arange(-k, k + 1, device=device, dtype=torch.float32),
            indexing="ij"
        )

        # build filter bank
        for s_idx, scale in enumerate(self.scales):
            for o_idx in range(self.orientations):
                sigma = self.sigma[s_idx, o_idx].clamp(1.0, 6.0)
                freq = self.freq[s_idx, o_idx].clamp(0.05, 0.5)
                theta = self.theta[s_idx, o_idx]
                gamma = self.gamma[s_idx, o_idx].clamp(0.3, 0.9)

                xr = xg * torch.cos(theta) + y * torch.sin(theta)
                yr = -xg * torch.sin(theta) + y * torch.cos(theta)
                gb = torch.exp(
                    -0.5 * ((xr ** 2 + (gamma ** 2) * yr ** 2) / sigma ** 2)
                ) * torch.cos(2 * math.pi * freq * xr)
                gb = gb / gb.abs().sum()
                filters.append(gb)

        filters = torch.stack(filters).unsqueeze(1).repeat(self.in_channels, 1, 1, 1)
        out = F.conv2d(x, filters, padding=self.kernel_size // 2,
                       groups=self.in_channels)
        out = torch.abs(out)

        # project and normalize
        out = self.proj(out)
        B, C, H, W = out.shape
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#        print("Gabor out shape:", out.shape)
        return out

class MultiLevelDWTExtractor(nn.Module):
    """
    Multi-level DWT feature extractor for handcrafted features.
    Compatible with transformer-based backbones (returns [B, 3, H, W]).
    """

    def __init__(self, in_channels=3, wavelet='db2', levels=2):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.in_channels = in_channels

        # total subbands: LL + (LH, HL, HH) * levels
        self.out_channels = in_channels * (1 + 3 * levels)

        # project all subbands back to 3 channels
        self.reduce = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=False
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input tensor
        Returns:
            out: [B, 3, H, W] handcrafted feature map (aligned with input)
        """
        B, C, H, W = x.shape
        outs = []

        # loop batch-wise for wavelet decomposition
        for i in range(B):
            chs = []
            for c in range(C):
                arr = x[i, c].detach().cpu().numpy()
                coeffs = pywt.wavedec2(arr, wavelet=self.wavelet, level=self.levels)

                # collect all subbands: LL + (LH, HL, HH)*levels
                bands = [coeffs[0]]
                for l in range(1, len(coeffs)):
                    LH, HL, HH = coeffs[l]
                    bands += [LH, HL, HH]

                # resize all subbands to LL shape for consistent stacking
                LL_shape = coeffs[0].shape[::-1]
                merged = np.stack(
                    [cv2.resize(b, LL_shape) for b in bands],
                    axis=0
                )
                chs.append(torch.tensor(merged, device=x.device))
            outs.append(torch.cat(chs, dim=0))

        # stack all batches -> [B, out_channels, H', W']
        out = torch.stack(outs).float()

        # interpolate back to input spatial size [H, W]
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # reduce subbands → 3 channels for compatibility
        out = self.reduce(out)

        # layer normalization (normalize along channel dim)
        B, C, H, W = out.shape
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        in_chans=3,
        embed_dims=[144, 288, 576, 1152],
        img_size: int = 1024,
        patch_size: int = 4,
        embed_dim: int = 144,# initial embed dim
        num_heads: int = 2,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        handcrafted_tune: bool = True,
        embedding_tune: bool = True,
        use_gating: bool = True,
        shared_projection: bool = True,
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 6, 36, 4),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            16,
            8
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            23,
            33,
            43
        ),
        return_interm_layers=True,  # return feats from every stage
        adapter_internal: str = 'mlp',  # 'mlp' or 'attention'
    ):
        super().__init__()
        
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        cur_stage = 1
        self.embed_dim = embed_dims
        self.depth = stages
        self.blocks = nn.ModuleList()
        self.scale_factor = 32
        self.prompt_type = 'highpass' 
        self.tuning_stage = "1234"
        self.input_type = 'fft'      # [OPTIONS: fft, all, gaussian, srm, wavelet, gabor]
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'gated_dual_branch' # [OPTIONS: adaptor, gated_dual_branch]
        self.use_gating = True
        self.shared_projection = True

        self.adapter_internal = adapter_internal  # 'mlp' or 'attention'

        # Build prompt generator (GDBA aware)
        self.prompt_generator = PromptGenerator(
            self.scale_factor, self.prompt_type, self.embed_dim,
            self.tuning_stage, self.depth,
            self.input_type, self.freq_nums,
            self.handcrafted_tune, self.embedding_tune, self.adaptor,
            img_size,
            adapter_internal=self.adapter_internal,
            use_gating = self.use_gating,
            shared_projection = self.shared_projection,
        )

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inp = x
        x = self.patch_embed(x)  # [B,H,W,C_init]

        # handcrafted features per stage (already low-dim per stage)
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(inp)

        self.block1 = []
        self.block2 = []
        self.block3 = []
        self.block4 = []
        outputs = []

        for i, blk in enumerate(self.blocks):
            if i < 3:
                self.block1.append(blk)
            elif 2 < i < 9:
                self.block2.append(blk)
            elif 8 < i < 45:
                self.block3.append(blk)
            elif 44 < i:
                self.block4.append(blk)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        if '1' in self.tuning_stage:
            prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
        for i, blk in enumerate(self.block1):
            if '1' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
            x = blk(x)
            if i == 1:
                outputs.append(x.permute(0, 3, 1, 2))

        if '2' in self.tuning_stage:
            prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
        for i, blk in enumerate(self.block2):
            if '2' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
            x = blk(x)
            if i == 4:
                outputs.append(x.permute(0, 3, 1, 2))

        if '3' in self.tuning_stage:
            prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
        for i, blk in enumerate(self.block3):
            if '3' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt3, 3, i)
            x = blk(x)
            if i == 34:
                outputs.append(x.permute(0, 3, 1, 2))

        if '4' in self.tuning_stage:
            prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
        for i, blk in enumerate(self.block4):
            if '4' in self.tuning_stage:
                x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
            x = blk(x)
            if i == 2:
                outputs.append(x.permute(0, 3, 1, 2))

        return outputs
        
        
def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
        

class GridMLPAdapter(nn.Module):
    """
    Enhanced Grid MLP Adapter for feature modulation.
    Input: [B, H, W, C_low]
    """
    def __init__(self, dim_low, hidden_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim_low * hidden_ratio)

        self.norm = nn.LayerNorm(dim_low)

        self.mlp = nn.Sequential(
            nn.Linear(dim_low, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_low),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, H, W, C_low]
        """
        residual = x
        x = self.norm(x)         # Apply LN before MLP
        x = self.mlp(x)          # Bottleneck MLP
        return residual + x      # Residual connection


class PromptGenerator(nn.Module):
    """
    Gated Dual-Branch Adapter for hierarchical encoders

    Supports 'adaptor' modes:
        * 'gated_dual_branch'  (NEW)  <-- recommended
        * 'adaptor'            (legacy single-branch lightweight MLP)
        - adapter_internal: 'mlp' or 'attention'  (for 'gated_dual_branch' only)

    """
    def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size,
                 adapter_internal: str = 'mlp',use_gating=True, shared_projection = True):
        super().__init__()

        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims  # list of 4 stage dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage  # e.g., "1234"
        self.depths = depths  # tuple of 4 stage depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adapter_internal = adapter_internal  # 'mlp' or 'attention'
        self.enable_viz = False
        self.attn_maps = {}
        self.gates = {}
        self.adaptor = adaptor

        self.use_gating = use_gating # False for no gating
        self.shared_projection = shared_projection  # False for per-depth

        if self.input_type == 'gaussian':
            self.gaussian_filter = GaussianFilter()
        if self.input_type == 'srm':
            self.srm_filter = SRMFilter()
        if self.input_type == 'all':
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)
        if self.input_type == 'gabor':
            self.gabor_filter = LearnableGaborBank(in_channels=3)
            self.gabor_reduce = nn.Conv2d(
#                                in_channels=self.gabor_filter.out_channels,  
                                in_channels=36,  
                                out_channels=3,  # back to RGB-like 3-channel
                                kernel_size=1,
                                bias=False
                            )
        if self.input_type == 'wavelet':
            self.wavelet_filter = MultiLevelDWTExtractor(in_channels=3, wavelet='db2', levels=2)
            

        # Handcrafted feature generators (per stage)
        if self.handcrafted_tune:
            print("Inside handcrafted features")
            if '1' in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                                                embed_dim=self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                                in_chans=self.embed_dims[0] // self.scale_factor,
                                                                embed_dim=self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                                in_chans=self.embed_dims[1] // self.scale_factor,
                                                                embed_dim=self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                                in_chans=self.embed_dims[2] // self.scale_factor,
                                                                embed_dim=self.embed_dims[3] // self.scale_factor)

        # Embedding projection to low-dim (per stage)
        if self.embedding_tune:
            if '1' in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
            if '2' in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
            if '3' in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
            if '4' in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)

        # Adaptor heads
        if self.adaptor == 'adaptor':
            if '1' in self.tuning_stage:
                for i in range(self.depths[0] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
                        nn.GELU(),
                    )
                    setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

            if '2' in self.tuning_stage:
                for i in range(self.depths[1] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
                        nn.GELU(),
                    )
                    setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

            if '3' in self.tuning_stage:
                for i in range(self.depths[2] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
                        nn.GELU(),
                    )
                    setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

            if '4' in self.tuning_stage:
                for i in range(self.depths[3] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
                        nn.GELU(),
                    )
                    setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
                self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])

        elif self.adaptor == 'gated_dual_branch':
            # per-depth dual-branch (semantic + handcrafted), fusion + gate, per-stage shared projector
            
            attn_pool_stride = getattr(self, 'attn_pool_stride', 4)

            def make_branch_adapter(stage_idx):
                dim_low = self.embed_dims[stage_idx] // self.scale_factor
                if self.adapter_internal == 'mlp':
                    return GridMLPAdapter(dim_low)

                else:
                    raise ValueError(f"Unknown adapter_internal: {self.adapter_internal}")

            # determine enabled branches
            def get_enabled_branches():
                return {
                    'sem': self.embedding_tune,
                    'hand': self.handcrafted_tune
                }
            
            def create_stage_modules(stage_idx, embed_dim, depth, C_low):
                """
                Helper to create modules for a single stage.
                stage_idx: 0,1,2,3 for stages 1,2,3,4
                embed_dim: target embedding dimension for this stage
                depth: number of blocks in this stage
                C_low: low-dimensional feature size
                """

                enabled = get_enabled_branches()
                use_gating = getattr(self, 'use_gating', True)

                # calculate fusion input dimension based on enabled branches
                fusion_input_dim = 0
                if enabled['sem']:
                    fusion_input_dim += C_low
                if enabled['hand']:
                    fusion_input_dim += C_low
                
                # Edge case: both disabled, use C_low as fallback to avoid Linear(0, C_low)
                if fusion_input_dim == 0:
                    fusion_input_dim = C_low

                modules = {}
                for i in range(depth + 1):
                    # Create semantic adapter ONLY if enabled
                    if enabled['sem']:
                        modules[f'sem_adapter_{i}'] = make_branch_adapter(stage_idx)
                    
                    # Create handcrafted adapter ONLY if enabled
                    if enabled['hand']:
                        modules[f'hand_adapter_{i}'] = make_branch_adapter(stage_idx)
                    
                    # Fusion MLP with DYNAMIC input dimension
                    modules[f'fusion_mlp_{i}'] = nn.Sequential(
                        nn.Linear(fusion_input_dim, C_low),
                        nn.GELU()
                    )

                    # Gate MLP ONLY if gating is enabled
                    if use_gating:
                        gate_mlp = nn.Linear(C_low, C_low)
                        gate_mlp.bias.data.fill_(-1.5)  # Initialize bias to favor suppression
                        modules[f'gate_mlp_{i}'] = gate_mlp
                    
                    # Per-depth projection ONLY if not using shared projection
                    if not self.shared_projection:
                        modules[f'proj_mlp_{i}'] = nn.Linear(C_low, embed_dim)
                
                # Shared projection layer
                if self.shared_projection:
                    modules['shared_proj'] = nn.Linear(C_low, embed_dim)

                return modules
            
            # Stage 1
            if '1' in self.tuning_stage:
                C1_low = self.embed_dims[0] // self.scale_factor
                stage1_modules = create_stage_modules(
                    stage_idx=0, 
                    embed_dim=self.embed_dims[0], 
                    depth=self.depths[0], 
                    C_low=C1_low
                )

                # Register modules via setattr
                for name, module in stage1_modules.items():
                    if name == 'shared_proj':
                        self.shared_mlp1 = module
                    else:
                        parts = name.rsplit('_', 1)  # ['sem_adapter', '0']
                        final_name = f'{parts[0]}1_{parts[1]}'  # 'sem_adapter1_0'
                        setattr(self, final_name, module)
            
            # Stage 2
            if '2' in self.tuning_stage:
                C2_low = self.embed_dims[1] // self.scale_factor
                stage2_modules = create_stage_modules(
                    stage_idx=1, 
                    embed_dim=self.embed_dims[1], 
                    depth=self.depths[1], 
                    C_low=C2_low
                )
                for name, module in stage2_modules.items():
                    if name == 'shared_proj':
                        self.shared_mlp2 = module
                    else:
                        parts = name.rsplit('_', 1)  # ['sem_adapter', '0']
                        final_name = f'{parts[0]}2_{parts[1]}'  # 'sem_adapter2_0'
                        setattr(self, final_name, module)
            
            # Stage 3
            if '3' in self.tuning_stage:
                C3_low = self.embed_dims[2] // self.scale_factor
                stage3_modules = create_stage_modules(
                    stage_idx=2, 
                    embed_dim=self.embed_dims[2], 
                    depth=self.depths[2], 
                    C_low=C3_low
                )
                for name, module in stage3_modules.items():
                    if name == 'shared_proj':
                        self.shared_mlp3 = module
                    else:
                        parts = name.rsplit('_', 1)  # ['sem_adapter', '0']
                        final_name = f'{parts[0]}3_{parts[1]}'  # 'sem_adapter3_0'
                        setattr(self, final_name, module)
            
            # Stage 4
            if '4' in self.tuning_stage:
                C4_low = self.embed_dims[3] // self.scale_factor
                stage4_modules = create_stage_modules(
                    stage_idx=3, 
                    embed_dim=self.embed_dims[3], 
                    depth=self.depths[3], 
                    C_low=C4_low
                )
                for name, module in stage4_modules.items():
                    if name == 'shared_proj':
                        self.shared_mlp4 = module
                    else:
                        parts = name.rsplit('_', 1)  # ['sem_adapter', '0']
                        final_name = f'{parts[0]}4_{parts[1]}'  # 'sem_adapter4_0'
                        setattr(self, final_name, module)

        else:
            raise ValueError(f"Unknown adaptor mode: {self.adaptor}")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_handcrafted(self, x):
        """Return handcrafted features per stage, or None if branch disabled"""
        if not self.handcrafted_tune:
            return tuple(None for _ in range(4))

        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums, self.prompt_type)

        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        elif self.input_type == 'gaussian':
            x = self.gaussian_filter.conv_gauss(x)

        elif self.input_type == 'srm':
            x = self.srm_filter.srm_layer(x)
            
        elif self.input_type == 'wavelet':
            x = self.wavelet_filter(x)
            
        elif self.input_type == 'gabor':
            x = self.gabor_filter(x)
            x = self.gabor_reduce(x)

        B = x.shape[0]
        h_feat, H, W = None, None, None
        handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None

        if '1' in self.tuning_stage or any(s in self.tuning_stage for s in ['2', '3', '4']):
            h_feat, H, W = self.handcrafted_generator1(x)
            handcrafted1 = h_feat if '1' in self.tuning_stage else None
            # handcrafted1, H1, W1 = self.handcrafted_generator1(x)

        if '2' in self.tuning_stage or any(s in self.tuning_stage for s in ['3', '4']):
            if h_feat is not None:
                h_feat, H, W = self.handcrafted_generator2(h_feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
            handcrafted2 = h_feat if '2' in self.tuning_stage else None

        if '3' in self.tuning_stage or '4' in self.tuning_stage:
            if h_feat is not None:
                h_feat, H, W = self.handcrafted_generator3(h_feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
            handcrafted3 = h_feat if '3' in self.tuning_stage else None

        if '4' in self.tuning_stage:
            if h_feat is not None:
                h_feat, H, W = self.handcrafted_generator4(h_feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
            handcrafted4 = h_feat if '4' in self.tuning_stage else None

        return handcrafted1, handcrafted2, handcrafted3, handcrafted4

    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(embedding_feature)
        else:
            embedding_feature = None   # semantic branch disabled, pass None downstream

        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature
        else:
            handcrafted_feature = None  # handcrafted branch disabled, pass None downstream

        return handcrafted_feature, embedding_feature

    def get_embedding_feature(self, x, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
            embedding_feature = embedding_generator(x)

            return embedding_feature
        else:
            return None

    def get_handcrafted_feature(self, x, block_num):
        if self.handcrafted_tune:
            handcrafted_generator = getattr(self, 'handcrafted_generator{}'.format(str(block_num)))
            handcrafted_feature = handcrafted_generator(x)

            return handcrafted_feature
        else:
            return None
        

    def _store_viz(self, block_num, depth_num, sem_out, hand_out, fused, gate):
        """Store attention maps and gates for visualization."""
        if not hasattr(self, 'attn_maps'):
            self.attn_maps = {}
        if not hasattr(self, 'gates'):
            self.gates = {}
        
        key = f'stage{block_num}_depth{depth_num}'
        self.attn_maps[key] = {
            'semantic': sem_out.detach().cpu(),
            'handcrafted': hand_out.detach().cpu(),
            'fused': fused.detach().cpu()
        }
        self.gates[key] = gate.detach().cpu()


    def get_prompt(self, x, prompt, block_num, depth_num):
        """
            x:                 [B,H,W,C_stage]  (current tokens)
            prompt:            (handcrafted_low, embedding_low)
                               handcrafted_low: [B,H,W,C_low]
                               embedding_low:   [B,H,W,C_low]
            block_num:         1..4 (stage id)
            depth_num:         int (which block inside stage loop)
            returns: x + prompt_feat
        """
        hand_low, emb_low = prompt
        if (hand_low is None) and (emb_low is None):
            return x

        # Per-mode behavior
        if self.adaptor == 'adaptor':
            lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
            shared_mlp = getattr(self, 'shared_mlp' + str(block_num))

            B, H, W, _ = x.shape
            
            feat = 0
            if self.handcrafted_tune:
                if hand_low.dim() == 3:
                    hand_low = hand_low.view(B, H, W, -1)
                feat = feat + hand_low
            if self.embedding_tune:
                if emb_low.dim() == 3:
                    emb_low = emb_low.view(B, H, W, -1)
                feat = feat + emb_low

            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)

            return x + feat

        elif self.adaptor == 'gated_dual_branch':
            # Retrieve modules with safe getattr (returns None if not created)
            sem_ad = getattr(self, f'sem_adapter{block_num}_{depth_num}', None)
            hand_ad = getattr(self, f'hand_adapter{block_num}_{depth_num}', None)
            fuse_m = getattr(self, f'fusion_mlp{block_num}_{depth_num}')

            # Handle shared vs per-depth projection
            if hasattr(self, f'proj_mlp{block_num}_{depth_num}'):
                # Per-depth projection mode
                shared = getattr(self, f'proj_mlp{block_num}_{depth_num}')
            else:
                # Shared projection mode (default)
                shared = getattr(self, f'shared_mlp{block_num}')

            B, H, W, _ = x.shape
            stage_idx = block_num - 1  # Convert 1-based to 0-based
            C_low = self.embed_dims[stage_idx] // self.scale_factor

            # Reshape inputs if needed
            if emb_low is not None and emb_low.dim() == 3:
                emb_low = emb_low.view(B, H, W, -1)
            if hand_low is not None and hand_low.dim() == 3:
                hand_low = hand_low.view(B, H, W, -1)

            # Collect outputs from ENABLED branches only
            outputs_to_concat = []

            # Semantic branch: process only if adapter exists AND input exists AND branch enabled
            if (sem_ad is not None and emb_low is not None and getattr(self, 'embedding_tune', True)):
                sem_out = sem_ad(emb_low)  # [B,H,W,C_low]
                outputs_to_concat.append(sem_out)
            
            # Handcrafted branch
            if (hand_ad is not None and hand_low is not None and getattr(self, 'handcrafted_tune', True)):
                hand_out = hand_ad(hand_low)  # [B,H,W,C_low]
                outputs_to_concat.append(hand_out)
            
            # Edge case: both branches disabled --> return input unchanged (no adaptation)
            if len(outputs_to_concat) == 0:
                return x
            
            # Feature fusion
            fused = torch.cat(outputs_to_concat, dim=-1)
            fused = fuse_m(fused)  # [B,H,W,C_low]

            # Gating vs Direct addition
            use_gating = getattr(self, 'use_gating', True)
            if use_gating:
                # Gate MLP should exist if gating is enabled
                gate_m = getattr(self, f'gate_mlp{block_num}_{depth_num}', None)
                if gate_m is None:
                    raise RuntimeError(
                        f"Gating enabled but gate_mlp not found for stage{block_num}_depth{depth_num}. "
                        "Check that use_gating=True during module creation."
                    )
                gate = torch.sigmoid(gate_m(fused.mean(dim=(1, 2))))  # [B,C_low]
                fused = fused * gate[:, None, None, :]  # Channel-wise modulation
            
            # Visualization
            if getattr(self, 'enable_viz', False):
                self._store_viz(
                block_num, depth_num, 
                outputs_to_concat[0] if len(outputs_to_concat) > 0 and getattr(self, 'embedding_tune', True) else None,
                outputs_to_concat[1] if len(outputs_to_concat) > 1 and getattr(self, 'handcrafted_tune', True) else None,
                fused, 
                gate if use_gating else None
            )
                
            # Project and Inject
            feat = shared(fused)  # [B,H,W,C_stage]
            return x + feat  # Residual connection

        else:
            raise ValueError(f"Unknown adaptor mode: {self.adaptor}")
        

    def fft(self, x, rate, prompt_type):
        mask = torch.zeros(x.shape).to('cuda')
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        if prompt_type == 'highpass':
            fft = fft * (1 - mask)
        elif prompt_type == 'lowpass':
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv
        
        
class GaussianFilter(nn.Module):
    def __init__(self):
        super(GaussianFilter, self).__init__()
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def conv_gauss(self, img):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, self.kernel, groups=img.shape[1])
        return out
    
    
class SRMFilter(nn.Module):
    def __init__(self):
        super(SRMFilter, self).__init__()
        self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,)
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 2 / 4, -4 / 4, 2 / 4, 0],
                   [0, -1 / 4, 2 / 4, -1 / 4, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
                   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                   [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1 / 2, -2 / 2, 1 / 2, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        self.srm_layer.weight.data = torch.Tensor(
            [[filter1, filter1, filter1],
             [filter2, filter2, filter2],
             [filter3, filter3, filter3]]
        )

        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def conv_srm(self, img):
        out = self.srm_layer(img)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x