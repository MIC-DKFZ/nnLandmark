"""
author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# import configs_TransMorph as configs
import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib


import itertools
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np

from typing import Tuple
from collections import OrderedDict
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper

from .BiFormer.bra_legacy import BiLevelRoutingAttention
from .BiFormer._common import Attention, AttentionLePE, DWConv

import ml_collections
from torchinfo import summary

        
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path = 0., layer_scale_init_value = -1,
                 num_heads = 8, n_win = (4,4,2), qk_dim = None, qk_scale = None,
                 kv_per_win = 4, kv_downsample_ratio = 4, kv_downsample_kernel = None,
                 kv_downsample_mode = 'ada_avgpool',
                 topk = 4, param_attention = "qkvo", param_routing = False, diff_routing = False, soft_routing = False,
                 mlp_ratio = 4, mlp_dwconv = False,
                 side_dwconv = 5, before_attn_dwconv = 3, pre_norm = True, auto_pad = False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv3d(dim, dim, kernel_size = before_attn_dwconv, padding = 1, groups = dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim = dim, num_heads = num_heads, n_win = n_win, qk_dim = qk_dim,
                                                qk_scale = qk_scale, kv_per_win = kv_per_win,
                                                kv_downsample_ratio = kv_downsample_ratio,
                                                kv_downsample_kernel = kv_downsample_kernel,
                                                kv_downsample_mode = kv_downsample_mode,
                                                topk = topk, param_attention = param_attention,
                                                param_routing = param_routing,
                                                diff_routing = diff_routing, soft_routing = soft_routing,
                                                side_dwconv = side_dwconv,
                                                auto_pad = auto_pad)
        elif topk == -1:
            self.attn = Attention(dim = dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim = dim, side_dwconv = side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w d c -> n c h d w'),  # compatiability
                                      nn.Conv3d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv3d(dim, dim, 5, padding = 2, groups = dim),  # pseudo attention
                                      nn.Conv3d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w d -> n h w d c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)
        return x


#  config = CONFIGS['BiFormer'],
class BiFormer_Unet(nn.Module):
    # def __init__(self, n_class = 14, depth = [2, 2, 4, 2], in_chans = 1, embed_dim = [96, 192, 384, 768],
    #              head_dim = 8, qk_scale = None, representation_size = None,
    #              drop_path_rate = 0., drop_rate = 0.,
    #              use_checkpoint_stages = [],
    #              ########
    #              n_win = 5,
    #              kv_downsample_mode = 'identity',
    #              kv_per_wins = [-1, -1, -1, -1],
    #              topks = [1, 4, 16, 16],
    #              side_dwconv = 5,
    #              layer_scale_init_value = -1,
    #              qk_dims = [96, 192, 384, 768],
    #              param_routing = False, diff_routing = False, soft_routing = False,
    #              pre_norm = True,
    #              pe = None,
    #              pe_stages = [0],
    #              before_attn_dwconv = 3,
    #              auto_pad = True,
    #              kv_downsample_kernels = [4, 2, 1, 1],
    #              kv_downsample_ratios = [4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
    #              mlp_ratios = [4, 4, 4, 4],
    #              param_attention = 'qkvo',
    #              out_indices = (0, 1, 2, 3),
    #              mlp_dwconv = False):
    #[32, 64, 128, 256] embed_dim/head_dim=head个数
    def __init__(self, n_class = 14, depth = [2, 2, 4, 2], in_chans = 1, embed_dim = [64, 128, 256, 512],qk_dims = [64, 128, 256, 512],
                 head_dim = 8, qk_scale = None, representation_size = None,
                 drop_path_rate = 0., drop_rate = 0.,
                 use_checkpoint_stages = [],
                 ########
                #  n_win = 5,
                #  n_win = 4,
                 n_win = (4, 4, 2),
                 side_dwconv = 5,
                 kv_downsample_mode = 'identity',
                 kv_per_wins = [-1, -1, -1, -1],
                 topks = [1, 4, 16, 16],
                 layer_scale_init_value = -1,
                 param_routing = False, diff_routing = False, soft_routing = False,
                 pre_norm = True,
                 pe = None,
                 pe_stages = [0],
                 before_attn_dwconv = 3,
                 auto_pad = True,
                 kv_downsample_kernels = [4, 2, 1, 1],
                 kv_downsample_ratios = [4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios = [4, 4, 4, 4],
                 param_attention = 'qkvo',
                 out_indices = (0, 1, 2, 3),
                 mlp_dwconv = False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super(BiFormer_Unet, self).__init__()
        self.if_convskip = True
        self.if_transskip = True
        self.n_class = n_class
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        # 试试down4但是不加卷积
        #down4
        stem = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim[0] // 2, kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
            nn.InstanceNorm3d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim[0] // 2, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
            nn.InstanceNorm3d(embed_dim[0]),
        )
        # DOWN2
        # stem = nn.Sequential(
        #     nn.Conv3d(in_chans, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
        #     nn.InstanceNorm3d(embed_dim[0]),
        # )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim = embed_dim[0], name = pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)
        # patch embedding
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
                nn.InstanceNorm3d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim = embed_dim[i + 1], name = pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]#[4,8,16,32]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim = embed_dim[i], drop_path = dp_rates[cur + j],
                        layer_scale_init_value = layer_scale_init_value,
                        topk = topks[i],
                        num_heads = nheads[i],
                        n_win = n_win,
                        qk_dim = qk_dims[i],
                        qk_scale = qk_scale,
                        kv_per_win = kv_per_wins[i],
                        kv_downsample_ratio = kv_downsample_ratios[i],
                        kv_downsample_kernel = kv_downsample_kernels[i],
                        kv_downsample_mode = kv_downsample_mode,
                        param_attention = param_attention,
                        param_routing = param_routing,
                        diff_routing = diff_routing,
                        soft_routing = soft_routing,
                        mlp_ratio = mlp_ratios[i],
                        mlp_dwconv = mlp_dwconv,
                        side_dwconv = side_dwconv,
                        before_attn_dwconv = before_attn_dwconv,
                        pre_norm = pre_norm,
                        auto_pad = auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        # out_indices = [0, 1, 2, 3]
        for i_layer in out_indices:
            layer = nn.LayerNorm(embed_dim[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()


        self.head = nn.Linear(embed_dim[-1], n_class) if n_class > 0 else nn.Identity()
        self.apply(self._init_weights)
        # 不加卷积
        # self.up0 = DecoderBlock(embed_dim[3], embed_dim[2] , skip_channels = embed_dim[2] if self.if_transskip else 0,
        #                         use_batchnorm = False)
        # self.up1 = DecoderBlock(embed_dim[2], embed_dim[1], skip_channels = embed_dim[1]if self.if_transskip else 0,
        #                         use_batchnorm = False)  # 384, 20, 20, 64
        # self.up2 = DecoderBlock(embed_dim[1] , embed_dim[0], skip_channels = embed_dim[0] if self.if_transskip else 0,
        #                         use_batchnorm = False)  # 384, 40, 40, 64
        # self.up3 = DecoderBlock(embed_dim[0], 16, skip_channels = 16 if self.if_convskip else 0,
        #                         use_batchnorm = False)  # 384, 80, 80, 128   
             
        #未改
        self.up0 = DecoderBlock(embed_dim[3], embed_dim[2] , skip_channels = embed_dim[2] if self.if_transskip else 0,
                                use_batchnorm = False)
        self.up1 = DecoderBlock(embed_dim[2], embed_dim[1], skip_channels = embed_dim[1]if self.if_transskip else 0,
                                use_batchnorm = False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim[1] , embed_dim[0], skip_channels = embed_dim[0] if self.if_transskip else 0,
                                use_batchnorm = False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim[0], embed_dim[0] // 2, skip_channels = embed_dim[0] // 2 if self.if_convskip else 0,
                                use_batchnorm = False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim[0] // 2, 16,
                                skip_channels = 16 if self.if_convskip else 0,
                                use_batchnorm = False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(1, embed_dim[0] // 2, 3, 1, use_batchnorm = False)
        self.c2 = Conv3dReLU(1, 16, 3, 1, use_batchnorm = False)

        self.final_conv = nn.Conv3d(16, self.n_class, 1)

        # self.reg_head = RegistrationHead(
        #     in_channels = 16,
        #     out_channels = 3,
        #     kernel_size = 3,
        # )
        # self.spatial_trans = SpatialTransformer(config.img_size)

        self.avg_pool = nn.AvgPool3d(3, stride = 2, padding = 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def down_features(self, x):
        outs = []
        for i in range(4):
            # print(x.size(),"111111111111111111")#torch.Size([1, 1, 128, 128, 64]) 111111111111111111
            x = self.downsample_layers[i](x)
            # print(x.size(),"22222222222222222")#torch.Size([1, 64, 32, 32, 16]) 22222222222222222
            x = self.stages[i](x)
            # print(x.size(),"33333333333333")
            x = x.permute(0,2,3,4,1)
            norm_layer = getattr(self, f'norm{i}')
            x = norm_layer(x)
            x = x.permute(0,4,1,2,3)
            # print(x.size(),"4444444444444")
            outs.append(x)
        return outs

    def forward(self, x):
        """
        Expect input x with layout (B, C, Z, X, Y).
        Internally we permute to (B, C, X, Y, Z) so Conv3d/Pool operate on (D=X, H=Y, W=Z).
        At the end we permute the model output back to (B, C_out, Z, X, Y).
        """
        # convert layout from (B, C, Z, X, Y) -> (B, C, X, Y, Z)
        # original axes: 0=B,1=C,2=Z,3=X,4=Y
        # target axes:   0=B,1=C,2=X,3=Y,4=Z  -> permute(0,1,3,4,2)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        moving = x[:, 0:1, :, :]  # [2, 1, 160, 192, 224]
        # print(x.size(),"--------------0")
        x_s0 = x.clone()  # [2, 2, 160, 192, 224]
        x_s1 = self.avg_pool(x)  # [2, 2, 80, 96, 112]
        f4 = self.c1(x_s1)  # [2, 48, 80, 96, 112]
        f5 = self.c2(x_s0)  # [2, 16, 160, 192, 224]
        # print(x.size(),"--------------1")

        out_feats = self.down_features(x)
        f1 = out_feats[-2]  # [2, 384, 10, 12, 14]
        f2 = out_feats[-3]  # [2, 192, 20, 24, 28]
        f3 = out_feats[-4]  # [2, 96, 40, 48, 56]

        x = self.up0(out_feats[-1], f1)  # [2, 384, 10, 12, 14]
        x = self.up1(x, f2)  # [2, 192, 20, 24, 28]
        x = self.up2(x, f3)  # [2, 96, 40, 48, 56]
        # 不用卷积
        # print(x.size(),"----------x")#torch.Size([1, 64, 40, 40, 40]) ----------x
        # print(f4.size(),"----------f4")#torch.Size([1, 32, 80, 80, 80]) ----------f4
        # print(f5.size(),"----------f5")#torch.Size([1, 16, 160, 160, 160]) ----------f5
        # x = self.up3(x, f5)  # [2, 48, 80, 96, 112]
        # 用卷积
        x = self.up3(x, f4)  # [2, 48, 80, 96, 112]
        x = self.up4(x, f5)  # [2, 16, 160, 192, 224]

        out = self.final_conv(x)
        # flow = self.reg_head(x)  # [2, 3, 160, 192, 224]
        # out = self.spatial_trans(moving, flow)  # [2, 1, 160, 192, 224]
        # return out, flow
        
        # Convert output back to original layout: (B, C_out, X, Y, Z) -> (B, C_out, Z, X, Y)
        # permute indices: (0,1,2,3,4) -> (0,1,4,2,3)
        out = out.permute(0, 1, 4, 2, 3).contiguous()

        return out


if __name__ == '__main__':
    # config = CONFIGS['BiFormer']
    model = BiFormer_Unet()
    summary(model, (1,1, 128, 128, 64), depth=3)
    # summary(model, (1,1, 160, 160, 160), depth=3)
    #out [1, 14, 160, 160, 160]







