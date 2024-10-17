#modified from Scale-Aware Modulation Meet Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_,to_3tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from typing import Optional, Sequence, Tuple, Type, Union
import os

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W ,D):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W, D))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    #dim=60, ca_num_heads=3. ca_num_heads=-1, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0, proj_drop=0,ca_attention=1
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups=self.dim//ca_num_heads
        # print(self.ca_attention)
        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv3d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv3d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.InstanceNorm3d(dim*expand_ratio)
            self.proj1 = nn.Conv3d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.dw_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        
    def forward(self, x, H, W, D):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, D, self.ca_num_heads, C//self.ca_num_heads).permute(4, 0, 5, 1, 2, 3)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W, D)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i],2)
            s_out = s_out.reshape(B, C, H, W,D)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v

        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.dw_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B,C, H, W,D)).view(B, C, N).transpose(1, 2)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    #dim=60, ca_num_heads=3,sa_num_heads=-1.mlp_ratio=2,qkv_bias=true, qk_scale=None, use_layerscale=False, layerscale_value1e-4, drop_path=0
    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1,expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #dim=60, ca_num_heads=3. ca_num_heads=-1, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0, proj_drop=0,ca_attention=1
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W, D):

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W, D))

        return x



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=96, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        img_size = to_3tuple(img_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2,patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W ,D


class Head(nn.Module):
    def __init__(self, in_chans,head_conv, dim):
        super(Head, self).__init__()

        stem = [nn.Conv3d(in_chans, int(dim/2), head_conv, 2,  padding=1, bias=False), nn.InstanceNorm3d(int(dim/2)), nn.ReLU(True)]
        stem.append(nn.Conv3d(int(dim/2), dim, kernel_size=1, stride=1,padding=0, groups=int(dim/2)))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W,D


#  embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
#                 qkv_bias=True, depths=[2, 2, 2, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2).to(device)



class UnetBlock_Encode_BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_BottleNeck, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            
        )

        self.conv2_2 = nn.Sequential(
            # nn.AvgPool3d(kernel_size=4, stride=2),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0, groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            
        )

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x
    

class UnetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Up, self).__init__()
        self.conv = self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=1,
                      padding=0, groups=out_channel),
            nn.BatchNorm3d(out_channel),
            nn.ReLU6(inplace=True),
        )
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
    
class SMT(nn.Module):
    def __init__(self, img_size=96, in_chans=1, num_classes=14, embed_dims=[60, 120, 240, 480],
                 ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2,2,2,2], 
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_name: Union[Tuple, str] = "instance", depths=[2, 2, 2, 2], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(in_chans,head_conv, embed_dims[i])#
            else:
                patch_embed = OverlapPatchEmbed(img_size = img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                #dim=60, ca_num_heads=3,sa_num_heads=-1.mlp_ratio=2,qkv_bias=true, qk_scale=None, use_layerscale=False, layerscale_value1e-4, drop_path=0
                dim=embed_dims[i], ca_num_heads=ca_num_heads[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                ca_attention=0 if i==2 and j%2!=0 else ca_attentions[i], expand_ratio=expand_ratio)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


        
        self.Encode_BottleNeck_block5 = UnetBlock_Encode_BottleNeck(embed_dims[3], embed_dims[3])



        self.encoder1_bottleneck = UnetBlock_Encode_BottleNeck(2*embed_dims[0], embed_dims[0])
        self.encoder2_bottleneck = UnetBlock_Encode_BottleNeck(2*embed_dims[1], embed_dims[1])
        self.encoder3_bottleneck = UnetBlock_Encode_BottleNeck(2*embed_dims[2], embed_dims[2])

        
        self.up1 = UnetBlock_Up(embed_dims[3], embed_dims[2])
        self.up2 = UnetBlock_Up(embed_dims[2], embed_dims[1])
        self.up3 = UnetBlock_Up(embed_dims[1], embed_dims[0])

        self.outup = nn.ConvTranspose3d(
                    embed_dims[0] , int(embed_dims[0]/2), kernel_size=2, stride=2,)

        self.out = UnetOutBlock(spatial_dims=3, in_channels=int(embed_dims[0]/2), out_channels=num_classes)

    def forward(self, x):
        origin_input = x
        B = x.shape[0]
        outs=[]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W, D = patch_embed(x)
            
            for blk in block:
                x = blk(x, H, W, D)
            x = norm(x)
            x = x.reshape(B, H, W,D, -1).permute(0, 4, 1, 2,3).contiguous()

            outs.append(x)

        x5 = self.Encode_BottleNeck_block5(outs[3])
        x_out = outs[3] + x5

        x_out = self.up1(x_out)       
        x_out = torch.cat((x_out, outs[2]), dim=1)
        x_out = self.encoder3_bottleneck(x_out)


        x_out = self.up2(x_out)
        x_out = torch.cat((x_out, outs[1]), dim=1)
        x_out = self.encoder2_bottleneck(x_out)

        x_out = self.up3(x_out)
        x_out = torch.cat((x_out, outs[0]), dim=1)
        x_out = self.encoder1_bottleneck(x_out)

        x_out = self.out(self.outup(x_out))
        return x_out
        # return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W,D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


if __name__ == '__main__':
    import time
    from thop import profile
    # device = torch.device("cuda:1")
    device = torch.device("cpu")

    model = SMT(
                embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
                qkv_bias=True, depths=[2, 2, 2, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2).to(device)

    input = torch.rand(2,1,96,96,96).to(device)
    flops, params = profile(model, inputs=(input, ))

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    start1= time.time()
    
    for i in range(101):
        start = time.time()
        input = torch.rand(2,1,96,96,96).to(device)
       
        out = model(input)
        end = time.time()
        print(end-start)
    print(time.time()-start1)
   