
from .trans_layers import LayerNorm
import pdb
from timm.models.layers import DropPath, Mlp
import math
from functools import partial
from typing import Tuple

import torch
from torch import nn, _assert, Tensor
import torch.nn.functional as F
from .davit import SpatialBlock,ChannelBlock 
from .deformable_conv import  DeformConv3d
__all__ = [
    'ConvNormAct',
    'BasicBlock',
    'Bottleneck',
    'DepthwiseSeparableConv',
    'MultiscaleBlock'
]
class Transdavitblock(nn.Module):
    def __init__(self, dim,):
        super().__init__()
        
        
        self.SBblock = SpatialBlock(
                        dim=dim,
                        num_heads=int(dim/32),
                        mlp_ratio=2,
                        qkv_bias=False,
                        drop_path=0.,
                        norm_layer=nn.LayerNorm,
                        ffn=True,
                        cpe_act=False,
                        window_size=7,)   

        self.CBblock = ChannelBlock(
                    dim=dim,
                    num_heads=int(dim/32),
                    mlp_ratio=2,
                    qkv_bias=False,
                    drop_path=0.,
                    norm_layer=nn.LayerNorm,
                    ffn=True,
                    cpe_act=False,)   
    def forward(self, x):
        B, C, D,H,W = x.shape
        size = (x.size(2), x.size(3),x.size(4))
        # print(sizec)
        x = x.flatten(2).transpose(1, 2)#B,N,C
        # print(x.shape)
        x,size = self.SBblock (x,size) #B,N,C
        x,size = self.CBblock( x,size)
        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        # print(x.shape)
        return x

class seriesconv(nn.Module):
    def __init__(self,in_chan, out_chan, 
         norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        
        interchannel = int(out_chan//3)
        self.conv0 = ConvNormAct(in_chan,  interchannel, kernel_size=3, stride=1, padding=1, norm=norm, act=act, preact=preact)  # preact = True
        self.conv0 = ConvNormAct(in_chan,  interchannel, kernel_size=3, stride=1, padding=1, norm=norm, act=act, preact=preact)  # preact = True

        self.conv0 = ConvNormAct(in_chan,  interchannel, kernel_size=3, stride=1, padding=1, norm=norm, act=act, preact=preact)  # preact = True

        self.conv1 = nn.Conv3d( interchannel,  interchannel, kernel_size = 5,  padding=2)
        self.conv2 = nn.Conv3d( interchannel,  interchannel, kernel_size = 7, padding=3)
        self.conv3 = nn.Conv3d( int(interchannel*3),  out_chan, kernel_size = 1, padding=0) #对齐channel
        self.shortcut = nn.Sequential()

        if in_chan != out_chan:
            # print(555)
            # self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
            self.shortcut = ConvNormAct(in_chan, out_chan, 1,  stride=1, padding=0,norm=norm, act=act, preact=preact) # attentionUnet 要改回去

    def forward(self, x):
        shortcut = x
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0) 
        conv2 = self.conv2(conv1) 
        out = torch.cat([conv0,conv1,conv2], dim=1)
        out = self.conv3(out) 
        out += self.shortcut(shortcut)
        return out






        # attn = self.rnnlayer(attn) 







class MultiscaleConv(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        # self.conv0 = ConvNormAct(dim, dim, kernel_size, stride=stride, padding=padding, norm=norm, act=act, preact=preact) #3，2，1

        # self.conv0 = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        # self.conv0_1 = nn.Conv3d(dim, dim, 7, padding=3)

        # self.conv0_1 = nn.Conv3d(dim, dim, (7,1,1), padding=(3,0,0), groups=dim)
        # self.conv0_2 = nn.Conv3d(dim, dim, (1,7,1), padding=(0,3,0), groups=dim)
        # self.conv0_3 = nn.Conv3d(dim, dim, (1,1,7), padding=(0,0,3), groups=dim)
        # # self.conv1_1 = nn.Conv3d(dim, dim, 5, padding=2)

        # self.conv1_1 = nn.Conv3d(dim, dim, (5,1,1), padding=(2,0,0), groups=dim)
        # self.conv1_2 = nn.Conv3d(dim, dim, (1,5,1), padding=(0,2,0), groups=dim) #换成空洞卷积ASPP
        # self.conv1_3 = nn.Conv3d(dim, dim, (1,1,5), padding=(0,0,2), groups=dim)
        # self.conv2_1 = nn.Conv3d(dim, dim, 3, padding=1)

        self.conv2_1 = nn.Conv3d(dim, dim, (3,3,1), padding=(1,1,0), groups=dim)
        self.conv2_2 = nn.Conv3d(dim, dim, (3,1,3), padding=(1,0,1), groups=dim)
        self.conv2_3 = nn.Conv3d(dim, dim, (1,3,3), padding=(0,1,1), groups=dim)

        # self.rnnlayer = Sequencer3DBlock(dim, int(dim/2), mlp_ratio=3.0, rnn_layer=LSTM3D, mlp_layer=Mlp,
        #                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
        #                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
        #                 drop=0., drop_path=0.) #输入输出同channel

        # self.conv0 = ConvNormAct(dim, dim, kernel_size, stride=stride, padding=padding, norm=norm, act=act, preact=preact) #3，2，1

        # self.rnnlayer = Transdavitblock(dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)
        # self.conv3 = nn.Conv3d(dim, out_ch, 1)
        # self.sk = SKBlock(dim)
        # print('555')

    def forward(self, x):
        shortcut = x
        # print(x.shape)
        # attn = self.conv0(x)
        # attn = self.rnnlayer(x) 
        # attn = self.rnnlayer(attn) 
        # print(555)


        # attn_0 = self.conv0_1(x)
        # attn_0 = self.conv0_2(attn_0)
        # attn_0 = self.conv0_3(attn_0)


        # attn_1 = self.conv1_1(x)
        # attn_1 = self.conv1_2(attn_1)
        # attn_1 = self.conv1_3(attn_1)


        # attn_2 = self.conv2_1(x)
        # attn_2 = self.conv2_2(attn_2)
        # attn_2 = self.conv2_3(attn_2)
        

        attn_1 = self.conv2_1(x)
        attn_2 = self.conv2_2(x)
        attn_3 = self.conv2_3(x)


        # atten_3 = self.rnnlayer(x) 
        '''sk block'''

        attn = attn_3 + attn_1 + attn_2
        # attn = self.sk(attn,attn_0,attn_1,attn_2)

        # attn = attn + attn_0 + attn_1 + attn_2+atten_3 #最初用的这个
        # attn = self.conv3(atten_3+shortcut) #lstm only
        attn = self.conv3(attn) #lstm only


        # out = self.conv3(attn * shortcut)
        return attn

class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN as IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x): 
    
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out 

        

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):

        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True): #之前为True，False for oocs block
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        # self.sse = sSEModule(out_ch)
        # self.se = SEBlock(out_ch)
        # self.BD = BridgeBlock(out_ch)
        # self.eca = eca_block(out_ch)
        # self.A1 = DoubleAttentionLayer(in_ch, out_ch,out_ch,False)
        # self.A2 = DoubleAttentionLayer(out_ch, out_ch,out_ch,True)
        # self.conv3 = ConvNormAct(out_ch, out_ch, 1,  stride=1, padding=0,norm=norm, act=act, preact=preact) #try

   
        if stride != 1 or in_ch != out_ch:

            # self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
            self.shortcut = ConvNormAct(in_ch, out_ch, 1,  stride=1, padding=0,norm=norm, act=act, preact=preact) #try

    def forward(self, x):
        residual = x
        # print(self.short)
        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.se(out)
        # out += self.shortcut(residual)

        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        # out = self.BD(out2,out1)
        # out += self.shortcut(residual)
        
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.eca(out)
        # out = self.sse(out)

        # out = self.A1(x)
        # out = self.A2(out)
        # self.conv3(out)
        out += self.shortcut(residual)


        return out

class MultiscaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]
        self.conv0 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        # self.conv0 = ConvNormAct(in_ch, out_ch, 1,  stride=1, padding=0,norm=norm, act=act, preact=preact) # attentionUnet 要改回去
        self.conv1 = ConvNormAct(out_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        # self.conv3 = nn.Conv3d( out_ch*3,  out_ch, kernel_size = 1,  padding=0)
        # self.fusion_norm = norm(out_ch)
        # self.conv2 = seriesconv(out_ch, out_ch, norm=norm, act=act, preact=preact)
        # self.conv1  = DeformConv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2  = DeformConv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.eca = eca_block(out_ch)
        # self.sse = sSEModule(out_ch*3)
        self.sse = sSEModule(out_ch)

        # self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)
        # self.conv3 = MultiscaleConv(out_ch,kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)#size 不变
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            # print(555)
            # self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
            self.shortcut = ConvNormAct(in_ch, out_ch, 1,  stride=1, padding=0,norm=norm, act=act, preact=preact) # attentionUnet 要改回去

    def forward(self, x):
        residual = x

        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        # out10 = torch.cat([out1,out0], dim=1)

        out2 = self.conv2(out1)
        # out = torch.cat([out2,out1,out0], dim=1)
        # out = self.conv3(out)

        # norm1 = self.fusion_norm(out0)
        # norm2 = self.fusion_norm(out1)
        # norm3 = self.fusion_norm(out2)
        # out = norm1 + norm2 +norm3
        out = out0 + out1 + out2

        # eca_attn = self.eca(out)
        attn = self.sse(out,out2)
        # attn = eca_attn+sse_attn
        # attn  = self.conv3(attn)

        # out = self.conv2(out)

        # attn += self.shortcut(residual)
        # out = self.conv3(out)

        return attn+out0

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, groups=1, dilation=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.expansion = 2
        self.conv1 = ConvNormAct(in_ch, out_ch//self.expansion, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//self.expansion, out_ch//self.expansion, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch//self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, list):
            padding = [i//2 for i in kernel_size]
        else:
            padding = kernel_size // 2

        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out
    

class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
                        nn.Conv3d(in_ch, in_ch//ratio, kernel_size=1),
                        act(),
                        nn.Conv3d(in_ch//ratio, in_ch, kernel_size=1),
                        nn.Sigmoid()
        )
    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out
    
class sSEModule(nn.Module):
    """
    Channel squeeze & spatial excitation attention module, as proposed in https://arxiv.org/abs/1808.08127.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x,out):
        return out * self.sSE(x)
    
class SCSEModule(nn.Module):
    """
    Concurrent spatial and channel squeeze & excitation attention module, as proposed in https://arxiv.org/pdf/1803.02579.pdf.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
class BridgeBlock(nn.Module):
    def __init__(self, in_ch,  reduction=4,norm = nn.BatchNorm1d, act=nn.ReLU):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.squeeze = nn.Sequential(
                nn.Linear(in_ch, in_ch//reduction, bias=False),
                # nn.Conv3d(in_ch, in_ch//reduction, kernel_size=1),
                norm(in_ch//reduction)
            )
     
        self.excitation = nn.Sequential(
                        act(inplace=True),
                        nn.Linear(in_ch//reduction,in_ch,  bias=False),

                        # nn.Conv3d(in_ch//reduction, in_ch, kernel_size=1),
                        nn.Sigmoid()
        )
    def forward(self, currentlayer, previouslayer):
        currentlayer_pool = self.pool(currentlayer)
        previouslayer_pool = self.pool(previouslayer)
        b, ch, _, _, _ = currentlayer_pool.size()
        layer1 = previouslayer_pool.view(b, -1).contiguous()
        layer2 = currentlayer_pool.view(b, -1).contiguous()

        out1 = self.squeeze(layer1)
        out2 = self.squeeze(layer2)


        attn = self.excitation(out1+out2).view(b, ch, 1, 1, 1)

        return currentlayer * attn
    
class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct = False):
        """

        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv3d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv3d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv3d(in_channels, c_n, kernel_size = 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, h, w, d = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.reshape(batch_size, self.c_m, h * w * d)
        tmpA = A.reshape(batch_size, self.c_m, h * w * d)
        attention_maps = B.reshape(batch_size, self.c_n, h * w * d)
        attention_vectors = V.reshape(batch_size, self.c_n, h * w * d)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.reshape(batch_size, self.c_m, h, w, d)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return self.norm(tmpZ)
 
# 定义ECANet的类
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        super(eca_block, self).__init__()
        
        # 根据输入通道数自适应调整卷积核大小
        # print(in_channel)
        kernel_size = int(abs((math.log(in_channel, 2)+b)/gama))
        # print(kernel_size )
       
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size-1
        # print(kernel_size)
        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2
        
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w, d = inputs.shape
        # 全局平均池化 [b,c,h,w,d==>[b,c,1,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1,1]==>[b,1,c]
        x = x.reshape([b,1,c])
        # 1D卷积 [b,1,c]==>[b,1,c] # 在这里加上上层的
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1,1]
        x = x.reshape([b,c,1,1,1])
        
        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs

class SKBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        reduction_factor = 4
        self.radix = 4
        attn_chs = max(in_ch * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
           
            nn.Conv3d(in_ch, attn_chs, 1),
            # nn.BatchNorm3d(attn_chs),
            # nn.ReLU(inplace=True),

            nn.Conv3d(attn_chs, self.radix*in_ch, 1)
        )
       
    def forward(self, x,w,y,z): #radix = 3
        B, C, H, W, D = x.shape
        w = w.view(B, C, 1, H, W, D)
        y = y.view(B, C, 1, H, W, D)
        z = z.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, w, y, z], dim=2) 
        # print(x.shape) #[1, 64, 2, 96, 96, 96]
        x_gap = x.sum(dim=2)
        # print('sk')
        # print(x_gap.shape) #[1, 64, 96, 96, 96]
        x_gap = x_gap.mean((2, 3, 4), keepdim=True) # average pool
        # print(x_gap.shape) #[1, 64, 1, 1, 1]
        x_attn = self.se(x_gap) # squeeze and excite channel
        # print(x_attn.shape) #[1, 128, 1, 1, 1]
        x_attn = x_attn.view(B, C, self.radix) 
        # print(x_attn.shape) #[1, 64, 2]

        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out

# class DropPath(nn.Module):
    
#     def __init__(self, p=0):
#         super().__init__()
#         self.p = p

#     def forward(self, x):
#         if (not self.p) or (not self.training):
#             return x

#         batch_size = x.shape[0]
#         random_tensor = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)
#         binary_mask = self.p < random_tensor

#         x = x.div(1 - self.p)
#         x = x * binary_mask

#         return x


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]
        expanded = expansion * in_ch
        self.se = se

        self.expand_proj = nn.Identity() if (expansion==1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0, norm=norm, act=act, preact=True)

        self.depthwise = ConvNormAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=expanded, act=act, norm=norm, preact=True)

        if self.se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))


    def forward(self, x):
        residual = x

        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x += self.shortcut(residual)

        return x


class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size -1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]

        expanded = expansion * in_ch

        self.stride= stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x

