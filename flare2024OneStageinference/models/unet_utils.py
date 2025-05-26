import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct
import pdb 
from .OOCS_3dKernels import On_Off_Center_filters_3D


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class OOCS_Block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        # block_list = []
        #in_ch == out_ch
        # radius=1.0, gamma=1. / 2;  kernel =3;  2.0, gamma=2. / 3,kernel =5
        self.conv_On_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=in_ch, 
                                                    out_channels=out_ch, off=False)#.to(self.device)
        self.conv_Off_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=in_ch, 
                                                    out_channels=out_ch, off=True)#.to(self.device)
       

        if pool:
            self.downsample = nn.MaxPool3d(down_scale)
            # block_list.append(nn.MaxPool3d(down_scale))

            self.block1 = block(in_ch, out_ch, kernel_size=kernel_size, norm=norm)
        else:
            self.block1 = block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm)

        
        self.block2 = block(out_ch, out_ch,  kernel_size=kernel_size, norm=norm)
        self.block3 = block(out_ch, out_ch,  kernel_size=kernel_size, norm=norm)


        # self.conv = nn.Sequential(*block_list)
    def forward(self, x):
        sm_on = self.surround_modulation_DoG_on(x)
        sm_off = self.surround_modulation_DoG_off(x)
        x = self.downsample(x)

         # On and Off surround modulation
        # print(x.shape)
        # sm_on = self.surround_modulation_DoG_on(x) + x
        # sm_off = self.surround_modulation_DoG_off(x) + x

        output1 = self.block1(x)
        # output2 = self.block1(x)
       
        input3 = sm_on + output1
        input4 = sm_off + output1

        output3 = self.block2(input3)
        output4 = self.block3(input4)

        outputs = torch.cat((output3, output4), 1)
        return outputs

        # return self.conv(x)
    def surround_modulation_DoG_on(self, input):
        # i_norm = F.instance_norm(input)
        # re_act = F.relu(i_norm, inplace=True)
        # print( re_act.shape)
        # print( self.conv_On_filters.shape)
        device_id = torch.device(input.device)
        output = F.conv3d(input, weight=self.conv_On_filters.to(device_id), stride=2, padding=1) # padding = 1 for kernel =3        # print(output.shape)
        return F.relu(output, inplace=True) #output

    def surround_modulation_DoG_off(self, input):
        # i_norm = F.instance_norm(input)
        # re_act = F.relu(i_norm, inplace=True)
        device_id = torch.device(input.device)
        output = F.conv3d(input, weight=self.conv_Off_filters.to(device_id), stride=2, padding=1) # padding = 1 for kernel =3, padding = 2 for kernel =5
        return F.relu(output, inplace=True) #output



class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        block_list = []

        if pool:
            block_list.append(nn.MaxPool3d(down_scale))
            block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))
        else:
            
            # block_list.append(nn.Conv3d(
            # in_channels=in_ch, 
            # out_channels=out_ch, 
            # kernel_size=3,
            # stride=1,
            # padding=1,))

            block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):

            block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.up_scale = up_scale


        block_list = []

        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))
        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=True)

        out = torch.cat([x2, x1], dim=1)

        out = self.conv(out)

        return out


class OOCS_up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.up_scale = up_scale

        self.conv_On_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=in_ch+out_ch, 
                                                    out_channels=in_ch+out_ch, off=False)#.to(self.device)
        self.conv_Off_filters = On_Off_Center_filters_3D(radius=1.0, gamma=1. / 2., in_channels=in_ch+out_ch, 
                                                    out_channels=in_ch+out_ch, off=True)#.to(self.device)

        self.block1 = block(in_ch+out_ch, in_ch+out_ch, kernel_size=kernel_size, norm=norm)
        # for i in range(num_block-1):
        self.block2 = block(in_ch+out_ch, out_ch//2, kernel_size=kernel_size, norm=norm)
        self.block3 = block(in_ch+out_ch, out_ch//2, kernel_size=kernel_size, norm=norm)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)

        # out = self.conv(out)

        sm_on = self.surround_modulation_DoG_on(x) + x
        sm_off = self.surround_modulation_DoG_off(x) + x

        output1 = self.block1(x)
        # output2 = self.block1(x)
       
        input3 = sm_on + output1
        input4 = sm_off + output1

        output3 = self.block2(input3)
        output4 = self.block3(input4)

        out = torch.cat((output3, output4), 1)

        return out

    
        # return self.conv(x)
    def surround_modulation_DoG_on(self, input):
        i_norm = F.instance_norm(input)
        re_act = F.relu(i_norm, inplace=True)
        # print( re_act.shape)
        # print( self.conv_On_filters.shape)
        device_id = re_act.device
        device_id = torch.device(re_act.device)
        output = F.conv3d(re_act, weight=self.conv_On_filters.to(device_id), stride=1, padding=1) # padding = 1 for kernel =3        # print(output.shape)
        return output#F.relu(output, inplace=True)

    def surround_modulation_DoG_off(self, input):
        i_norm = F.instance_norm(input)
        re_act = F.relu(i_norm, inplace=True)
        device_id = re_act.device
        device_id = torch.device(re_act.device)
        output = F.conv3d(re_act, weight=self.conv_Off_filters.to(device_id), stride=1, padding=1) # padding = 1 for kernel =3
        return output #F.relu(output, inplace=True)