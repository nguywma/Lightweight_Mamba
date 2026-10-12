import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.CBAM import CBAM
from networks.bottleneck import SKUnit
from networks.mamba_block import *
from networks.axialdw import AxialDW
from mmcv.ops.carafe import CARAFEPack
from networks.mamba_block import MultiFeatureAggregation

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_ch, mixer_kernel=mixer_kernel)
        self.bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))

        self.act = nn.ReLU()

        self.identity_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.pw(x)
        x = self.down(x)

        return x + self.alpha * self.identity_conv(self.down(identity))


class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c_up, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        # CARAFE explicitly generates local kernels for feature reassembly, much like LQA_AQU
        self.up = CARAFEPack(in_c_up, 2, up_kernel=5, up_group=4, encoder_kernel=3)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up = nn.ConvTranspose2d(in_c_up, in_c_up, kernel_size=2, stride=2)
        self.pw = nn.Conv2d(in_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel)
        self.act = nn.ReLU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.pw2(self.dw(self.act(self.bn(self.pw(x)))))
        return x
    
class BEVFusionv1(nn.Module):           # adaptive representation fusion module
    def __init__(self, channel):
        super().__init__()

        self.attention_bev = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_sem = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_com = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.adapter_sem = nn.Conv2d(channel//2, channel, 1)       # 1/2 => 1
        self.adapter_com = nn.Conv2d(channel//2, channel, 1)       # 1/2 => 1

    def forward(self, bev_features, sem_features, com_features):
        sem_features = self.adapter_sem(sem_features)           
        com_features = self.adapter_com(com_features)       
        attn_bev = self.attention_bev(bev_features)
        attn_sem = self.attention_sem(sem_features)
        attn_com = self.attention_com(com_features)

        fusion_features = torch.mul(bev_features, attn_bev) \
            + torch.mul(sem_features, attn_sem) \
            + torch.mul(com_features, attn_com)
        
        return fusion_features

    
class BEVUNetv1(nn.Module):
    def __init__(self, n_class, n_height, dilation, bilinear, group_conv, input_batch_norm, dropout, circular_padding, dropblock):
        super().__init__()
        self.inc = inconv(64 ,64, dilation, input_batch_norm, circular_padding)   # Batchnorm => (Conv2d => Batchnorm => leaky_ReLU) * 2
        self.down1 = EncoderBlock(64, 128)
        self.down2 = EncoderBlock(128 , 256)
        self.down3 = EncoderBlock(256, 512)
        self.down4 = EncoderBlock(512, 512)

        self.cbam4 = CBAM(512)
        self.cbam3 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam1 = CBAM(64)
        
        self.up1 = DecoderBlock(512, 1024, 512)
        self.up2 = DecoderBlock(512, 768, 256)
        self.up3 = DecoderBlock(256, 384, 128)
        self.up4 = DecoderBlock(128, 192, 128)


        self.bottleneck = MultiFeatureAggregation(512, 512)
        
        self.dropout = nn.Dropout(dropout)
        self.outc = outconv(128, n_class)
        
        channels = [128, 256, 512]
        self.bev_fusions = nn.ModuleList([BEVFusionv1(channels[i]) for i in range(3)])  #BEVFusionv1(x) with x equals 128, 256 or 512
    
    def forward(self, x , sem_fea_list, com_fea_list):        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        
        x2_f = self.bev_fusions[0](x2, sem_fea_list[0], com_fea_list[0])        
        x3 = self.down2(x2_f)

        x3_f = self.bev_fusions[1](x3, sem_fea_list[1], com_fea_list[1])        
        x4 = self.down3(x3_f)

        x4_f = self.bev_fusions[2](x4, sem_fea_list[2], com_fea_list[2])        
        x5 = self.down4(x4_f)

        x = self.bottleneck(x5)
        
        x4_f = self.cbam4(x4_f)
        x3_f = self.cbam3(x3_f)
        x2_f = self.cbam2(x2_f)
        x1 = self.cbam1(x1)
        
        x = self.up1(x, x4_f)
        x = self.up2(x, x3_f)    
        x = self.up3(x, x2_f)        
        x = self.up4(x, x1)
        
        
        x = self.outc(self.dropout(x))        
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
