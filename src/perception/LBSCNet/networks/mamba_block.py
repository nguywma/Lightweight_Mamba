import torch
import torch.nn as nn
from networks.vssblock import *
from networks.axialdw import AxialDW
# from networks.mambavision import *

class MambaSplit(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ins_norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.vss = VSSBlock(hidden_dim=in_ch // 2)
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        x = self.dw(x)
        x1_chunk, x2_chunk = torch.chunk(x, 2, dim = 1)
        
        x1 = x1_chunk.permute(0, 2, 3, 1)
        x1 = self.vss(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = x1 + self.scale*x1_chunk
        
        x2 = x2_chunk.permute(0, 2, 3, 1)
        x2 = self.vss(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 + self.scale*x2_chunk
        
        x = torch.cat([x1, x2], dim = 1)
        x = self.act(self.ins_norm(x))
        
        return x
        
class MaxAvgFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch + in_ch, in_ch, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        ori = x
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_avg, x_max], dim = 1)
        x = self.sigmoid(self.conv(x))
        x = x * ori
        return x

class MultiFeatureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, dropout = 0.1):
        super().__init__()
        self.maxavgfusion = MaxAvgFusion(in_ch)
        self.mamba = MambaSplit(in_ch)
        self.adw = AxialDW(in_ch, mixer_kernel=(3,3))
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)  # Use Dropout2d for spatial dropout

    def forward(self, x):
        x1 = self.mamba(x)
        x2 = self.maxavgfusion(x)
        x = x1 + x2
        x = self.pw(self.dropout(self.act(self.bn(self.adw(x)))))
        return x 
