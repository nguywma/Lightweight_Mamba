import torch
import torch.nn as nn
import torch.nn.functional as F

class LightUp(nn.Module):
    """
    Lightweight content-aware upsampling (replaces LDA_AQU).
    Uses Pixel Shuffle + Channel SE-gate + DW refinement.
    VRAM-efficient: O(C*H*W) vs O(k²*C*H*W) for deformable attention.
    """
    def __init__(self, in_channels, scale_factor=2, reduction=16):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 1) Channel-attention gate (SE-style) - content-aware
        # This allows the network to emphasize important features before upscaling
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 2) Pixel shuffle expand: in_channels -> in_channels * r²
        # PixelShuffle is much more VRAM efficient than grid_sample based upsampling
        self.expand = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 1, bias=False)
        self.ps = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(in_channels)
        
        # 3) Depthwise refinement (spatial)
        # Refines the spatial features after upscaling to minimize artifacts
        self.dw_refine = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                   groups=in_channels, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # Apply channel attention (content-aware gating)
        attn = self.se(x)
        x = x * attn
        
        # Upscale using PixelShuffle
        x = self.expand(x)
        x = self.ps(x)
        x = self.bn(x)
        
        # Spatial refinement
        x = x + self.act(self.dw_refine(x))
        
        return x
