"""
YOLO-style neck with FPN feature aggregation.
Aggregates multi-scale features from backbone (P2–P5) and produces
four detection feature maps at strides 4, 8, 16, and 32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNSiLU


class YOLONeck(nn.Module):
    """
    FPN-style neck for multi-scale feature aggregation.
    """
    
    def __init__(self, in_channels_list, out_channels=160):
        """
        Args:
            in_channels_list: List of input channel counts for each scale [P2, P3, P4, P5]
            out_channels: Output channels for each scale
        """
        super().__init__()
        
        c2, c3, c4, c5 = in_channels_list
        
        # 1x1 lateral convs to unify channel dimensions
        self.lateral_p5 = ConvBNSiLU(c5, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_p4 = ConvBNSiLU(c4, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_p3 = ConvBNSiLU(c3, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_p2 = ConvBNSiLU(c2, out_channels, kernel_size=1, stride=1, padding=0)
        
        # 3x3 smoothing convs after fusion
        self.smooth_p5 = ConvBNSiLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth_p4 = ConvBNSiLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = ConvBNSiLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth_p2 = ConvBNSiLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, features):
        """
        Forward pass through FPN neck.
        
        Args:
            features: List of 4 feature maps from backbone
                - P2: (B, c2, H/4,  W/4)
                - P3: (B, c3, H/8,  W/8)
                - P4: (B, c4, H/16, W/16)
                - P5: (B, c5, H/32, W/32)
        
        Returns:
            List of 4 aggregated feature maps:
                - N2: (B, out_channels, H/4,  W/4)
                - N3: (B, out_channels, H/8,  W/8)
                - N4: (B, out_channels, H/16, W/16)
                - N5: (B, out_channels, H/32, W/32)
        """
        p2, p3, p4, p5 = features
        
        # Lateral projections
        l5 = self.lateral_p5(p5)
        l4 = self.lateral_p4(p4)
        l3 = self.lateral_p3(p3)
        l2 = self.lateral_p2(p2)
        
        # Top-down pathway
        n5 = self.smooth_p5(l5)
        
        up5 = F.interpolate(l5, size=l4.shape[2:], mode='nearest')
        n4 = self.smooth_p4(l4 + up5)
        
        up4 = F.interpolate(n4, size=l3.shape[2:], mode='nearest')
        n3 = self.smooth_p3(l3 + up4)
        
        up3 = F.interpolate(n3, size=l2.shape[2:], mode='nearest')
        n2 = self.smooth_p2(l2 + up3)
        
        return [n2, n3, n4, n5]
