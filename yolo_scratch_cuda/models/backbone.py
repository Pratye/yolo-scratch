"""
YOLO-style backbone with CSP blocks, BatchNorm, and SiLU activation.
This version is shallower and narrower to target ~25–35M parameters and
produces four feature maps (P2–P5) at strides 4, 8, 16, and 32.
"""

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """SiLU activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """
    CSP (Cross Stage Partial) block.
    Splits input, processes through two paths, and concatenates.
    """
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Main path
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNSiLU(hidden_channels * 2, out_channels, 1)  # *2 because of concat
        
        # Shortcut path
        self.conv_shortcut = ConvBNSiLU(in_channels, hidden_channels, 1) if in_channels != hidden_channels else nn.Identity()
        
        # Bottleneck blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBNSiLU(hidden_channels, hidden_channels, 3),
                ConvBNSiLU(hidden_channels, hidden_channels, 3)
            ) for _ in range(num_blocks)
        ])
        
        self.act = SiLU()
    
    def forward(self, x):
        # Main path
        main = self.conv1(x)
        main = self.blocks(main)
        
        # Shortcut path
        shortcut = self.conv_shortcut(x)
        
        # Concatenate
        out = torch.cat([main, shortcut], dim=1)
        
        # Final conv and activate
        out = self.conv2(out)
        return self.act(out)


class YOLOBackbone(nn.Module):
    """
    YOLO-style backbone with CSP blocks.
    
    Input: 1-channel grayscale images
    Output: Multi-scale feature maps at 4 different scales:
        - P2: stride 4
        - P3: stride 8
        - P4: stride 16
        - P5: stride 32
    """
    
    def __init__(self, in_channels: int = 1, width_mult: float = 1):
        """
        Args:
            in_channels: Input channels (1 for grayscale)
            width_mult: Width multiplier for channels (1.25-1.5x to reach 8-10M params)
        """
        super().__init__()
        
        # Base channels (widened for 8-10M parameter target)
        c2 = int(64 * width_mult)   # P2: stride 4
        c3 = int(128 * width_mult)  # P3: stride 8
        c4 = int(256 * width_mult)  # P4: stride 16
        c5 = int(384 * width_mult)  # P5: stride 32
        
        # Stem: downsample to stride 4
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, c2 // 2, kernel_size=3, stride=2),  # /2
            ConvBNSiLU(c2 // 2, c2, kernel_size=3, stride=2),           # /4 -> P2 input
        )
        
        # P2: stride 4
        self.stage_p2 = CSPBlock(c2, c2, num_blocks=1)
        
        # P3: stride 8
        self.down_p2_p3 = ConvBNSiLU(c2, c3, kernel_size=3, stride=2)   # /8
        self.stage_p3 = CSPBlock(c3, c3, num_blocks=2)
        
        # P4: stride 16
        self.down_p3_p4 = ConvBNSiLU(c3, c4, kernel_size=3, stride=2)   # /16
        self.stage_p4 = CSPBlock(c4, c4, num_blocks=2)
        
        # P5: stride 32
        self.down_p4_p5 = ConvBNSiLU(c4, c5, kernel_size=3, stride=2)   # /32
        self.stage_p5 = CSPBlock(c5, c5, num_blocks=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            List of feature maps at different scales:
            - P2: (B, c2, H/4,  W/4)
            - P3: (B, c3, H/8,  W/8)
            - P4: (B, c4, H/16, W/16)
            - P5: (B, c5, H/32, W/32)
        """
        x = self.stem(x)          # /4
        p2 = self.stage_p2(x)     # P2
        
        x = self.down_p2_p3(p2)   # /8
        p3 = self.stage_p3(x)     # P3
        
        x = self.down_p3_p4(p3)   # /16
        p4 = self.stage_p4(x)     # P4
        
        x = self.down_p4_p5(p4)   # /32
        p5 = self.stage_p5(x)     # P5
        
        return [p2, p3, p4, p5]
