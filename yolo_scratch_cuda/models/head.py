"""
Anchor-free detection head for YOLO.
Outputs:
    - bbox regression (cx, cy, w, h)
    - objectness
    - class logits
"""

import torch
import torch.nn as nn
from .backbone import ConvBNSiLU


class YOLOHead(nn.Module):
    """
    Anchor-free detection head.
    For each scale, predicts:
    - Bounding box regression (4 values: cx, cy, w, h) in normalized [0, 1]
    - Objectness score (1 value)
    - Class logits (num_classes values)
    """
    
    def __init__(self, in_channels: int = 160, num_classes: int = 5, num_scales: int = 4):
        """
        Args:
            in_channels: Number of input channels from neck
            num_classes: Number of classes (5 for crater types: A, AB, B, BC, C)
            num_scales: Number of detection scales (4: P2–P5)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        
        # Shared feature extraction for each scale
        self.shared_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1),
                ConvBNSiLU(in_channels, in_channels, 3, 1, 1),
            ) for _ in range(num_scales)
        ])
        
        # Detection heads for each scale
        # Each head outputs: bbox (4) + objectness (1) + classes (num_classes)
        # Bbox format: [cx, cy, w, h] in normalized coordinates [0, 1]
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.SiLU(),
                nn.Conv2d(in_channels, 4, kernel_size=1)
            ) for _ in range(num_scales)
        ])
        self.objectness_heads = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1) for _ in range(num_scales)
        ])
        self.class_heads = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, kernel_size=1) for _ in range(num_scales)
        ])
    
    def forward(self, features):
        """
        Forward pass through detection head.
        
        Args:
            features: List of 4 feature maps from neck
                - feat2: (B, in_channels, H/4,  W/4)  - P2, stride 4
                - feat3: (B, in_channels, H/8,  W/8)  - P3, stride 8
                - feat4: (B, in_channels, H/16, W/16) - P4, stride 16
                - feat5: (B, in_channels, H/32, W/32) - P5, stride 32
        
        Returns:
            List of 4 dictionaries, one per scale, each containing:
                - bbox: (B, 4, H, W) - bbox predictions (logits for cx, cy, w, h)
                - objectness: (B, 1, H, W) - objectness logits
                - classes: (B, num_classes, H, W) - class logits
        """
        outputs = []
        
        for i, feat in enumerate(features):
            x = self.shared_convs[i](feat)
            
            bbox = self.bbox_heads[i](x)           # (B, 4, H, W)
            objectness = self.objectness_heads[i](x)  # (B, 1, H, W)
            classes = self.class_heads[i](x)          # (B, num_classes, H, W)
            
            out_dict = {
                'bbox': bbox,
                'objectness': objectness,
                'classes': classes
            }
            
            outputs.append(out_dict)
        
        return outputs
