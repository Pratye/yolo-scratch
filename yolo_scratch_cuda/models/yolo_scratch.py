"""
Complete YOLO model from scratch for crater detection.
Combines backbone, neck, and head.
"""

import torch
import torch.nn as nn
from .backbone import YOLOBackbone
from .neck import YOLONeck
from .head import YOLOHead


class YOLOScratch(nn.Module):
    """
    Complete YOLO-style object detection model from scratch.
    Architecture: Backbone -> Neck -> Head
    """
    
    def __init__(self, num_classes: int = 5, in_channels: int = 1):
        """
        Args:
            num_classes: Number of classes (5 for crater types: A, AB, B, BC, C)
            in_channels: Input channels (1 for grayscale)
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Build model components
        # Adjust width_mult to target ~10M parameters
        # With width_mult=1.0: ~7.9M params
        # With width_mult=1.2: ~11.4M params
        # Target: ~10M, so use width_mult=1.15
        width_mult = 1.15
        self.backbone = YOLOBackbone(in_channels=in_channels, width_mult=width_mult)
        # Channel layout must match YOLOBackbone (after width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        c5 = int(384 * width_mult)
        # Widen neck output channels proportionally
        out_channels = int(160 * width_mult)
        self.neck = YOLONeck(in_channels_list=[c2, c3, c4, c5], out_channels=out_channels)
        self.head = YOLOHead(in_channels=out_channels, num_classes=num_classes, num_scales=4)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            List of 4 dictionaries (one per scale), each containing:
                - bbox: (B, 4, H, W) - bbox predictions (logits for cx, cy, w, h)
                - objectness: (B, 1, H, W) - objectness logits
                - classes: (B, num_classes, H, W) - class logits
        """
        # Backbone: extract multi-scale features
        features = self.backbone(x)
        
        # Neck: aggregate features
        neck_features = self.neck(features)
        
        # Head: detection predictions
        predictions = self.head(neck_features)
        
        return predictions
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        total_params = self.count_parameters()
        
        summary = f"""
YOLO Scratch Model Summary
==========================
Input channels: {self.in_channels} (grayscale)
Number of classes: {self.num_classes}
Total parameters: {total_params:,} ({total_params/1e6:.2f}M)

Architecture:
- Backbone: YOLOBackbone (CSP blocks, BatchNorm, SiLU) with P2–P5 outputs (strides 4, 8, 16, 32)
- Neck: YOLONeck (FPN feature aggregation)
- Head: YOLOHead (anchor-free detection, cx/cy/w/h)

Output scales: 4 (1/4, 1/8, 1/16, 1/32 of input resolution)
"""
        return summary
