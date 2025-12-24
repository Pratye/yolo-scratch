"""
Classification loss for crater classes.
Uses CrossEntropyLoss for multi-class classification.
Handles -1 (unknown class) gracefully.
"""

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """
    Classification loss using CrossEntropyLoss.
    Handles -1 (unknown class) by ignoring those samples.
    """
    
    def __init__(self, num_classes=5, ignore_index=-1):
        """
        Args:
            num_classes: Number of classes (5 for crater types)
            ignore_index: Class index to ignore (-1 for unknown)
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, pred_classes, target_classes, valid_mask=None):
        """
        Calculate classification loss.
        
        Args:
            pred_classes: (B, num_classes, H, W) predicted class logits
            target_classes: (B, H, W) target class indices (0-4, or -1 for unknown)
            valid_mask: Optional (B, H, W) mask for valid samples
        
        Returns:
            loss: Scalar classification loss
        """
        B, num_classes, H, W = pred_classes.shape
        
        # Reshape for CrossEntropyLoss: (B*H*W, num_classes) and (B*H*W,)
        pred_flat = pred_classes.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target_flat = target_classes.view(-1)
        
        # Calculate loss
        loss = self.ce_loss(pred_flat, target_flat)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            valid_flat = valid_mask.view(-1)
            loss = loss * valid_flat
        
        # Average over valid samples (excluding ignored)
        num_valid = (target_flat != self.ignore_index).float()
        if valid_mask is not None:
            num_valid = num_valid * valid_mask.view(-1)
        
        num_valid_sum = num_valid.sum()
        if num_valid_sum > 0:
            loss = (loss * num_valid).sum() / num_valid_sum
        else:
            loss = loss.sum() * 0.0
        
        return loss
