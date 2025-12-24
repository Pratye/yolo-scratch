"""
Objectness loss for YOLO detection.
Uses BCEWithLogitsLoss for positive/negative sample classification.
"""

import torch
import torch.nn as nn


class ObjectnessLoss(nn.Module):
    """
    Objectness loss using BCEWithLogitsLoss.
    Positive samples: IoU > pos_threshold
    Negative samples: IoU < neg_threshold
    """
    
    def __init__(self, pos_threshold=0.5, neg_threshold=0.4):
        """
        Args:
            pos_threshold: IoU threshold for positive samples
            neg_threshold: IoU threshold for negative samples
        """
        super().__init__()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred_objectness, target_objectness, ignore_mask=None):
        """
        Calculate objectness loss.
        
        Args:
            pred_objectness: (B, 1, H, W) predicted objectness logits
            target_objectness: (B, 1, H, W) target objectness (1 for positive, 0 for negative, -1 for ignore)
            ignore_mask: Optional (B, 1, H, W) mask for ignored regions
        
        Returns:
            loss: Scalar objectness loss
        """
        # Create valid mask (ignore -1 values)
        valid_mask = (target_objectness >= 0).float()
        if ignore_mask is not None:
            valid_mask = valid_mask * (1 - ignore_mask.float())
        
        # Calculate loss only on valid samples
        loss = self.bce_loss(pred_objectness, target_objectness.clamp(0, 1))
        loss = loss * valid_mask
        
        # Average over valid samples
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum() * 0.0
        
        return loss
