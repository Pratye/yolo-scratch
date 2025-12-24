"""
Combined YOLO loss function for crater detection.

This loss is tuned toward the CDA-quality metric rather than COCO mAP:
- Bbox regression uses a center-weighted CIoU variant
- Classification is down-weighted (not directly scored in CDA)
- Spatial responsibility: only center grid cell gets GT assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bbox_loss import CenterWeightedCIoULoss
from .objectness_loss import ObjectnessLoss
from .class_loss import ClassificationLoss
from config import loss_cfg


class YOLOLoss(nn.Module):
    """
    Combined YOLO loss function.
    Weighted combination of:
    - Bounding box loss (CIoU)
    - Objectness loss (BCE)
    - Classification loss (CrossEntropy)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        bbox_loss_weight: float | None = None,
        objectness_loss_weight: float | None = None,
        class_loss_weight: float | None = None,
        aux_mask_loss_weight: float | None = None,  # legacy, kept for backward compat (ignored)
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.4,
    ):
        """
        Args:
            num_classes: Number of classes
            bbox_loss_weight: Weight for bbox loss
            objectness_loss_weight: Weight for objectness loss
            class_loss_weight: Weight for classification loss
            pos_threshold: IoU threshold for positive samples
            neg_threshold: IoU threshold for negative samples
        """
        super().__init__()
        # Center-weighted CIoU emphasizes crater center accuracy.
        self.bbox_loss = CenterWeightedCIoULoss(
            center_weight=loss_cfg.center_weight,
            size_weight=loss_cfg.size_weight,
        )
        self.objectness_loss = ObjectnessLoss(pos_threshold, neg_threshold)
        self.class_loss = ClassificationLoss(num_classes)
        
        # Loss weights (if explicit weights are not passed, fall back to config)
        self.bbox_loss_weight = bbox_loss_weight if bbox_loss_weight is not None else loss_cfg.bbox_loss_weight
        self.objectness_loss_weight = (
            objectness_loss_weight if objectness_loss_weight is not None else loss_cfg.objectness_loss_weight
        )
        self.class_loss_weight = class_loss_weight if class_loss_weight is not None else loss_cfg.class_loss_weight
    
    def forward(
        self,
        predictions,
        targets,
        current_epoch: int = 1,
    ):
        """
        Calculate combined YOLO loss.
        
        Args:
            predictions: List of 3 dicts (one per scale), each containing:
                - bbox: (B, 4, H, W)
                - objectness: (B, 1, H, W)
                - classes: (B, num_classes, H, W)
            targets: List of target dicts (one per scale), each containing:
                - bbox: (B, 4, H, W) or None
                - objectness: (B, 1, H, W)
                - classes: (B, H, W)
        
        Returns:
            loss_dict: Dictionary with individual and total losses
        """
        total_bbox_loss = 0.0
        total_objectness_loss = 0.0
        total_class_loss = 0.0
        aux_mask_loss = 0.0
        
        num_scales = len(predictions)
        
        for scale_idx in range(num_scales):
            pred = predictions[scale_idx]
            target = targets[scale_idx]
            
            # Bbox loss (only on positive samples).
            # Bboxes are parameterized as [cx, cy, w, h] in normalized [0, 1].
            if target['bbox'] is not None and target['bbox'].numel() > 0:
                # Shapes
                B_pred, _, H_pred, W_pred = pred['bbox'].shape
                
                # Reshape bbox predictions and targets for loss calculation
                pred_bbox_flat = pred['bbox'].permute(0, 2, 3, 1).contiguous().view(-1, 4)
                target_bbox_flat = target['bbox'].permute(0, 2, 3, 1).contiguous().view(-1, 4)
                
                # Positives from objectness target
                obj_target = target['objectness'].squeeze(1)  # (B, H, W)
                pos_mask = (obj_target > 0.5).view(-1)
                
                if pos_mask.sum() > 0:
                    # Decode bbox predictions: cx, cy use sigmoid; w, h use exp() with min clamping
                    pred_cx = torch.sigmoid(pred_bbox_flat[:, 0])  # [0, 1]
                    pred_cy = torch.sigmoid(pred_bbox_flat[:, 1])  # [0, 1]
                    pred_w = torch.exp(pred_bbox_flat[:, 2]).clamp(min=0.01)  # exp() with min constraint
                    pred_h = torch.exp(pred_bbox_flat[:, 3]).clamp(min=0.01)  # exp() with min constraint
                    pred_xyxy = torch.stack([
                        pred_cx - pred_w * 0.5,
                        pred_cy - pred_h * 0.5,
                        pred_cx + pred_w * 0.5,
                        pred_cy + pred_h * 0.5,
                    ], dim=-1)
                    
                    tgt_cx = target_bbox_flat[:, 0]
                    tgt_cy = target_bbox_flat[:, 1]
                    tgt_w = torch.clamp(target_bbox_flat[:, 2], min=0.01)
                    tgt_h = torch.clamp(target_bbox_flat[:, 3], min=0.01)
                    tgt_xyxy = torch.stack([
                        tgt_cx - tgt_w * 0.5,
                        tgt_cy - tgt_h * 0.5,
                        tgt_cx + tgt_w * 0.5,
                        tgt_cy + tgt_h * 0.5,
                    ], dim=-1)
                    
                    bbox_loss = self.bbox_loss(
                        pred_xyxy[pos_mask],
                        tgt_xyxy[pos_mask]
                    )
                    total_bbox_loss += bbox_loss
            
            # Objectness loss
            obj_loss = self.objectness_loss(
                pred['objectness'],
                target['objectness']
            )
            total_objectness_loss += obj_loss
            
            # Classification loss (only on positive samples)
            class_loss = self.class_loss(
                pred['classes'],
                target['classes'],
                valid_mask=(target['objectness'] > 0.5).squeeze(1)
            )
            total_class_loss += class_loss
        
        # Average over scales
        total_bbox_loss = total_bbox_loss / num_scales
        total_objectness_loss = total_objectness_loss / num_scales
        total_class_loss = total_class_loss / num_scales
        
        # Weighted combination (crater-optimized).
        total_loss = (
            self.bbox_loss_weight * total_bbox_loss +
            self.objectness_loss_weight * total_objectness_loss +
            self.class_loss_weight * total_class_loss
        )
        
        return {
            'total_loss': total_loss,
            'bbox_loss': total_bbox_loss,
            'objectness_loss': total_objectness_loss,
            'class_loss': total_class_loss,
        }
