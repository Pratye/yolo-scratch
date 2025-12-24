"""
Bounding box loss functions: CIoU and center-weighted CIoU.

For the crater challenge we care more about center accuracy than
pure overlap, so we provide a variant that explicitly upweights
the center-distance term and mildly regularizes width/height.
"""

import torch
import torch.nn as nn
import math


def bbox_iou(box1, box2, xyxy=True, eps=1e-7):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) tensor
        box2: (M, 4) tensor
        xyxy: If True, boxes are in [x_min, y_min, x_max, y_max] format
              If False, boxes are in [x_center, y_center, width, height] format
        eps: Small value to avoid division by zero
    
    Returns:
        iou: (N, M) tensor of IoU values
    """
    if xyxy:
        # Convert to center format for easier calculation
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
        
        # Intersection
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + eps
        
        iou = inter_area / union_area
        
        # Calculate center distances and diagonal distances for CIoU
        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b1_w = b1_x2 - b1_x1
        b1_h = b1_y2 - b1_y1
        
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        b2_w = b2_x2 - b2_x1
        b2_h = b2_y2 - b2_y1
        
        center_dist_sq = (b1_cx.unsqueeze(1) - b2_cx.unsqueeze(0)) ** 2 + \
                        (b1_cy.unsqueeze(1) - b2_cy.unsqueeze(0)) ** 2
        
        # Diagonal distance of smallest enclosing box
        c_w = torch.max(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0)) - torch.min(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        c_h = torch.max(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0)) - torch.min(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        c_diag_sq = c_w ** 2 + c_h ** 2 + eps
        
        # Aspect ratio consistency
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(b2_w.unsqueeze(0) / (b2_h.unsqueeze(0) + eps)) -
            torch.atan(b1_w.unsqueeze(1) / (b1_h.unsqueeze(1) + eps)), 2
        )
        alpha = v / (1 - iou + v + eps)
        
        return iou, center_dist_sq, c_diag_sq, alpha
    else:
        raise NotImplementedError("Center format not implemented")


class CenterWeightedCIoULoss(nn.Module):
    """
    Center-weighted CIoU loss for crater bounding box regression.
    
    Compared to vanilla CIoU, this variant:
        - Places extra weight on center distance (for CDA-like metrics)
        - Adds a mild width/height consistency term
    """
    
    def __init__(self, center_weight: float = 2.0, size_weight: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.eps = eps
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) predicted boxes [x_min, y_min, x_max, y_max]
            target_boxes: (N, 4) target boxes [x_min, y_min, x_max, y_max]
        """
        iou, center_dist_sq, c_diag_sq, _ = bbox_iou(pred_boxes, target_boxes, xyxy=True, eps=self.eps)
        
        # IoU term (1 - IoU)
        iou_term = 1.0 - iou.diag()
        
        # Center-distance term (normalized squared distance)
        center_term = center_dist_sq.diag() / (c_diag_sq.diag() + self.eps)
        
        # Size term: relative squared error on width/height
        pw = pred_boxes[:, 2] - pred_boxes[:, 0]
        ph = pred_boxes[:, 3] - pred_boxes[:, 1]
        tw = target_boxes[:, 2] - target_boxes[:, 0]
        th = target_boxes[:, 3] - target_boxes[:, 1]
        
        size_term = ((pw - tw) / (tw + self.eps)) ** 2 + ((ph - th) / (th + self.eps)) ** 2
        
        loss = iou_term + self.center_weight * center_term + self.size_weight * size_term
        return loss.mean()
