"""
Ellipse consistency loss for crater detection.

This loss encourages consistency between:
1. Ellipse parameters derived from predicted mask (via image moments)
2. Ellipse parameters implied by predicted bounding box

This acts as a geometry-aware proxy loss that aligns mask and bbox predictions
without directly optimizing on CDA score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class EllipseConsistencyLoss(nn.Module):
    """
    Penalizes mismatch between mask-derived and bbox-derived ellipse parameters.
    
    From mask: Compute ellipse via image moments (center, axes, orientation)
    From bbox: Convert bbox [cx, cy, w, h] to ellipse [cx, cy, a, b, angle=0]
    
    Loss components:
    - Center distance (L2)
    - Axis ratio mismatch
    - Orientation mismatch (for mask-derived ellipse)
    """
    
    def __init__(
        self,
        center_weight: float = 1.0,
        axis_weight: float = 1.0,
        angle_weight: float = 0.5,
    ):
        """
        Args:
            center_weight: Weight for center mismatch loss
            axis_weight: Weight for axis ratio mismatch loss
            angle_weight: Weight for orientation mismatch loss
        """
        super().__init__()
        self.center_weight = center_weight
        self.axis_weight = axis_weight
        self.angle_weight = angle_weight
    
    def _mask_to_ellipse(self, mask: torch.Tensor) -> tuple:
        """
        Compute ellipse parameters from mask using image moments.
        
        Args:
            mask: Binary mask (B, 1, H, W), values in [0, 1]
        
        Returns:
            (cx, cy, semi_major, semi_minor, angle_deg): Ellipse parameters
                Returns None if mask is empty or invalid
        """
        B, C, H, W = mask.shape
        
        # Binarize mask
        mask_binary = (mask > 0.5).float()
        
        # Compute moments for each sample in batch
        cx_list = []
        cy_list = []
        sma_list = []
        smb_list = []
        angle_list = []
        valid_list = []
        
        for b in range(B):
            m = mask_binary[b, 0].cpu().numpy()  # (H, W)
            
            # Check if mask has any foreground
            if m.sum() < 10:  # Too few pixels
                valid_list.append(False)
                continue
            
            # Compute image moments
            moments = cv2.moments(m)
            if moments['m00'] < 1e-6:
                valid_list.append(False)
                continue
            
            # Center
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            # Central moments for covariance
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            
            # Eigenvalues of covariance matrix give axes
            # Cov = [[mu20, mu11], [mu11, mu02]]
            trace = mu20 + mu02
            det = mu20 * mu02 - mu11 * mu11
            
            if det < 0:
                valid_list.append(False)
                continue
            
            lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det))
            lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det))
            
            if lambda1 < 0 or lambda2 < 0:
                valid_list.append(False)
                continue
            
            # Semi-axes (scaled by 4 for ellipse fitting convention)
            sma = 2 * np.sqrt(lambda1)
            smb = 2 * np.sqrt(lambda2)
            
            # Ensure semi_major >= semi_minor
            if sma < smb:
                sma, smb = smb, sma
            
            # Orientation
            if abs(mu20 - mu02) < 1e-6:
                angle = 0.0
            else:
                angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            angle_deg = np.degrees(angle)
            
            cx_list.append(cx)
            cy_list.append(cy)
            sma_list.append(sma)
            smb_list.append(smb)
            angle_list.append(angle_deg)
            valid_list.append(True)
        
        if not any(valid_list):
            return None
        
        # Convert to tensors
        cx_tensor = torch.tensor(cx_list, device=mask.device, dtype=mask.dtype)
        cy_tensor = torch.tensor(cy_list, device=mask.device, dtype=mask.dtype)
        sma_tensor = torch.tensor(sma_list, device=mask.device, dtype=mask.dtype)
        smb_tensor = torch.tensor(smb_list, device=mask.device, dtype=mask.dtype)
        angle_tensor = torch.tensor(angle_list, device=mask.device, dtype=mask.dtype)
        
        return (cx_tensor, cy_tensor, sma_tensor, smb_tensor, angle_tensor), valid_list
    
    def _bbox_to_ellipse(self, bbox: torch.Tensor) -> tuple:
        """
        Convert bbox [cx, cy, w, h] to ellipse parameters.
        
        Args:
            bbox: Bbox tensor (N, 4) with [cx, cy, w, h] in normalized [0, 1]
        
        Returns:
            (cx, cy, semi_major, semi_minor, angle): Ellipse parameters
        """
        cx = bbox[:, 0]
        cy = bbox[:, 1]
        w = bbox[:, 2]
        h = bbox[:, 3]
        
        # Semi-axes (half of width/height)
        sma = torch.max(w, h) / 2.0
        smb = torch.min(w, h) / 2.0
        
        # Angle is 0 for axis-aligned bbox (we don't predict rotation in bbox)
        angle = torch.zeros_like(cx)
        
        return cx, cy, sma, smb, angle
    
    def forward(
        self,
        mask: torch.Tensor,
        bbox: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute ellipse consistency loss.
        
        Args:
            mask: Predicted mask (B, 1, H, W), values in [0, 1] after sigmoid
            bbox: Predicted bbox (N, 4) with [cx, cy, w, h] in normalized [0, 1]
            valid_mask: Optional mask indicating valid samples (B,) or (N,)
        
        Returns:
            loss: Scalar loss value
        """
        # Get mask-derived ellipse
        mask_ellipse_result = self._mask_to_ellipse(mask)
        if mask_ellipse_result is None:
            return torch.tensor(0.0, device=mask.device, requires_grad=True)
        
        (mask_cx, mask_cy, mask_sma, mask_smb, mask_angle), valid_list = mask_ellipse_result
        
        # Filter to valid samples
        if valid_mask is not None:
            valid_list = [v and valid_mask[i].item() for i, v in enumerate(valid_list)]
        
        if not any(valid_list):
            return torch.tensor(0.0, device=mask.device, requires_grad=True)
        
        # Get bbox-derived ellipse (only for valid samples)
        valid_indices = [i for i, v in enumerate(valid_list) if v]
        bbox_valid = bbox[valid_indices]
        bbox_cx, bbox_cy, bbox_sma, bbox_smb, bbox_angle = self._bbox_to_ellipse(bbox_valid)
        
        # Extract valid mask ellipse params
        mask_cx_valid = mask_cx[valid_indices]
        mask_cy_valid = mask_cy[valid_indices]
        mask_sma_valid = mask_sma[valid_indices]
        mask_smb_valid = mask_smb[valid_indices]
        mask_angle_valid = mask_angle[valid_indices]
        
        # Center distance loss (normalized by image size, assume 1.0 for normalized coords)
        center_diff_x = mask_cx_valid - bbox_cx
        center_diff_y = mask_cy_valid - bbox_cy
        center_loss = torch.sqrt(center_diff_x**2 + center_diff_y**2 + 1e-6).mean()
        
        # Axis ratio loss (relative difference)
        mask_ratio = mask_sma_valid / (mask_smb_valid + 1e-6)
        bbox_ratio = bbox_sma / (bbox_smb + 1e-6)
        axis_loss = torch.abs(mask_ratio - bbox_ratio).mean()
        
        # Orientation loss (mask has angle, bbox angle is 0, so just penalize mask angle)
        # Normalize angle to [0, 90] degrees
        angle_normalized = torch.abs(mask_angle_valid) % 90.0
        angle_loss = (angle_normalized / 90.0).mean()  # Normalized to [0, 1]
        
        # Weighted combination
        total_loss = (
            self.center_weight * center_loss +
            self.axis_weight * axis_loss +
            self.angle_weight * angle_loss
        )
        
        return total_loss
