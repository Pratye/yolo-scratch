"""
Angular coverage loss for crater detection.

This loss encourages full rim coverage by penalizing angular bins
with near-zero activation. It converts the predicted mask into polar
coordinates around the bbox center and checks coverage across angle bins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AngularCoverageLoss(nn.Module):
    """
    Penalizes incomplete angular coverage of crater rim.
    
    Strategy:
    1. Convert mask to polar coordinates around bbox center
    2. Divide angle into fixed bins (e.g., 36 bins = 10 degrees each)
    3. Compute activation per bin
    4. Penalize bins with near-zero activation
    
    This encourages the model to predict complete rims rather than partial arcs.
    """
    
    def __init__(
        self,
        num_bins: int = 36,
        min_activation: float = 0.1,
        penalty_weight: float = 1.0,
    ):
        """
        Args:
            num_bins: Number of angular bins (default 36 = 10 deg per bin)
            min_activation: Minimum activation threshold (below this is penalized)
            penalty_weight: Weight for penalty per under-activated bin
        """
        super().__init__()
        self.num_bins = num_bins
        self.min_activation = min_activation
        self.penalty_weight = penalty_weight
    
    def _mask_to_polar_bins(
        self,
        mask: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Convert mask to polar coordinates and compute activation per angular bin.
        
        Args:
            mask: Predicted mask (B, 1, H, W), values in [0, 1]
            cx: Bbox center x coordinates (B,) in normalized [0, 1]
            cy: Bbox center y coordinates (B,) in normalized [0, 1]
            H: Mask height
            W: Mask width
        
        Returns:
            bin_activations: (B, num_bins) activation per angular bin
        """
        B = mask.shape[0]
        device = mask.device
        
        # Convert normalized coords to pixel coords
        cx_pixel = cx * W
        cy_pixel = cy * H
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        bin_activations_list = []
        
        for b in range(B):
            m = mask[b, 0]  # (H, W)
            cx_b = cx_pixel[b]
            cy_b = cy_pixel[b]
            
            # Compute relative coordinates
            dx = x_coords - cx_b
            dy = y_coords - cy_b
            
            # Compute angles (in radians, [-pi, pi])
            angles = torch.atan2(dy, dx)  # (H, W)
            
            # Convert to [0, 2*pi] then normalize to [0, num_bins]
            angles_normalized = (angles + np.pi) / (2 * np.pi) * self.num_bins
            bin_indices = torch.clamp(angles_normalized.long(), 0, self.num_bins - 1)  # (H, W)
            
            # Compute activation per bin
            bin_activations = torch.zeros(self.num_bins, device=device)
            for bin_idx in range(self.num_bins):
                bin_mask = (bin_indices == bin_idx)
                if bin_mask.any():
                    bin_activations[bin_idx] = m[bin_mask].mean()
            
            bin_activations_list.append(bin_activations)
        
        return torch.stack(bin_activations_list)  # (B, num_bins)
    
    def forward(
        self,
        mask: torch.Tensor,
        bbox: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute angular coverage loss.
        
        Args:
            mask: Predicted mask (B, 1, H, W), values in [0, 1] after sigmoid
            bbox: Predicted bbox (B, 4) with [cx, cy, w, h] in normalized [0, 1]
            valid_mask: Optional mask indicating valid samples (B,)
        
        Returns:
            loss: Scalar loss value
        """
        B, C, H, W = mask.shape
        
        # Extract bbox centers
        cx = bbox[:, 0]  # (B,)
        cy = bbox[:, 1]  # (B,)
        
        # Filter to valid samples
        if valid_mask is not None:
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=mask.device, requires_grad=True)
            mask = mask[valid_indices]
            cx = cx[valid_indices]
            cy = cy[valid_indices]
            B = len(valid_indices)
        
        # Compute angular bin activations
        bin_activations = self._mask_to_polar_bins(mask, cx, cy, H, W)  # (B, num_bins)
        
        # Penalize bins below minimum activation
        under_activated = (bin_activations < self.min_activation).float()  # (B, num_bins)
        penalty_per_sample = under_activated.sum(dim=1) / self.num_bins  # (B,) normalized
        
        # Average loss across batch
        loss = self.penalty_weight * penalty_per_sample.mean()
        
        return loss
