"""
Rim-weighted mask loss for crater detection.

This loss emphasizes crater rim pixels over interior pixels,
encouraging the model to learn accurate rim boundaries for better
ellipse fitting and CDA Gaussian angle computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class RimWeightedMaskLoss(nn.Module):
    """
    BCE + Dice loss with rim-weighted pixel importance.
    
    Rim band is extracted from GT mask by:
    1. Computing mask edges (boundary)
    2. Dilating edges to form a thin rim band
    3. Assigning higher weights to rim pixels
    
    Pixel weights:
    - Rim pixels: weight = rim_weight (default 4.0)
    - Interior pixels: weight = 1.0
    - Background pixels: weight = bg_weight (default 0.5)
    """
    
    def __init__(
        self,
        rim_weight: float = 4.0,
        bg_weight: float = 0.5,
        rim_dilation_kernel: int = 3,
        use_dice: bool = True,
        dice_weight: float = 0.5,
    ):
        """
        Args:
            rim_weight: Weight multiplier for rim pixels
            bg_weight: Weight multiplier for background pixels
            rim_dilation_kernel: Kernel size for rim dilation
            use_dice: Whether to add Dice loss
            dice_weight: Weight for Dice loss (BCE weight = 1 - dice_weight)
        """
        super().__init__()
        self.rim_weight = rim_weight
        self.bg_weight = bg_weight
        self.rim_dilation_kernel = rim_dilation_kernel
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
    
    def _extract_rim_band(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract rim band from binary mask.
        
        Args:
            mask: Binary mask (B, 1, H, W) or (B, H, W), values in [0, 1]
        
        Returns:
            rim_band: Binary mask indicating rim pixels (same shape as mask)
        """
        # Convert to numpy for OpenCV operations
        if mask.dim() == 4:
            mask_np = mask.squeeze(1).cpu().numpy()  # (B, H, W)
        else:
            mask_np = mask.cpu().numpy()
        
        batch_size = mask_np.shape[0]
        rim_bands = []
        
        for b in range(batch_size):
            m = (mask_np[b] > 0.5).astype(np.uint8) * 255
            
            # Compute mask boundary using erosion
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(m, kernel, iterations=1)
            boundary = m - eroded  # Boundary pixels
            
            # Dilate boundary to form rim band
            dilate_kernel = np.ones((self.rim_dilation_kernel, self.rim_dilation_kernel), np.uint8)
            rim_band = cv2.dilate(boundary, dilate_kernel, iterations=1)
            rim_band = (rim_band > 0).astype(np.float32)
            
            rim_bands.append(rim_band)
        
        rim_band_tensor = torch.from_numpy(np.stack(rim_bands)).to(mask.device)
        if mask.dim() == 4:
            rim_band_tensor = rim_band_tensor.unsqueeze(1)  # (B, 1, H, W)
        
        return rim_band_tensor
    
    def _compute_pixel_weights(
        self,
        mask: torch.Tensor,
        rim_band: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pixel-wise weights for loss.
        
        Args:
            mask: GT binary mask (B, 1, H, W)
            rim_band: Rim band mask (B, 1, H, W)
        
        Returns:
            weights: Pixel weights (B, 1, H, W)
        """
        weights = torch.ones_like(mask) * self.bg_weight  # Background default
        
        # Interior pixels (mask=1, rim_band=0)
        interior_mask = (mask > 0.5) & (rim_band < 0.5)
        weights[interior_mask] = 1.0
        
        # Rim pixels (rim_band=1)
        weights[rim_band > 0.5] = self.rim_weight
        
        return weights
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: GT binary mask (B, 1, H, W)
        
        Returns:
            dice_loss: Scalar loss
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice
        
        return dice_loss
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rim-weighted mask loss.
        
        Args:
            pred_logits: Predicted mask logits (B, 1, H, W)
            target_mask: GT binary mask (B, 1, H, W), values in [0, 1]
        
        Returns:
            loss: Scalar loss value
        """
        # Extract rim band from GT mask
        rim_band = self._extract_rim_band(target_mask)
        
        # Compute pixel weights
        pixel_weights = self._compute_pixel_weights(target_mask, rim_band)
        
        # Weighted BCE loss
        bce_per_pixel = self.bce_logits(pred_logits, target_mask)  # (B, 1, H, W)
        weighted_bce = (bce_per_pixel * pixel_weights).mean()
        
        # Dice loss (optional)
        if self.use_dice:
            dice_loss = self._dice_loss(pred_logits, target_mask)
            total_loss = (1.0 - self.dice_weight) * weighted_bce + self.dice_weight * dice_loss
        else:
            total_loss = weighted_bce
        
        return total_loss
