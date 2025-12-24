"""
YOLOv8 Detection Loss (v8DetectionLoss) - implemented from scratch.
Includes CIoU loss, DFL loss, and classification loss with TaskAlignedAssigner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) - boxes in format [x, y, w, h] or [x1, y1, x2, y2]
        box2: (M, 4) - boxes in format [x, y, w, h] or [x1, y1, x2, y2]
        xywh: If True, boxes are in [x, y, w, h] format
        CIoU: If True, use Complete IoU
    
    Returns:
        iou: (N, M) - IoU values
    """
    # Convert to xyxy format
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    
    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    
    return iou


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for object detection.
    Assigns gt to anchors based on the weighted sum of classification and localization scores.
    """
    
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Assign ground truth to predictions.
        
        Args:
            pd_scores: (B, num_anchors, num_classes) - predicted class scores
            pd_bboxes: (B, num_anchors, 4) - predicted boxes in xyxy format
            anc_points: (num_anchors, 2) - anchor points
            gt_labels: (B, max_num_gt) - ground truth class labels
            gt_bboxes: (B, max_num_gt, 4) - ground truth boxes in xyxy format
            mask_gt: (B, max_num_gt) - mask for valid ground truths
        
        Returns:
            target_labels: (B, num_anchors) - assigned class labels
            target_bboxes: (B, num_anchors, 4) - assigned boxes
            target_scores: (B, num_anchors, num_classes) - target scores
            fg_mask: (B, num_anchors) - foreground mask
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        
        if self.n_max_boxes == 0:
            device = pd_scores.device
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes).long(),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0])
            )
        
        # Get positive mask and alignment metrics
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        
        # CRITICAL FIX: Ensure at least one anchor per GT box BEFORE select_highest_overlaps
        # This guarantees positive samples even with poor predictions
        for b in range(self.bs):
            for gt_idx in range(self.n_max_boxes):
                if mask_gt[b, gt_idx]:
                    # Check if any anchors are already assigned to this GT
                    if mask_pos[b, :, gt_idx].sum() == 0:
                        # No anchors assigned - force assignment based on IoU
                        ious = overlaps[b, :, gt_idx]  # (num_anchors,)
                        
                        # Find anchors inside GT box
                        mask_in_gt = self.select_candidates_in_gts(
                            anc_points, gt_bboxes[b:b+1, gt_idx:gt_idx+1]
                        ).squeeze(0).squeeze(-1)  # (num_anchors,)
                        
                        if mask_in_gt.sum() > 0:
                            # Use best IoU among anchors inside GT
                            valid_ious = ious * mask_in_gt.float()
                            best_anchor = valid_ious.argmax()
                        else:
                            # No anchors inside - use best IoU overall (very lenient)
                            best_anchor = ious.argmax()
                        
                        # Force assignment
                        mask_pos[b, best_anchor, gt_idx] = 1.0
        
        # Select topk based on alignment metric (now guaranteed to have positive samples)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        
        # CRITICAL: Force at least one positive sample per GT box
        # This ensures box/DFL losses are ALWAYS computed
        for b in range(self.bs):
            for gt_idx in range(self.n_max_boxes):
                if mask_gt[b, gt_idx]:
                    # Check if any anchors are assigned to this GT in this batch
                    assigned = fg_mask[b] > 0
                    gt_assigned = (target_gt_idx[b][assigned] == gt_idx).any()
                    
                    if not gt_assigned:
                        # No assignment - force assign best IoU anchor
                        ious = overlaps[b, :, gt_idx]
                        best_anchor = ious.argmax()
                        
                        # Create new assignment
                        mask_pos[b, best_anchor, gt_idx] = 1.0
                        # Update target_gt_idx and fg_mask
                        target_gt_idx[b, best_anchor] = gt_idx
                        fg_mask[b, best_anchor] = 1
        
        # Final recompute to ensure consistency
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        
        # Assign targets
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        
        # Normalize alignment metric
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-1)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
    
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask and alignment metrics."""
        # Compute alignment metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        
        # Get in-box mask
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        
        # Combine masks: mask_in_gts is (bs, n_anchors, n_gt), mask_gt is (bs, n_gt)
        # We need to expand mask_gt to match mask_in_gts shape
        mask_pos = mask_in_gts * mask_gt.unsqueeze(1)
        
        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """Calculate alignment metric and overlaps."""
        # IoU between predicted and gt boxes
        overlaps = bbox_iou(pd_bboxes.unsqueeze(2), gt_bboxes.unsqueeze(1), xywh=False, CIoU=False).squeeze(-1).clamp_(0)
        
        # Classification scores for gt classes
        gt_labels_expand = gt_labels.unsqueeze(1).expand(-1, pd_scores.shape[1], -1).long()
        cls_scores = pd_scores.gather(-1, gt_labels_expand)
        
        # Alignment metric = cls_score^alpha * iou^beta
        align_metric = cls_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return align_metric, overlaps
    
    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        """Select candidates that are inside ground truth boxes."""
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        
        # Expand dimensions
        xy = xy_centers.view(1, n_anchors, 1, 2).expand(bs, -1, n_boxes, -1)
        gt = gt_bboxes.view(bs, 1, n_boxes, 4).expand(-1, n_anchors, -1, -1)
        
        # Check if centers are inside boxes
        lt = xy - gt[..., :2]  # left-top
        rb = gt[..., 2:] - xy  # right-bottom
        bbox_deltas = torch.cat((lt, rb), dim=-1)
        
        return bbox_deltas.amin(dim=-1).gt_(eps)
    
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        """Select anchors with highest IoU overlap for each GT."""
        # Get mask with max IoU for each GT
        fg_mask = mask_pos.sum(dim=-1)
        
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(-1) > 1).expand(-1, -1, n_max_boxes)
            max_overlaps_idx = overlaps.argmax(dim=-1)
            
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(-1, max_overlaps_idx.unsqueeze(-1), 1)
            
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(dim=-1)
        
        # Find which GT each anchor is assigned to
        target_gt_idx = mask_pos.argmax(dim=-1)
        
        return target_gt_idx, fg_mask, mask_pos
    
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Get target labels, boxes, and scores."""
        batch_idx = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_idx * self.n_max_boxes
        
        # Get targets
        target_labels = gt_labels.flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # Set background to num_classes
        target_labels.clamp_(0, self.num_classes)
        
        # One-hot encode
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.float32,
            device=target_labels.device
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1).long(), 1)
        
        # Mask out background
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        
        return target_labels, target_bboxes, target_scores.float()


class BboxLoss(nn.Module):
    """Bounding box loss with CIoU and DFL."""
    
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        Calculate bounding box loss.
        
        Args:
            pred_dist: (B, num_anchors, 4*reg_max) - predicted distributions
            pred_bboxes: (B, num_anchors, 4) - predicted boxes
            anchor_points: (num_anchors, 2) - anchor points
            target_bboxes: (B, num_anchors, 4) - target boxes
            target_scores: (B, num_anchors, num_classes) - target scores
            target_scores_sum: () - sum of target scores
            fg_mask: (B, num_anchors) - foreground mask
        
        Returns:
            loss_iou: CIoU loss
            loss_dfl: DFL loss
        """
        # IoU loss
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL loss
        if self.reg_max > 1:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        
        return loss_iou, loss_dfl
    
    @staticmethod
    def _df_loss(pred_dist, target):
        """Distribution Focal Loss."""
        target = target.clamp_(0, pred_dist.shape[-1] - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        
        loss_left = F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl
        loss_right = F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr
        
        return (loss_left + loss_right).mean(-1, keepdim=True)


def bbox2dist(anchor_points, bbox, reg_max):
    """Convert bbox to distribution format."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


class v8DetectionLoss:
    """YOLOv8 Detection Loss."""
    
    def __init__(self, model, tal_topk=10):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.model = model
        self.tal_topk = tal_topk
        
        # Get detect head
        if hasattr(model, 'model') and hasattr(model.model, 'detect'):
            detect = model.model.detect
        elif hasattr(model, 'detect'):
            detect = model.detect
        else:
            # Find Detect module
            for m in model.modules():
                if type(m).__name__ == 'Detect':
                    detect = m
                    break
        
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.stride = detect.stride
        self.device = next(model.parameters()).device
        
        # Loss components
        # CRITICAL FIX: Ultra-lenient assigner to ensure positive samples
        # topk=50: Consider top 50 predictions per GT (very lenient)
        # alpha=0.5: Balance between cls and IoU (lower = more IoU weight)
        # beta=4.0: Lower beta = less penalty for low IoU (was 6.0)
        self.assigner = TaskAlignedAssigner(topk=50, num_classes=self.nc, alpha=0.5, beta=4.0)
        self.bbox_loss = BboxLoss(self.reg_max)
        self.epoch = 0  # Track epoch for warmup
        
        # Loss weights
        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
    
    def __call__(self, preds, batch):
        """
        Calculate loss.
        
        Args:
            preds: List of predictions from each scale
            batch: Batch dict with 'img', 'cls', 'bboxes', 'batch_idx'
        
        Returns:
            loss: Total loss
            loss_items: (box_loss, cls_loss, dfl_loss)
        """
        loss = torch.zeros(3, device=self.device)
        
        # Preprocess predictions
        feats = preds
        if isinstance(feats, tuple):
            feats = feats[1] if len(feats) == 2 else feats
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.nc + self.reg_max * 4, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Targets
        targets = self.preprocess(batch['cls'], batch['bboxes'], batch['batch_idx'], batch_size, imgsz)
        gt_labels, gt_bboxes, mask_gt = targets.split((1, 4, 1), 2)
        mask_gt = mask_gt.gt_(0).squeeze(-1)
        
        # Decode predictions
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, stride_tensor)
        
        # Convert pred_bboxes to pixel coordinates for assignment
        pred_bboxes_pixel = pred_bboxes * stride_tensor
        
        # Assign targets
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_pixel.detach().type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels.squeeze(-1),
            gt_bboxes,
            mask_gt
        )
        
        # CRITICAL FIX: Force positive samples if none found
        # This ensures box/DFL losses are ALWAYS computed
        num_fg = fg_mask.sum().item()
        num_gt = mask_gt.sum().item()
        
        # ALWAYS use fallback if no positive samples (even in eval mode during training)
        if num_fg == 0 and num_gt > 0:
            # Emergency fallback: assign anchors based purely on IoU
            # Both pred_bboxes_pixel and gt_bboxes are in pixel coordinates
            ious = bbox_iou(
                pred_bboxes_pixel.unsqueeze(2),  # (B, num_anchors, 1, 4)
                gt_bboxes.unsqueeze(1),         # (B, 1, n_gt, 4)
                xywh=False, 
                CIoU=False
            ).squeeze(-1)  # (B, num_anchors, n_gt)
            
            # For each GT box, assign best IoU anchor
            fg_mask_new = torch.zeros_like(fg_mask)
            target_bboxes_new = torch.zeros_like(target_bboxes)
            target_labels_new = torch.full_like(target_labels, self.nc)
            
            num_assigned = 0
            for b in range(batch_size):
                for gt_idx in range(gt_bboxes.shape[1]):
                    if mask_gt[b, gt_idx]:
                        # Find anchor with best IoU
                        best_anchor = ious[b, :, gt_idx].argmax()
                        fg_mask_new[b, best_anchor] = 1
                        target_bboxes_new[b, best_anchor] = gt_bboxes[b, gt_idx]
                        target_labels_new[b, best_anchor] = gt_labels[b, gt_idx, 0]
                        num_assigned += 1
            
            # Update targets
            fg_mask = fg_mask_new
            target_bboxes = target_bboxes_new
            target_labels = target_labels_new
            
            # Update target_scores
            target_scores = torch.zeros_like(target_scores)
            for b in range(batch_size):
                for anchor_idx in range(target_labels.shape[1]):
                    if fg_mask[b, anchor_idx]:
                        cls = target_labels[b, anchor_idx].long()
                        if 0 <= cls < self.nc:
                            target_scores[b, anchor_idx, cls] = 1.0
            
            # Fallback assignment complete
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        # Bbox loss - NOW GUARANTEED TO HAVE POSITIVE SAMPLES
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )
        else:
            # Should never reach here after fallback, but just in case
            loss[0] = torch.tensor(0.0, device=self.device)
            loss[2] = torch.tensor(0.0, device=self.device)
        
        # Apply weights
        loss[0] *= self.hyp_box
        loss[1] *= self.hyp_cls
        loss[2] *= self.hyp_dfl
        
        return loss.sum() * batch_size, loss.detach()
    
    def preprocess(self, targets_cls, targets_bbox, batch_idx, batch_size, img_size):
        """Preprocess targets to match expected format."""
        if len(targets_cls) == 0:
            return torch.zeros(batch_size, 0, 6, device=self.device)
        
        # Find max number of objects per image
        max_len = max((batch_idx == i).sum().item() for i in range(batch_size))
        
        # Create output tensor [batch, max_objs, 6] where 6 = [cls, x1, y1, x2, y2, ?]
        out = torch.zeros(batch_size, max_len, 6, device=self.device)
        
        # Scale boxes from normalized [0, 1] to pixel coordinates [0, img_size]
        # CRITICAL FIX: Boxes must be in same space as anchor points!
        # img_size is [W, H], boxes are [x1, y1, x2, y2], so repeat for both x and y
        scale = img_size.repeat(2)[:4]  # [W, H, W, H]
        targets_bbox_scaled = targets_bbox * scale
        
        for i in range(batch_size):
            mask = batch_idx == i
            n = mask.sum()
            if n:
                # Concatenate class (1 col) + bbox (4 cols) + mask (1 col)
                out[i, :n, 0] = targets_cls[mask, 0]  # class
                out[i, :n, 1:5] = targets_bbox_scaled[mask]  # bbox in pixel space
                out[i, :n, 5] = 1.0  # CRITICAL FIX: Mark valid GT boxes with 1.0
        
        return out
    
    def bbox_decode(self, anchor_points, pred_dist, stride_tensor):
        """Decode predicted distribution to bounding boxes."""
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            # Reshape and apply softmax to get distribution
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                torch.arange(c // 4, device=pred_dist.device, dtype=pred_dist.dtype).view(-1, 1)
            ).view(b, a, 4)
        
        return dist2bbox(pred_dist, anchor_points.unsqueeze(0), xywh=False, dim=-1)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

