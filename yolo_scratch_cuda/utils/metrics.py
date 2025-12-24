"""
Evaluation metrics for crater detection.

Includes:
- COCO-style Precision / Recall / mAP (for reference)
- CDA-style proxy metric (cda_proxy) that more closely follows the
  NASA Crater Detection Challenge scorer by emphasizing:
    * Center accuracy
    * Axis ratios
    * Strong FP penalties
"""

import math
import torch
import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    
    Args:
        box1: (4,) tensor [x_min, y_min, x_max, y_max]
        box2: (4,) tensor [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU value
    """
    # Calculate intersection
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])
    
    inter_area = torch.clamp(x2_inter - x1_inter, min=0.0) * torch.clamp(y2_inter - y1_inter, min=0.0)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def calculate_iou_batch(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes (vectorized).
    
    Args:
        boxes1: (N, 4) tensor [x_min, y_min, x_max, y_max]
        boxes2: (M, 4) tensor [x_min, y_min, x_max, y_max]
    
    Returns:
        (N, M) tensor of IoU values
    """
    # Expand dimensions for broadcasting: (N, 1, 4) and (1, M, 4)
    boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)
    
    # Calculate intersection
    x1_inter = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])  # (N, M)
    y1_inter = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])  # (N, M)
    x2_inter = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])  # (N, M)
    y2_inter = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])  # (N, M)
    
    inter_area = torch.clamp(x2_inter - x1_inter, min=0.0) * torch.clamp(y2_inter - y1_inter, min=0.0)
    
    # Calculate union
    boxes1_area = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])  # (N, 1)
    boxes2_area = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])  # (1, M)
    
    union_area = boxes1_area.squeeze(-1).unsqueeze(-1) + boxes2_area.squeeze(0).unsqueeze(0) - inter_area
    
    # Avoid division by zero
    ious = inter_area / (union_area + 1e-6)
    
    return ious


def evaluate_detections(
    predictions: List[Dict],
    targets: List[Dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate detections and calculate metrics.
    
    Args:
        predictions: List of prediction dicts, each with:
            - boxes: (N, 4) tensor [x_min, y_min, x_max, y_max] (normalized)
            - scores: (N,) tensor of confidence scores
            - labels: (N,) tensor of class labels
        targets: List of target dicts, each with:
            - boxes: (M, 4) tensor (normalized)
            - labels: (M,) tensor
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with metrics: precision, recall, mAP50, mAP50_95
    """
    # Collect all detections and ground truth
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_target_boxes = []
    all_target_labels = []
    
    for pred, target in zip(predictions, targets):
        # Filter predictions by confidence
        if len(pred['boxes']) > 0:
            valid = pred['scores'] >= conf_threshold
            all_pred_boxes.append(pred['boxes'][valid])
            all_pred_scores.append(pred['scores'][valid])
            all_pred_labels.append(pred['labels'][valid])
        else:
            all_pred_boxes.append(torch.zeros((0, 4), device=pred['boxes'].device))
            all_pred_scores.append(torch.zeros((0,), device=pred['scores'].device))
            all_pred_labels.append(torch.zeros((0,), dtype=torch.long, device=pred['labels'].device))
        
        all_target_boxes.append(target['boxes'])
        all_target_labels.append(target['labels'])
    
    # Calculate metrics for each IoU threshold
    # For mAP50-95: use standard COCO thresholds [0.5, 0.55, ..., 0.95] (10 thresholds)
    # For mAP50: use single threshold [0.5]
    if iou_threshold == 0.5:
        # Standard COCO mAP50-95: 0.5 to 0.95 with 0.05 step
        iou_thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)
    else:
        # Fallback for other thresholds
        iou_thresholds = np.arange(iou_threshold, min(1.0, iou_threshold + 0.45) + 1e-6, 0.05)
    ap_scores = []
    
    for iou_thresh in iou_thresholds:
        tp_list = []
        fp_list = []
        scores_list = []
        n_gt = 0
        
        for pred_boxes, pred_scores, pred_labels, target_boxes, target_labels in zip(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_target_boxes, all_target_labels
        ):
            n_gt += len(target_boxes)
            
            if len(pred_boxes) == 0:
                continue
            
            if len(target_boxes) == 0:
                # All predictions are false positives
                fp_list.extend(pred_scores.cpu().tolist())
                scores_list.extend(pred_scores.cpu().tolist())
                continue
            
            # Sort by score
            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]
            pred_scores = pred_scores[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            
            # Vectorized IoU calculation for all pairs
            iou_matrix = calculate_iou_batch(pred_boxes, target_boxes)  # (N_pred, N_gt)
            
            # Match predictions to ground truth (greedy matching)
            matched_gt = torch.zeros(len(target_boxes), dtype=torch.bool, device=target_boxes.device)
            
            for i in range(len(pred_boxes)):
                # Find best unmatched GT box
                available_mask = ~matched_gt
                if not available_mask.any():
                    # All GT boxes matched, rest are FPs
                    fp_list.append(pred_scores[i].item())
                    scores_list.append(pred_scores[i].item())
                    continue
                
                # Get IoUs with available GT boxes
                available_ious = iou_matrix[i, available_mask]
                if len(available_ious) == 0:
                    fp_list.append(pred_scores[i].item())
                    scores_list.append(pred_scores[i].item())
                    continue
                
                # Find best match among available
                best_iou = available_ious.max().item()
                best_gt_idx_relative = available_ious.argmax().item()
                best_gt_idx = torch.where(available_mask)[0][best_gt_idx_relative].item()
                
                scores_list.append(pred_scores[i].item())
                
                if best_iou >= iou_thresh:
                    tp_list.append(pred_scores[i].item())
                    matched_gt[best_gt_idx] = True
                else:
                    fp_list.append(pred_scores[i].item())
        
        # Calculate AP at this IoU threshold
        ap = calculate_ap(tp_list, fp_list, n_gt)
        ap_scores.append(ap)
    
    # Calculate Precision and Recall at the base IoU threshold
    tp_list_50 = []
    fp_list_50 = []
    n_gt_50 = 0
    
    for pred_boxes, pred_scores, pred_labels, target_boxes, target_labels in zip(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_target_boxes, all_target_labels
    ):
        n_gt_50 += len(target_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        if len(target_boxes) == 0:
            fp_list_50.extend(pred_scores.cpu().tolist())
            continue
        
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Vectorized IoU calculation for all pairs
        iou_matrix = calculate_iou_batch(pred_boxes, target_boxes)  # (N_pred, N_gt)
        
        matched_gt = torch.zeros(len(target_boxes), dtype=torch.bool, device=target_boxes.device)
        
        for i in range(len(pred_boxes)):
            # Find best unmatched GT box
            available_mask = ~matched_gt
            if not available_mask.any():
                # All GT boxes matched, rest are FPs
                fp_list_50.append(pred_scores[i].item())
                continue
            
            # Get IoUs with available GT boxes
            available_ious = iou_matrix[i, available_mask]
            if len(available_ious) == 0:
                fp_list_50.append(pred_scores[i].item())
                continue
            
            # Find best match among available
            best_iou = available_ious.max().item()
            best_gt_idx_relative = available_ious.argmax().item()
            best_gt_idx = torch.where(available_mask)[0][best_gt_idx_relative].item()
            
            if best_iou >= 0.5:
                tp_list_50.append(pred_scores[i].item())
                matched_gt[best_gt_idx] = True
            else:
                fp_list_50.append(pred_scores[i].item())
    
    n_tp = len(tp_list_50)
    n_fp = len(fp_list_50)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / n_gt_50 if n_gt_50 > 0 else 0.0
    
    # mAP50 is AP at IoU=0.5
    mAP50 = ap_scores[0] if len(ap_scores) > 0 else 0.0
    
    # mAP50-95 is average of all APs
    mAP50_95 = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'mAP50-95': mAP50_95
    }


def calculate_ap(tp: List[float], fp: List[float], n_gt: int) -> float:
    """
    Calculate Average Precision (AP) using all-point interpolation.
    
    Args:
        tp: List of true positive scores
        fp: List of false positive scores
        n_gt: Number of ground truth boxes
    
    Returns:
        AP value
    """
    if n_gt == 0:
        return 0.0
    
    # Combine and sort by score
    tp_scores = [(score, 1) for score in tp]
    fp_scores = [(score, 0) for score in fp]
    all_detections = sorted(tp_scores + fp_scores, key=lambda x: x[0], reverse=True)
    
    if len(all_detections) == 0:
        return 0.0
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum([det[1] for det in all_detections])
    fp_cumsum = np.cumsum([1 - det[1] for det in all_detections])
    
    # Calculate precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / n_gt
    
    # Calculate AP using all-point interpolation (COCO style)
    # Append sentinel values
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    
    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_cda_proxy(
    predictions: List[Dict],
    targets: List[Dict],
) -> Dict[str, float]:
    """
    Legacy CDA-style proxy metric (kept for comparison and ablation).
    New code should prefer `evaluate_cda_full`, which implements the
    official CDAquality formulation.
    """
    max_matches_per_image = 10
    
    total_center_err = 0.0
    total_ratio_err = 0.0
    total_matches = 0
    total_fp = 0
    num_images = len(predictions)
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        
        if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
            continue
        
        # Sort predictions by descending score and cap to top-K
        if pred_boxes.numel() > 0:
            order = torch.argsort(pred_scores, descending=True)
            if len(order) > max_matches_per_image:
                order = order[:max_matches_per_image]
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
        
        # If no GT, all predictions are FP
        if gt_boxes.numel() == 0:
            total_fp += len(pred_boxes)
            continue
        
        # Compute centers and widths/heights (normalized coords)
        # GT
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
        gt_cx = 0.5 * (gt_xmin + gt_xmax)
        gt_cy = 0.5 * (gt_ymin + gt_ymax)
        gt_w = gt_xmax - gt_xmin
        gt_h = gt_ymax - gt_ymin
        gt_ratio = gt_w / (gt_h + 1e-6)
        
        # Predictions
        if pred_boxes.numel() > 0:
            pr_xmin, pr_ymin, pr_xmax, pr_ymax = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
            pr_cx = 0.5 * (pr_xmin + pr_xmax)
            pr_cy = 0.5 * (pr_ymin + pr_ymax)
            pr_w = pr_xmax - pr_xmin
            pr_h = pr_ymax - pr_ymin
            pr_ratio = pr_w / (pr_h + 1e-6)
        else:
            # No predictions: all GT are missed; include no center/ratio signal, but no FP either.
            continue
        
        # Image diagonal in normalized space (sqrt(2) for [0,1]x[0,1])
        diag = np.sqrt(2.0)
        
        num_gt = gt_boxes.shape[0]
        matched_gt = torch.zeros(num_gt, dtype=torch.bool, device=gt_boxes.device)
        
        # For each prediction, match to closest GT center (greedy)
        for i in range(pred_boxes.shape[0]):
            # Compute squared center distances to all GT
            dx = pr_cx[i] - gt_cx
            dy = pr_cy[i] - gt_cy
            dist_sq = dx * dx + dy * dy
            
            # Mask out already matched GTs
            dist_sq_masked = dist_sq.clone()
            dist_sq_masked[matched_gt] = 1e9
            best_idx = torch.argmin(dist_sq_masked).item()
            
            if matched_gt[best_idx]:
                # All GT considered for this pred are taken -> FP
                total_fp += 1
                continue
            
            # Treat this as a match; accumulate errors
            matched_gt[best_idx] = True
            center_dist = float(torch.sqrt(dist_sq[best_idx]).item()) / diag  # normalized center error
            ratio_err = float(torch.abs(pr_ratio[i] - gt_ratio[best_idx]).item())
            
            total_center_err += center_dist
            total_ratio_err += ratio_err
            total_matches += 1
        
        # Any remaining unmatched predictions are FPs
        if pred_boxes.shape[0] > 0:
            # Count predictions that did not get a unique GT assignment
            matched_preds = int(matched_gt.sum().item())
            total_fp += max(0, pred_boxes.shape[0] - matched_preds)
    
    if num_images == 0:
        return {'cda_proxy': 0.0, 'center_error': 0.0, 'fp_per_image': 0.0}
    
    avg_center_err = total_center_err / max(total_matches, 1)
    fp_per_image = total_fp / num_images
    
    # CDA-style proxy: higher is better, 0-1-ish range.
    cda_proxy = np.exp(-3.0 * avg_center_err) * np.exp(-2.0 * fp_per_image)
    cda_proxy = float(np.clip(cda_proxy, 0.0, 1.0))
    
    return {
        'cda_proxy': cda_proxy,
        'center_error': float(avg_center_err),
        'fp_per_image': float(fp_per_image),
    }


def _calcYmat(a: float, b: float, phi: float) -> np.ndarray:
    """
    Construct the Y matrix for an ellipse parameterized by
    semimajor a, semiminor b, and orientation phi (radians).
    
    This follows the official scorer's definition used for
    Gaussian angle (dGA) computation.
    """
    c = math.cos(phi)
    s = math.sin(phi)
    unit_1 = np.array([[c, -s], [s, c]])
    unit_2 = np.array([[1.0 / (a ** 2), 0.0], [0.0, 1.0 / (b ** 2)]])
    unit_3 = np.array([[c, s], [-s, c]])
    return unit_1 @ unit_2 @ unit_3


def _calc_dGA(Yi: np.ndarray, Yj: np.ndarray, yi: np.ndarray, yj: np.ndarray) -> float:
    """
    Core Gaussian angle (dGA) computation between two ellipses,
    as in the official scorer:
        multiplicand = 4 * sqrt(det(Yi)*det(Yj)) / det(Yi + Yj)
        exponent = -0.5 * (yi - yj)^T Yi (Yi + Yj)^(-1) Yj (yi - yj)
        cos_term = multiplicand * exp(exponent)
        dGA = arccos(cos_term)
    """
    det_Yi = np.linalg.det(Yi)
    det_Yj = np.linalg.det(Yj)
    Y_sum = Yi + Yj
    det_sum = np.linalg.det(Y_sum)
    if det_sum <= 0 or det_Yi <= 0 or det_Yj <= 0:
        return math.pi / 2.0  # fallback: worst-case angle
    multiplicand = 4.0 * math.sqrt(det_Yi * det_Yj) / det_sum
    diff = yi - yj  # (2,1)
    try:
        inv_sum = np.linalg.inv(Y_sum)
    except np.linalg.LinAlgError:
        return math.pi / 2.0
    exponent_mat = (-0.5 * diff.T @ Yi @ inv_sum @ Yj @ diff)
    e = exponent_mat[0, 0]
    cos_term = multiplicand * math.exp(e)
    cos_term = min(1.0, max(-1.0, float(cos_term)))
    return float(math.acos(cos_term))


def cda_nms(
    predictions: Dict[str, torch.Tensor],
    max_detections: int = 10,
    dga_thresh: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    CDA-aware Non-Maximum Suppression using Gaussian angle (dGA) instead of IoU.
    
    This function suppresses duplicate crater detections based on their Gaussian
    angle similarity, which aligns with the official CDA metric used in the
    NASA Crater Detection Challenge. Unlike IoU-based NMS, dGA considers both
    spatial overlap and ellipse shape similarity, making it more appropriate for
    crater detection where center accuracy and axis ratios matter.
    
    Why dGA instead of IoU?
    - The CDA metric uses Gaussian angle (dGA) to measure similarity between
      ellipses, considering both center distance and ellipse shape (a, b, angle).
    - IoU only measures bounding box overlap, which doesn't capture the elliptical
      nature of craters or the importance of center accuracy in CDA scoring.
    - Using dGA for NMS ensures that suppression decisions align with how the
      final CDAquality score is computed, reducing false positives that would
      otherwise pass IoU-based filtering.
    
    Args:
        predictions: Dict with keys:
            - boxes: Tensor[N, 4] (xmin, ymin, xmax, ymax) normalized [0,1]
            - scores: Tensor[N] confidence scores
            - angles: Tensor[N] (optional) ellipse angles in radians (default=0)
        max_detections: Maximum number of detections to keep (default=10, matching
                        CDA scorer's top-10 limit)
        dga_thresh: Gaussian angle threshold for suppression (default=0.15 radians)
                    Two detections with dGA < dga_thresh are considered duplicates.
    
    Returns:
        Dict with same structure as input, but with duplicates suppressed:
            - boxes: Tensor[M, 4] where M <= max_detections
            - scores: Tensor[M]
            - angles: Tensor[M] (if provided in input)
            - labels: Tensor[M] (if provided in input)
    """
    boxes = predictions['boxes']  # (N, 4)
    scores = predictions['scores']  # (N,)
    angles = predictions.get('angles', torch.zeros(len(boxes), device=boxes.device))  # (N,)
    labels = predictions.get('labels', None)  # (N,) optional
    
    if len(boxes) == 0:
        result = {
            'boxes': boxes,
            'scores': scores,
            'angles': angles,
        }
        if labels is not None:
            result['labels'] = labels
        return result
    
    # Convert boxes to ellipse parameters
    # Box format: [xmin, ymin, xmax, ymax] normalized [0,1]
    # Ellipse: center (xc, yc), semimajor a, semiminor b, angle phi
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x_max - x_min  # width
    h = y_max - y_min  # height
    
    # Convert to ellipse parameters (a, b are half-width/height)
    # For craters, we treat width/height as ellipse axes
    # Ensure semimajor >= semiminor (a >= b)
    half_w = w / 2.0
    half_h = h / 2.0
    a = torch.max(half_w, half_h)  # semimajor axis (larger of half-width/half-height)
    b = torch.min(half_w, half_h)  # semiminor axis (smaller of half-width/half-height)
    xc = (x_min + x_max) / 2.0  # center x
    yc = (y_min + y_max) / 2.0  # center y
    # Angle: if width > height, ellipse is horizontal (phi=0), else vertical (phi=pi/2)
    # For simplicity, use predicted angle or 0 (can be refined if model predicts angles)
    phi = angles  # orientation angle (radians)
    
    # Sort by descending score
    sorted_indices = torch.argsort(scores, descending=True)
    
    kept_indices = []
    
    for idx in sorted_indices:
        if len(kept_indices) >= max_detections:
            break
        
        candidate_idx = idx.item()
        
        # Get candidate ellipse parameters
        cand_a = float(a[candidate_idx].item())
        cand_b = float(b[candidate_idx].item())
        cand_xc = float(xc[candidate_idx].item())
        cand_yc = float(yc[candidate_idx].item())
        cand_phi = float(phi[candidate_idx].item())
        
        # Check against all kept detections
        is_duplicate = False
        
        for kept_idx in kept_indices:
            # Get kept ellipse parameters
            kept_a = float(a[kept_idx].item())
            kept_b = float(b[kept_idx].item())
            kept_xc = float(xc[kept_idx].item())
            kept_yc = float(yc[kept_idx].item())
            kept_phi = float(phi[kept_idx].item())
            
            # Compute Y matrices for dGA calculation
            cand_Y = _calcYmat(cand_a, cand_b, cand_phi)
            kept_Y = _calcYmat(kept_a, kept_b, kept_phi)
            
            # Center vectors
            cand_center = np.array([[cand_xc], [cand_yc]])
            kept_center = np.array([[kept_xc], [kept_yc]])
            
            # Compute dGA
            dga = _calc_dGA(cand_Y, kept_Y, cand_center, kept_center)
            
            # If dGA is below threshold, this is a duplicate
            if dga < dga_thresh:
                is_duplicate = True
                break
        
        # If not a duplicate, keep it
        if not is_duplicate:
            kept_indices.append(candidate_idx)
    
    # Build result dict
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=boxes.device)
    
    result = {
        'boxes': boxes[kept_tensor],
        'scores': scores[kept_tensor],
        'angles': angles[kept_tensor],
    }
    
    if labels is not None:
        result['labels'] = labels[kept_tensor]
    
    return result


def evaluate_cda_full(
    predictions: List[Dict],
    targets: List[Dict],
) -> Dict[str, float]:
    """
    Full CDAquality implementation (image-averaged), closely following
    the official offline scorer from the NASA Crater Detection Challenge.
    
    For each image:
      - At most top-10 predictions (by confidence) are considered.
      - Each GT ellipse is matched to at most one detection based on
        minimum Gaussian angle (dGA).
      - Similarity is decided via a chi^2 test using:
            chi2_thresh^2 = 13.277
            pixel_error = 0.07 * Ab
            sigma_ref   = 0.85 / sqrt(Aa * Ab) * pixel_error
            chi2        = (dGA^2) / (sigma_ref^2)
        If chi2 >= chi2_thresh^2, the pair is rejected.
      - Each accepted match contributes partial credit:
            credit = 1 - dGA / pi
      - CDAquality per image is:
            CDAquality = (sum_credit / (TP + FP)) *
                         min(1, TP / min(N, 10))
        where N is #GT ellipses, TP and FP are true/false positives.
      - Special case N=0:
            CDAquality = 1 if FP == 0 else 0
    
    Inputs:
      predictions: list of dicts with keys:
          - 'boxes': (N, 4) [x_min, y_min, x_max, y_max] in normalized [0,1]
          - 'scores': (N,)
          - 'angles': (N,) in radians (optional; defaults to 0)
      targets: list of dicts with keys:
          - 'boxes': (M, 4)
          - 'angles': (M,) in radians (optional; defaults to 0)
    
    Returns (dataset-level aggregates):
      - 'cda_proxy': mean CDAquality over images (primary score)
      - 'center_error': mean normalized dGA/pi over matched TPs
      - 'fp_per_image': mean number of FPs per image
    """
    XI2_THRESH = 13.277
    NN_PIX_ERR_RATIO = 0.07
    max_preds_per_image = 10
    
    num_images = len(predictions)
    if num_images == 0:
        return {'cda_proxy': 0.0, 'center_error': 0.0, 'fp_per_image': 0.0}
    
    image_scores: List[float] = []
    total_tp = 0
    total_fp = 0
    total_norm_dga = 0.0  # accumulate dGA/pi over all TPs
    
    for pred, target in zip(predictions, targets):
        # Extract predictions
        boxes_p = pred['boxes']
        scores_p = pred['scores']
        angles_p = pred.get('angles', None)
        
        if angles_p is None:
            angles_p_np = None
        else:
            angles_p_np = angles_p.detach().cpu().numpy()
        
        boxes_t = target['boxes']
        angles_t = target.get('angles', None)
        if angles_t is None:
            angles_t_np = None
        else:
            angles_t_np = angles_t.detach().cpu().numpy()
        
        # Handle N=0 case (no GT craters)
        if boxes_t.numel() == 0:
            if boxes_p.numel() == 0:
                # No GT and no predictions -> perfect CDAquality for this image
                image_scores.append(1.0)
                continue
            else:
                # No GT but detections present -> CDAquality = 0
                image_scores.append(0.0)
                total_fp += int(boxes_p.shape[0])
                continue
        
        # Convert to numpy (normalized coordinates)
        boxes_p_np = boxes_p.detach().cpu().numpy()
        scores_p_np = scores_p.detach().cpu().numpy()
        boxes_t_np = boxes_t.detach().cpu().numpy()
        
        # Build crater representations
        pred_craters: List[Dict] = []
        if boxes_p_np.shape[0] > 0:
            order = np.argsort(-scores_p_np)
            if len(order) > max_preds_per_image:
                order = order[:max_preds_per_image]
            for idx in order:
                x_min, y_min, x_max, y_max = boxes_p_np[idx]
                w = max(float(x_max - x_min), 1e-6)
                h = max(float(y_max - y_min), 1e-6)
                a = 0.5 * w
                b = 0.5 * h
                xc = 0.5 * (float(x_min) + float(x_max))
                yc = 0.5 * (float(y_min) + float(y_max))
                if angles_p_np is None:
                    phi = 0.0
                else:
                    phi = float(angles_p_np[idx])
                pred_craters.append({
                    'a': a,
                    'b': b,
                    'xc': xc,
                    'yc': yc,
                    'phi': phi,
                    'matched': False,
                })
        
        gt_craters: List[Dict] = []
        for j in range(boxes_t_np.shape[0]):
            x_min, y_min, x_max, y_max = boxes_t_np[j]
            w = max(float(x_max - x_min), 1e-6)
            h = max(float(y_max - y_min), 1e-6)
            a = 0.5 * w
            b = 0.5 * h
            xc = 0.5 * (float(x_min) + float(x_max))
            yc = 0.5 * (float(y_min) + float(y_max))
            if angles_t_np is None:
                phi = 0.0
            else:
                phi = float(angles_t_np[j])
            gt_craters.append({
                'a': a,
                'b': b,
                'xc': xc,
                'yc': yc,
                'phi': phi,
                'matched': False,
            })
        
        N = len(gt_craters)
        if N == 0:
            # Already handled above, but keep guard
            image_scores.append(1.0 if len(pred_craters) == 0 else 0.0)
            continue
        
        # Per-image matching and CDAquality score
        dga_credits: List[float] = []   # 1 - dGA/pi per TP
        dga_norms: List[float] = []     # dGA/pi per TP
        
        for t in gt_craters:
            # Find best matching prediction for this GT (as in scorer)
            best_p = None
            best_dGA = math.pi / 2.0
            best_xi2 = float('inf')
            
            for p in pred_craters:
                if p['matched']:
                    continue
                # Cheap radius and center-distance filters (same spirit as scorer).
                rA = min(t['a'], t['b'])
                rB = min(p['a'], p['b'])
                if rA > 1.5 * rB or rB > 1.5 * rA:
                    continue
                r = min(rA, rB)
                if abs(t['xc'] - p['xc']) > r:
                    continue
                if abs(t['yc'] - p['yc']) > r:
                    continue
                
                # Build Y matrices and centers for dGA computation.
                A_Y = _calcYmat(t['a'], t['b'], t['phi'])
                B_Y = _calcYmat(p['a'], p['b'], p['phi'])
                A_y = np.array([[t['xc']], [t['yc']]])
                B_y = np.array([[p['xc']], [p['yc']]])
                
                d = _calc_dGA(A_Y, B_Y, A_y, B_y)
                
                # sigma_ref based on GT ellipse (Aa, Ab are semimajor/semiminor).
                Aa = max(t['a'], t['b'])
                Ab = min(t['a'], t['b'])
                comparison_sig = NN_PIX_ERR_RATIO * Ab
                ref_sig = 0.85 / math.sqrt(Aa * Ab + 1e-12) * comparison_sig
                xi2 = (d * d) / (ref_sig * ref_sig + 1e-12)
                
                if d < best_dGA:
                    best_dGA = d
                    best_p = p
                    best_xi2 = xi2
            
            # Chi^2 similarity check (official threshold)
            if best_p is not None and best_xi2 < XI2_THRESH:
                t['matched'] = True
                best_p['matched'] = True
                credit = 1.0 - best_dGA / math.pi   # partial credit for this TP
                dga_credits.append(credit)
                dga_norms.append(best_dGA / math.pi)
        
        if len(dga_credits) == 0:
            # No matches -> CDAquality = 0; all predictions (if any) are FPs.
            image_scores.append(0.0)
            total_fp += len(pred_craters)
            continue
        
        TP = len(dga_credits)
        FP = max(0, len(pred_craters) - TP)
        sum_credit = float(sum(dga_credits))
        
        # CDAquality as per problem statement.
        cdaq = (sum_credit / (TP + FP)) * min(1.0, TP / min(N, 10))
        image_scores.append(cdaq)
        
        total_tp += TP
        total_fp += FP
        total_norm_dga += float(sum(dga_norms))
    
    mean_cda = float(np.mean(image_scores)) if image_scores else 0.0
    mean_tp = total_tp / num_images
    fp_per_image = total_fp / num_images
    center_error = total_norm_dga / max(total_tp, 1)
    
    return {
        'cda_proxy': mean_cda,          # primary CDAquality-like score
        'center_error': float(center_error),
        'fp_per_image': float(fp_per_image),
    }


