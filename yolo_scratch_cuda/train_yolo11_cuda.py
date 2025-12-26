"""
Train YOLOv11 model for crater detection - CUDA/Colab Optimized Version.
Optimized for Tesla T4 (16GB VRAM) and Colab (15GB RAM).
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from tqdm import tqdm
import math
import gc  # For garbage collection

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.yolo11_model import build_yolo11
from losses.yolo_v8_loss import v8DetectionLoss
from datasets.crater_dataset_cuda import CraterDatasetCUDA, collate_fn_cuda
from utils.metrics import evaluate_detections


# Use optimized collate function from dataset module
collate_fn = collate_fn_cuda


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, epochs):
    """Train for one epoch with CUDA optimizations."""
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}')
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device efficiently (non-blocking for CUDA)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Zero gradients (set_to_none=True is faster)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        preds = model(batch['img'])
        
        # Calculate loss
        loss, loss_items = criterion(preds, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Update weights
        optimizer.step()
        
        # Detach and move to CPU for logging (reduces GPU memory)
        total_loss += loss.detach().cpu().item()
        total_box_loss += loss_items[0].detach().cpu().item()
        total_cls_loss += loss_items[1].detach().cpu().item()
        total_dfl_loss += loss_items[2].detach().cpu().item()
        
        # Explicitly delete batch data to free memory immediately
        del batch
        del preds
        del loss
        del loss_items
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'box': f'{total_box_loss/(batch_idx+1):.4f}',
            'cls': f'{total_cls_loss/(batch_idx+1):.4f}',
            'dfl': f'{total_dfl_loss/(batch_idx+1):.4f}'
        })
        
        # Memory cleanup for CUDA (less frequent than MPS)
        if device.type == 'cuda':
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if (batch_idx + 1) % 100 == 0:
                gc.collect()
    
    return {
        'loss': total_loss / len(dataloader),
        'box_loss': total_box_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'dfl_loss': total_dfl_loss / len(dataloader)
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, conf_threshold=0.25, iou_threshold=0.45, max_batches=None):
    """
    Validate model and calculate metrics.

    Args:
        max_batches: Limit validation to first N batches (saves memory)
    """
    model.eval()
    print(f"DEBUG: Model in eval mode: {not model.training}")
    
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0
    
    # Collect predictions and targets for metrics
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Validating')
    
    batches_processed = 0
    for batch in pbar:
        # Move to device (non-blocking for CUDA)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Forward pass
        preds = model(batch['img'])
        
        # Calculate loss
        loss, loss_items = criterion(preds, batch)
        
        # Accumulate losses
        total_loss += loss.item()
        total_box_loss += loss_items[0].item()
        total_cls_loss += loss_items[1].item()
        total_dfl_loss += loss_items[2].item()
        
        # Debug: Print prediction shapes
        if batches_processed == 0:  # Only print for first batch
            print(f"DEBUG: Model training mode: {model.training}")
            print(f"DEBUG: Predictions type: {type(preds)}")
            if isinstance(preds, list):
                print(f"DEBUG: Predictions is list with {len(preds)} elements")
                for i, p in enumerate(preds):
                    print(f"  Element {i} shape: {p.shape if hasattr(p, 'shape') else 'no shape'}")
            elif isinstance(preds, tuple):
                print(f"DEBUG: Predictions is tuple with {len(preds)} elements")
                for i, p in enumerate(preds):
                    print(f"  Element {i} shape: {p.shape if hasattr(p, 'shape') else 'no shape'}")
            else:
                print(f"DEBUG: Predictions shape: {preds.shape if hasattr(preds, 'shape') else 'no shape'}")

        # Decode predictions for metrics
        # preds is either: raw features (list) in training mode, or (decoded, raw_features) in eval mode
        if isinstance(preds, tuple) and len(preds) == 2:
            # Eval mode: (decoded_predictions, raw_features)
            # Use the already decoded predictions
            predictions = decode_predictions_for_metrics(preds[0], batch['img'].shape[-1], conf_threshold, iou_threshold, device)
        elif isinstance(preds, list):
            # Training mode: raw features (shouldn't happen in eval, but handle it)
            try:
                # Force inference decoding
                with torch.no_grad():
                    decoded_preds = model(batch['img'])  # This should return (decoded, raw)
                    if isinstance(decoded_preds, tuple):
                        predictions = decode_predictions_for_metrics(decoded_preds[0], batch['img'].shape[-1], conf_threshold, iou_threshold, device)
                    else:
                        # Fallback: empty predictions
                        predictions = [{
                            'boxes': torch.zeros((0, 4), device=device),
                            'scores': torch.zeros((0,), device=device),
                            'labels': torch.zeros((0,), dtype=torch.long, device=device)
                        }] * batch['img'].shape[0]
            except Exception as e:
                print(f"ERROR: Failed to decode predictions: {e}")
                predictions = [{
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device)
                }] * batch['img'].shape[0]
                continue
        else:
            # Unexpected format
            print(f"ERROR: Unexpected prediction format: {type(preds)}")
            predictions = [{
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            }] * batch['img'].shape[0]
            continue
        all_predictions.extend(predictions)
        
        # Prepare targets (boxes are already normalized in xyxy format from collate_fn)
        batch_size = batch['img'].shape[0]
        batch_idx_tensor = batch['batch_idx']
        bboxes = batch['bboxes']
        cls = batch['cls']
        
        # Group boxes by batch index
        for i in range(batch_size):
            mask = batch_idx_tensor == i
            if mask.sum() > 0:
                target = {
                    'boxes': bboxes[mask].to(device),  # (M, 4) normalized xyxy
                    'labels': cls[mask].squeeze(-1).to(device)  # (M,)
                }
            else:
                target = {
                    'boxes': torch.zeros((0, 4), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device)
                }
            all_targets.append(target)
        
        # Free batch memory immediately after processing
        del batch
        del preds
        del loss
        del loss_items
        del predictions
        
        # Memory cleanup for CUDA
        if device.type == 'cuda':
            if (batches_processed + 1) % 20 == 0:
                torch.cuda.empty_cache()
            if (batches_processed + 1) % 50 == 0:
                gc.collect()
        
        batches_processed += 1
        if max_batches is not None and batches_processed >= max_batches:
            break
    
    # Move predictions and targets to CPU for metrics calculation
    for pred in all_predictions:
        pred['boxes'] = pred['boxes'].cpu()
        pred['scores'] = pred['scores'].cpu()
        pred['labels'] = pred['labels'].cpu()
    
    for target in all_targets:
        target['boxes'] = target['boxes'].cpu()
        target['labels'] = target['labels'].cpu()
    
    # Calculate metrics
    metrics = evaluate_detections(
        all_predictions,
        all_targets,
        conf_threshold=conf_threshold,
        iou_threshold=0.5  # Use 0.5 for mAP50 calculation
    )
    
    num_batches = batches_processed if max_batches else len(dataloader)
    
    return {
        'loss': total_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'dfl_loss': total_dfl_loss / num_batches,
        **metrics  # Add precision, recall, mAP50, mAP50-95
    }


def decode_predictions_for_metrics(preds, img_size, conf_threshold, iou_threshold, device):
    """
    Decode YOLO11 predictions to boxes, scores, labels format for metrics.

    Args:
        preds: Model inference output (decoded_predictions, raw_features)
            decoded_predictions shape: (batch, num_anchors, 4 + nc)
            boxes are in xywh format (pixel coordinates), scores are sigmoided
        img_size: Image size (assumed square)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device

    Returns:
        List of prediction dicts with 'boxes', 'scores', 'labels' (normalized xyxy)
    """
    predictions = []

    # Extract decoded predictions from inference output
    if isinstance(preds, tuple) and len(preds) == 2:
        pred, _ = preds  # (decoded_output, raw_features)
    else:
        pred = preds

    # pred shape: (batch, num_anchors, 4 + nc)
    # Format: [x, y, w, h, class_score_0, class_score_1, ..., class_score_nc-1]
    # All values are in pixel coordinates, scores are sigmoided (0-1)
    batch_size = pred.shape[0]
    num_anchors = pred.shape[1]
    nc = pred.shape[2] - 4  # number of classes

    for b in range(batch_size):
        pred_batch = pred[b]  # (num_anchors, 4 + nc)

        # Split boxes and class scores
        boxes_xywh = pred_batch[:, :4]  # (N, 4) in xywh format (pixel coordinates)
        class_scores = pred_batch[:, 4:]  # (N, nc) sigmoided class scores (0-1)

        # Get max class scores and indices
        max_scores, class_ids = class_scores.max(dim=1)

        # Filter by confidence threshold
        mask = max_scores > conf_threshold

        if mask.sum() == 0:
            # No detections
            predictions.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            })
            continue

        # Apply mask
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Convert xywh to xyxy format
        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # Apply NMS
        keep_indices = nms_simple(boxes_xyxy, max_scores, iou_threshold)

        if len(keep_indices) == 0:
            predictions.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            })
            continue

        boxes_keep = boxes_xyxy[keep_indices]
        scores_keep = max_scores[keep_indices]
        labels_keep = class_ids[keep_indices]

        # Normalize boxes to [0, 1] (boxes are in pixel coordinates, img_size x img_size)
        boxes_normalized = boxes_keep / img_size

        # Clamp to [0, 1]
        boxes_normalized = torch.clamp(boxes_normalized, 0.0, 1.0)

        predictions.append({
            'boxes': boxes_normalized,
            'scores': scores_keep,
            'labels': labels_keep
        })

    return predictions


def nms_simple(boxes, scores, iou_threshold):
    """
    Simple NMS implementation.
    
    Args:
        boxes: (N, 4) tensor in xyxy format
        scores: (N,) tensor of scores
        iou_threshold: IoU threshold
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = calculate_iou_batch_simple(current_box, remaining_boxes)
        
        # Remove boxes with IoU > threshold
        if ious.dim() > 1:
            ious = ious.squeeze(0)
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep


def calculate_iou_batch_simple(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: (1, 4) or (N, 4) tensor in xyxy format
        boxes2: (M, 4) tensor in xyxy format
        
    Returns:
        (N, M) or (M,) tensor of IoU values
    """
    # Expand for broadcasting
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    
    # Calculate intersection
    x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Handle broadcasting
    if box1_area.shape[0] == 1:
        box1_area = box1_area.unsqueeze(1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou.squeeze() if iou.dim() > 1 and iou.shape[0] == 1 else iou


def cosine_lr_schedule(optimizer, epoch, epochs, lr_min=1e-6, lr_max=1e-3, warmup_epochs=3):
    """Cosine learning rate schedule with warmup."""
    if epoch < warmup_epochs:
        lr = lr_min + (lr_max - lr_min) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 for Crater Detection (CUDA/Colab Optimized)')
    parser.add_argument('--data', type=str, default='/content/data/train', help='Path to training data')
    parser.add_argument('--cfg', type=str, default='configs/yolo11n_crater.yaml', help='Model config file')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='Model scale (n=2.6M, s=9.4M, m=20M, l=25M, x=57M)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (optimized for T4)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda, cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers (CUDA can use multiprocessing)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/YOLO11_crater_cuda', help='Save directory')
    parser.add_argument('--resume', type=str, default='/content/drive/MyDrive/YOLO11_crater_cuda/last.pt', help='Resume from checkpoint')
    parser.add_argument('--max-val-batches', type=int, default=None, help='Limit validation to N batches (saves memory)')
    parser.add_argument('--val-conf', type=float, default=0.25, help='Validation confidence threshold (lower for early training)')
     
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset (optimized for CUDA with Ultralytics-style buffer)
    print(f"Loading dataset from {args.data}...")
    dataset = CraterDatasetCUDA(args.data, img_size=args.imgsz, cache_images=False, augment=True)
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # Create indices for split
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Use Subset to avoid duplicating dataset
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Optimize DataLoader settings for CUDA
    # CUDA can use multiprocessing efficiently
    num_workers = min(args.workers, 4)  # Limit to 4 for Colab
    persistent_workers = True if num_workers > 0 else False  # Keep workers alive for efficiency
    pin_memory = True if device.type == 'cuda' else False  # Faster CPU-GPU transfer
    prefetch_factor = 2 if num_workers > 0 else None  # Prefetch 2 batches ahead
    
    print(f"DataLoader config: workers={num_workers}, pin_memory={pin_memory}, "
          f"persistent={persistent_workers}, prefetch={prefetch_factor}")
    print(f"✓ CUDA optimization: Using multiprocessing and pinned memory for faster training")
    
    # Create dataloaders with CUDA-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,  # Enable for CUDA
        persistent_workers=persistent_workers,  # Keep workers alive
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    # Build model (override scale in config)
    print(f"Building model from {args.cfg} with scale '{args.scale}'...")
    
    # Load config and set scale
    import yaml
    with open(args.cfg, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['scale'] = args.scale
    
    model = build_yolo11(cfg=cfg_dict, ch=1, nc=5)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Setup loss
    criterion = v8DetectionLoss(model, tal_topk=10)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    best_mAP50 = 0.0
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', best_loss)
        best_mAP50 = ckpt.get('best_mAP50', best_mAP50)
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}, best_mAP50={best_mAP50:.4f}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        current_lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            lr_min=args.lr * 0.01, lr_max=args.lr,
            warmup_epochs=3
        )
        
        # Update loss function epoch (for warmup/lenient assignment)
        if hasattr(criterion, 'epoch'):
            criterion.epoch = epoch
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1, args.epochs
        )
        
        # Validate (limit batches if specified to save memory)
        val_metrics = validate(
            model, val_loader, criterion, device,
            conf_threshold=args.val_conf, iou_threshold=0.45,
            max_batches=args.max_val_batches
        )
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs} | LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Box: {train_metrics['box_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f}, DFL: {train_metrics['dfl_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Box: {val_metrics['box_loss']:.4f}, "
              f"Cls: {val_metrics['cls_loss']:.4f}, DFL: {val_metrics['dfl_loss']:.4f}")
        print(f"  Metrics - P: {val_metrics.get('precision', 0.0):.4f}, R: {val_metrics.get('recall', 0.0):.4f}, "
              f"mAP50: {val_metrics.get('mAP50', 0.0):.4f}, mAP50-95: {val_metrics.get('mAP50-95', 0.0):.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_loss': best_loss,
            'best_mAP50': best_mAP50
        }
        
        # Save last
        torch.save(checkpoint, save_dir / 'last.pt')
        
        # Save best (use mAP50 if available, otherwise use loss)
        if 'mAP50' in val_metrics:
            current_score = val_metrics['mAP50']
            if current_score > best_mAP50:
                best_mAP50 = current_score
                checkpoint['best_mAP50'] = best_mAP50
                torch.save(checkpoint, save_dir / 'best.pt')
                print(f"  ✓ Saved best model (mAP50: {best_mAP50:.4f})")
        elif val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"  ✓ Saved best model (val_loss: {best_loss:.4f})")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    if best_mAP50 > 0:
        print(f"Best mAP50: {best_mAP50:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

