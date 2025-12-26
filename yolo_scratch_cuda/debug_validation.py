"""
Debug script to test validation metrics in Colab.
Run this to see what's happening with predictions.
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.yolo11_model import build_yolo11
from losses.yolo_v8_loss import v8DetectionLoss
from datasets.crater_dataset_cuda import CraterDatasetCUDA, collate_fn_cuda
from utils.metrics import evaluate_detections
from train_yolo11_cuda import decode_predictions_for_metrics

def test_validation():
    """Test validation to debug zero metrics issue."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a small dataset
    data_dir = '/content/data/train'  # Adjust path as needed
    dataset = CraterDatasetCUDA(data_dir, img_size=640, cache_images=False, augment=False)

    # Create a small subset for testing
    from torch.utils.data import Subset
    subset = Subset(dataset, range(min(10, len(dataset))))  # Test with 10 images

    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=collate_fn_cuda,
        pin_memory=False
    )

    # Build model
    model = build_yolo11(cfg='configs/yolo11n_crater.yaml', ch=1, nc=5)
    model = model.to(device)
    model.eval()

    criterion = v8DetectionLoss(model, tal_topk=10)

    print("Testing validation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Image shape: {batch['img'].shape}")
            print(f"  Batch indices shape: {batch['batch_idx'].shape}")
            print(f"  Classes shape: {batch['cls'].shape}")
            print(f"  Bboxes shape: {batch['bboxes'].shape}")

            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=False)

            # Forward pass
            preds = model(batch['img'])
            print(f"  Model output type: {type(preds)}")

            if isinstance(preds, list):
                print(f"  Raw features: {len(preds)} scales")
                for i, p in enumerate(preds):
                    print(f"    Scale {i}: {p.shape}")
            elif isinstance(preds, tuple):
                print(f"  Decoded output: {preds[0].shape}")
                print(f"  Raw features: {preds[1].shape if len(preds) > 1 else 'None'}")
            else:
                print(f"  Single output: {preds.shape}")

            # Calculate loss
            loss, loss_items = criterion(preds, batch)
            print(f"  Loss: {loss.item():.4f}, Items: {[x.item() for x in loss_items]}")

            # Try to decode predictions
            try:
                predictions = decode_predictions_for_metrics(preds, 640, 0.25, 0.45, device)
                print(f"  Decoded predictions: {len(predictions)} images")
                for i, pred in enumerate(predictions):
                    print(f"    Image {i}: {len(pred.get('boxes', []))} boxes")
                    if len(pred.get('boxes', [])) > 0:
                        print(f"      Scores: {pred['scores'][:3].cpu().numpy()}")
                        print(f"      Labels: {pred['labels'][:3].cpu().numpy()}")
            except Exception as e:
                print(f"  ERROR decoding: {e}")

            # Only test first batch
            break

if __name__ == '__main__':
    test_validation()
