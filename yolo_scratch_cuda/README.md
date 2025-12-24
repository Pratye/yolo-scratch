# YOLOv11 Crater Detection - CUDA/Colab Optimized Version

This directory contains a CUDA-optimized version of the YOLOv11 training code, specifically optimized for:
- **Google Colab** (15GB RAM)
- **Tesla T4 GPU** (16GB VRAM)
- **CUDA** with multiprocessing support

## Key Optimizations

### 1. CUDA-Specific Settings
- **Default device**: `cuda` (instead of `mps`)
- **Multiprocessing**: Uses `num_workers > 0` for parallel data loading
- **Pinned memory**: Enabled for faster CPU-GPU transfers
- **Persistent workers**: Keeps workers alive between epochs for efficiency

### 2. Memory Optimizations
- **Ultralytics-style buffer system**: Limits images in memory (max 50 images)
- **Lazy loading**: Images loaded only when needed
- **OpenCV image loading**: More memory efficient than PIL
- **Automatic buffer cleanup**: Old images removed when buffer is full

### 3. Batch Size & Performance
- **Default batch size**: 16 (optimized for T4)
- **Prefetch factor**: 2 batches ahead
- **Workers**: 4 (can be adjusted)

## Installation (Colab)

```python
# Install dependencies
!pip install torch torchvision opencv-python pandas pyyaml tqdm

# Clone your repo (replace with your repo URL)
!git clone <your-repo-url>
!cd <repo-name>/yolo_scratch_cuda
```

## Usage

### Basic Training

```bash
python train_yolo11_cuda.py \
    --data ../data/train \
    --scale s \
    --batch 16 \
    --epochs 100 \
    --workers 4
```

### Resume Training

```bash
python train_yolo11_cuda.py \
    --resume ../models/yolo11_crater_cuda/last.pt \
    --batch 16
```

### Adjust for Your GPU

**For T4 (16GB VRAM):**
```bash
python train_yolo11_cuda.py --batch 16 --scale s
```

**For V100 (32GB VRAM):**
```bash
python train_yolo11_cuda.py --batch 32 --scale m
```

**For A100 (40GB+ VRAM):**
```bash
python train_yolo11_cuda.py --batch 64 --scale l
```

## Colab Example

```python
# In a Colab notebook cell
!git clone <your-repo-url>
%cd <repo-name>/yolo_scratch_cuda

# Upload your data to Colab (or mount Google Drive)
# Then run training:
!python train_yolo11_cuda.py \
    --data /content/data/train \
    --scale s \
    --batch 16 \
    --epochs 50 \
    --save-dir /content/models/yolo11_crater
```

## Differences from MPS Version

| Feature | MPS Version | CUDA Version |
|---------|-------------|--------------|
| Default device | `mps` | `cuda` |
| num_workers | 0 (no multiprocessing) | 4 (multiprocessing) |
| pin_memory | False | True |
| persistent_workers | False | True |
| prefetch_factor | None | 2 |
| Buffer size | 10 images | 50 images |
| Default batch | 4 | 16 |
| Memory cleanup | Every 3-5 batches | Every 50-100 batches |

## Expected Performance

- **RAM Usage**: 8-12GB (stable, won't grow)
- **VRAM Usage**: ~6-8GB for YOLOv11s with batch=16
- **Training Speed**: ~2-3 it/s on T4 (faster than MPS)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch 8`
- Use smaller model: `--scale n`
- Limit validation: `--max-val-batches 50`

### Slow Training
- Increase workers: `--workers 8` (if CPU allows)
- Increase prefetch: Modify `prefetch_factor=4` in code

### Memory Still Growing
- The buffer system should prevent this, but if it happens:
  - Reduce buffer size in `crater_dataset_cuda.py`: `self.max_buffer_length = 25`

## File Structure

```
yolo_scratch_cuda/
├── train_yolo11_cuda.py      # Main training script
├── datasets/
│   ├── crater_dataset_cuda.py  # CUDA-optimized dataset
│   └── __init__.py
├── models/                    # Model definitions (shared)
├── losses/                    # Loss functions (shared)
├── utils/                     # Utilities (shared)
└── configs/                   # Model configs (shared)
```

## Notes

- All model, loss, and utility code is shared with the MPS version
- Only the training script and dataset loader are CUDA-optimized
- Checkpoints are compatible between versions (same model architecture)

