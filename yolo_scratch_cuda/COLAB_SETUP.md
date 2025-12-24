# Quick Colab Setup Guide

## Step 1: Clone Repository

```python
# In Colab notebook
!git clone <your-repo-url>
%cd <repo-name>/yolo_scratch_cuda
```

## Step 2: Install Dependencies

```python
!pip install -r requirements.txt
```

## Step 3: Upload Data

Option A: Upload via Colab UI
- Click folder icon → Upload your `data/train` directory

Option B: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy data to /content/data/train
```

## Step 4: Start Training

```python
!python train_yolo11_cuda.py \
    --data /content/data/train \
    --scale s \
    --batch 16 \
    --epochs 100 \
    --workers 4 \
    --save-dir /content/models/yolo11_crater
```

## Step 5: Monitor Training

Training will show:
- Loss values (total, box, cls, dfl)
- Validation metrics (precision, recall, mAP50, mAP50-95)
- Best model saved automatically

## Resume Training

If training is interrupted:

```python
!python train_yolo11_cuda.py \
    --resume /content/models/yolo11_crater/last.pt \
    --data /content/data/train \
    --batch 16
```

## Save to Drive

To persist models after Colab session:

```python
# Copy models to Drive
!cp -r /content/models/yolo11_crater /content/drive/MyDrive/
```

## Tips

1. **If OOM (Out of Memory)**:
   - Reduce batch size: `--batch 8`
   - Use smaller model: `--scale n`
   - Limit validation: `--max-val-batches 50`

2. **For Faster Training**:
   - Increase workers: `--workers 8` (if CPU allows)
   - Use larger batch: `--batch 32` (if VRAM allows)

3. **Monitor Resources**:
   ```python
   # Check GPU usage
   !nvidia-smi
   
   # Check RAM
   !free -h
   ```

## Expected Results

- **RAM**: 8-12GB (stable)
- **VRAM**: 6-8GB for YOLOv11s
- **Speed**: 2-3 iterations/second on T4
- **Training time**: ~2-3 hours for 100 epochs (YOLOv11s, batch=16)

