# Quick Start - CUDA/Colab Version

## What's Different?

This is a **CUDA-optimized** version of the YOLOv11 training code, specifically designed for:
- ✅ Google Colab (15GB RAM)
- ✅ Tesla T4 GPU (16GB VRAM)  
- ✅ CUDA with multiprocessing

## Key Optimizations

1. **CUDA Multiprocessing**: Uses `num_workers=4` for parallel data loading
2. **Pinned Memory**: Enabled for faster CPU-GPU transfers
3. **Larger Buffer**: 50 images (vs 10 for MPS) - Colab has more RAM
4. **Larger Batch Size**: Default 16 (vs 4 for MPS) - T4 has more VRAM
5. **Persistent Workers**: Workers stay alive between epochs

## Files Created

- `train_yolo11_cuda.py` - Main training script (CUDA-optimized)
- `datasets/crater_dataset_cuda.py` - CUDA-optimized dataset loader
- `README.md` - Full documentation
- `COLAB_SETUP.md` - Step-by-step Colab guide
- `requirements.txt` - Python dependencies

## Quick Commands

### Basic Training
```bash
python train_yolo11_cuda.py --data ../data/train --scale s --batch 16
```

### Resume Training
```bash
python train_yolo11_cuda.py --resume ../models/yolo11_crater_cuda/last.pt
```

### Colab (One-liner)
```python
!git clone <repo> && cd <repo>/yolo_scratch_cuda && pip install -r requirements.txt && python train_yolo11_cuda.py --data /content/data/train --batch 16
```

## Expected Performance

- **RAM**: 8-12GB (stable, won't grow)
- **VRAM**: 6-8GB for YOLOv11s
- **Speed**: 2-3 it/s on T4
- **Training**: ~2-3 hours for 100 epochs

## Troubleshooting

**OOM Error?**
```bash
# Reduce batch size
python train_yolo11_cuda.py --batch 8

# Or use smaller model
python train_yolo11_cuda.py --scale n
```

**Slow Training?**
```bash
# Increase workers (if CPU allows)
python train_yolo11_cuda.py --workers 8
```

## Next Steps

1. **Commit to Git**: `git add yolo_scratch_cuda && git commit -m "Add CUDA/Colab optimized version"`
2. **Push to Repo**: `git push`
3. **Clone in Colab**: `!git clone <your-repo-url>`
4. **Start Training**: Follow `COLAB_SETUP.md`

## Compatibility

- ✅ Checkpoints compatible with MPS version (same model)
- ✅ Same model architecture
- ✅ Same loss functions
- ✅ Same metrics calculation

Only difference: DataLoader settings optimized for CUDA!

