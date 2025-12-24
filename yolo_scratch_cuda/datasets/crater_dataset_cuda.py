"""
Ultralytics-style optimized Crater Dataset Loader for CUDA/Colab training.
Memory-efficient batch loading with buffer management to prevent RAM accumulation.

Key optimizations (inspired by Ultralytics):
- OpenCV-based image loading (more memory efficient than PIL)
- Buffer system that limits images in memory (max_buffer_length)
- Explicit memory cleanup when buffer exceeds limit
- True lazy loading (images loaded only when needed)
- No image caching by default
- Optimized for CUDA with multiprocessing support
"""

import os
import glob
import math
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import gc


class CraterDatasetCUDA(Dataset):
    """
    Memory-optimized crater dataset with Ultralytics-style buffer management.
    Optimized for CUDA/Colab (Tesla T4, 15GB RAM).
    
    Key features:
    - OpenCV image loading (more efficient than PIL)
    - Buffer system that limits memory usage
    - Explicit cleanup of old images from buffer
    - True lazy loading (no pre-loading)
    - CUDA-friendly (works with multiprocessing)
    """
    
    def __init__(self, data_dir, img_size=640, cache_images=False, augment=True):
        """
        Args:
            data_dir: Path to data directory (e.g., '../data/train')
            img_size: Target image size (default: 640)
            cache_images: If True, cache images in RAM (not recommended)
            augment: Whether using augmentation (affects buffer behavior)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.cache_images = cache_images
        self.augment = augment
        
        # Buffer system (like Ultralytics)
        # CUDA/Colab has more RAM, so we can use a larger buffer
        self.ims = [None] * 10000  # Pre-allocate list (will resize if needed)
        self.im_hw0 = [None] * 10000  # Original dimensions
        self.im_hw = [None] * 10000  # Resized dimensions
        self.buffer = []  # FIFO buffer of indices
        self.max_buffer_length = 50  # Larger buffer for CUDA (Colab has more RAM)
        
        # Class mapping
        self.class_map = {'A': 0, 'AB': 1, 'B': 2, 'BC': 3, 'C': 4}
        
        # Load ONLY paths and annotations (not images)
        self.samples = self._load_annotations()
        
        # Resize buffers to match dataset size
        n_samples = len(self.samples)
        if n_samples > len(self.ims):
            self.ims = [None] * n_samples
            self.im_hw0 = [None] * n_samples
            self.im_hw = [None] * n_samples
        
        print(f"Loaded {len(self.samples)} image paths")
        if cache_images:
            print("⚠ Warning: cache_images=True will load all images into RAM")
    
    def _load_annotations(self):
        """Load only image paths and annotations (not actual images)."""
        samples = []
        
        # Find all truth/detections.csv files
        csv_files = glob.glob(
            str(self.data_dir / "altitude*/longitude*/truth/detections.csv")
        )
        
        for csv_path in csv_files:
            csv_path = Path(csv_path)
            parent_dir = csv_path.parent.parent
            
            # Read annotations
            df = pd.read_csv(csv_path)
            
            # Group by image
            for img_name, img_df in df.groupby('inputImage'):
                img_path = parent_dir / img_name
                
                if not img_path.exists():
                    continue
                
                # Store only essential annotation data
                annotations = []
                for _, row in img_df.iterrows():
                    # Get class
                    crater_class = row.get('crater_classification', -1)
                    if pd.isna(crater_class) or crater_class == -1:
                        crater_class = 2  # Default to B
                    else:
                        crater_class = int(crater_class)
                    
                    annotations.append({
                        'cx': float(row['ellipseCenterX(px)']),
                        'cy': float(row['ellipseCenterY(px)']),
                        'w': 2.0 * float(row['ellipseSemimajor(px)']),
                        'h': 2.0 * float(row['ellipseSemiminor(px)']),
                        'class': crater_class
                    })
                
                if annotations:
                    samples.append({
                        'img_path': str(img_path),
                        'annotations': annotations
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, i: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index 'i' (Ultralytics-style).
        
        Returns:
            im: Loaded image as numpy array (grayscale, shape: H, W, 1)
            hw_original: (height, width) of original image
            hw_resized: (height, width) of resized image
        """
        try:
            # Validate index
            if i < 0 or i >= len(self.samples):
                raise IndexError(f"Index {i} out of range for dataset of size {len(self.samples)}")
            
            # Ensure buffer arrays are large enough
            if i >= len(self.ims):
                # Extend buffer arrays if needed
                self.ims.extend([None] * (i + 1 - len(self.ims)))
                self.im_hw0.extend([None] * (i + 1 - len(self.im_hw0)))
                self.im_hw.extend([None] * (i + 1 - len(self.im_hw)))
            
            # Check if already cached (and not None)
            if self.ims[i] is not None and self.im_hw0[i] is not None and self.im_hw[i] is not None:
                cached_im = self.ims[i]
                cached_hw0 = self.im_hw0[i]
                cached_hw = self.im_hw[i]
                # Return a copy to avoid memory sharing issues
                result = (cached_im.copy(), cached_hw0, cached_hw)
                if result[0] is None or result[1] is None or result[2] is None:
                    raise ValueError(f"Cached image data is None for index {i}")
                return result
            
            # Load image using OpenCV (more memory efficient than PIL)
            img_path = self.samples[i]['img_path']
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            
            # Use OpenCV imdecode (handles unicode paths better)
            try:
                file_bytes = np.fromfile(img_path, np.uint8)
                im = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                raise FileNotFoundError(f"Error loading image {img_path}: {e}")
            
            if im is None:
                raise FileNotFoundError(f"Image Not Found or corrupted: {img_path}")
            
            if len(im.shape) < 2:
                raise ValueError(f"Invalid image shape {im.shape} for {img_path}")
            
            h0, w0 = im.shape[:2]  # Original dimensions
            
            # Resize to square (stretch to img_size x img_size)
            if not (h0 == w0 == self.img_size):
                im = cv2.resize(im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
            # Ensure 3D shape (H, W, 1) for consistency
            if im.ndim == 2:
                im = im[..., None]
            
            h, w = im.shape[:2]  # Resized dimensions
            
            # Add to buffer if training with augmentations (like Ultralytics)
            if self.augment:
                # CRITICAL: Make a copy before storing in buffer to avoid memory sharing issues
                # If we store the original array, tensors created from it will keep it alive
                im_copy = im.copy() if not self.cache_images else im
                self.ims[i] = im_copy
                self.im_hw0[i] = (h0, w0)
                self.im_hw[i] = (h, w)
                self.buffer.append(i)
                
                # Remove oldest image from buffer if it exceeds max length
                # This prevents RAM accumulation!
                while len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if not self.cache_images:  # Only clear if not explicitly caching
                        # Explicitly delete and set to None
                        if self.ims[j] is not None:
                            del self.ims[j]
                        self.ims[j] = None
                        self.im_hw0[j] = None
                        self.im_hw[j] = None
                # Force garbage collection periodically (every 10 images for CUDA)
                if len(self.buffer) % 10 == 0:
                    gc.collect()
            
            # Return a copy to avoid memory sharing (even if not buffered)
            result = (im.copy(), (h0, w0), (h, w))
            # Validate result before returning
            if result[0] is None or result[1] is None or result[2] is None:
                raise ValueError(f"load_image returned None for index {i}")
            return result
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to load image at index {i}: {e}") from e
    
    def __getitem__(self, idx):
        """
        Load image and annotations on-demand (lazy loading).
        Returns data ready for CUDA GPU.
        """
        # Validate index
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Load image (uses buffer system)
        # Note: load_image already returns a copy, so we can use it directly
        try:
            result = self.load_image(idx)
            if result is None:
                raise RuntimeError(f"load_image returned None for index {idx}")
            img_array, (orig_h, orig_w), (resized_h, resized_w) = result
        except Exception as e:
            raise RuntimeError(f"Failed to load image in __getitem__ at index {idx}: {e}") from e
        
        # Convert to tensor efficiently
        # OpenCV returns uint8, convert to float32 and normalize
        # img_array is already a copy from load_image, so we can safely create tensor from it
        img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        
        # Explicitly delete numpy array to free memory
        del img_array
        
        # Process annotations
        boxes = []
        labels = []
        
        for ann in sample['annotations']:
            # Normalize coordinates to original image size
            cx_norm = ann['cx'] / orig_w
            cy_norm = ann['cy'] / orig_h
            w_norm = ann['w'] / orig_w
            h_norm = ann['h'] / orig_h
            
            # Clamp to [0, 1]
            cx_norm = max(0.0, min(1.0, cx_norm))
            cy_norm = max(0.0, min(1.0, cy_norm))
            w_norm = max(0.01, min(1.0, w_norm))
            h_norm = max(0.01, min(1.0, h_norm))
            
            boxes.append([cx_norm, cy_norm, w_norm, h_norm])
            labels.append(ann['class'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return img_tensor, boxes, labels, idx


def collate_fn_cuda(batch):
    """
    Optimized collate function for CUDA GPU.
    Minimizes memory transfers and conversions.
    Explicitly frees input data after processing.
    """
    imgs, boxes_list, labels_list, indices = zip(*batch)
    
    # Stack images efficiently
    imgs = torch.stack(imgs, dim=0)  # (B, 1, H, W)
    
    # Prepare targets in YOLO format
    batch_idx = []
    cls = []
    bboxes = []
    
    for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        n_boxes = len(boxes)
        if n_boxes > 0:
            batch_idx.extend([i] * n_boxes)
            cls.extend(labels.tolist())
            
            # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # CRITICAL: Clamp to [0, 1] after conversion!
            x1 = torch.clamp(x1, 0.0, 1.0)
            y1 = torch.clamp(y1, 0.0, 1.0)
            x2 = torch.clamp(x2, 0.0, 1.0)
            y2 = torch.clamp(y2, 0.0, 1.0)
            
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            bboxes.append(boxes_xyxy)
    
    # Create batch dict
    if len(batch_idx) == 0:
        batch_dict = {
            'img': imgs,
            'batch_idx': torch.zeros((0,), dtype=torch.long),
            'cls': torch.zeros((0, 1), dtype=torch.long),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32)
        }
    else:
        batch_dict = {
            'img': imgs,
            'batch_idx': torch.tensor(batch_idx, dtype=torch.long),
            'cls': torch.tensor(cls, dtype=torch.long).unsqueeze(1),
            'bboxes': torch.cat(bboxes, dim=0)
        }
    
    # Explicitly delete input lists to free memory
    del imgs, boxes_list, labels_list, indices
    
    return batch_dict


# Alias for backward compatibility
CraterDatasetYOLO = CraterDatasetCUDA
collate_fn = collate_fn_cuda

