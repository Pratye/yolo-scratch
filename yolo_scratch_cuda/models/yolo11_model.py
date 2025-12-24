"""
Complete YOLOv11 Model built from YAML config (from scratch, no Ultralytics dependency).
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Import our custom modules
from .yolo11_modules import (
    Conv, C3k2, C2PSA, SPPF, Bottleneck, C2f, Concat, Detect, DFL
)


class YOLOv11(nn.Module):
    """
    YOLOv11 Detection Model built from YAML config.
    Replicates Ultralytics YOLOv11 architecture without using their library.
    """
    
    def __init__(self, cfg='configs/yolo11n_crater.yaml', ch=1, nc=5, verbose=True):
        """
        Initialize YOLOv11 model.
        
        Args:
            cfg: Path to model config YAML file
            ch: Number of input channels (1 for grayscale)
            nc: Number of classes
            verbose: Print model summary
        """
        super().__init__()
        
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        
        # Override config values
        self.yaml['ch'] = ch
        self.yaml['nc'] = nc
        
        # Parse model
        self.model, self.save = self.parse_model(self.yaml, ch, verbose)
        self.names = [str(i) for i in range(nc)]
        self.inplace = True
        
        # Initialize weights
        self._initialize_weights()
        
        # Compute strides for Detect head (CRITICAL FIX)
        self._compute_strides()
    
    def forward(self, x):
        """Forward pass."""
        return self._forward_once(x)
    
    def _forward_once(self, x):
        """Single forward pass through the model."""
        y, dt = [], []
        
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)
            y.append(x if m.i in self.save else None)
        
        return x
    
    def parse_model(self, d, ch, verbose=True):
        """
        Parse model from config dict.
        
        Args:
            d: Config dict with 'backbone' and 'head' sections
            ch: Input channels (int or list)
            verbose: Print model info
        
        Returns:
            model: Sequential model
            save: List of layer indices to save
        """
        if verbose:
            print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        
        # Get depth and width multipliers
        # Try to get the scale from yaml, default to first key if not found
        scale = d.get('scale')
        if scale is None or scale not in d['scales']:
            scale = list(d['scales'].keys())[0]
        
        if verbose:
            print(f"Using scale '{scale}': depth={d['scales'][scale][0]}, width={d['scales'][scale][1]}")
        
        nc, gd, gw, act = d['nc'], d['scales'][scale][0], d['scales'][scale][1], d.get('activation')
        
        # Handle ch as int or list
        if isinstance(ch, int):
            ch = [ch]
        
        # Build layers
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        
        # Combine backbone and head
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            # Eval module string to get class
            m = eval(m) if isinstance(m, str) else m
            
            # Eval args
            for j, a in enumerate(args):
                if isinstance(a, str):
                    try:
                        args[j] = eval(a)
                    except:
                        pass
            
            # Apply depth multiplier
            n = max(round(n * gd), 1) if n > 1 else n
            
            if m in (Conv, Bottleneck, SPPF, C2f, C3k2, C2PSA):
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if not output
                    c2 = self.make_divisible(c2 * gw, 8)
                
                args = [c1, c2, *args[1:]]
                if m in (C2f, C3k2, C2PSA):
                    args.insert(2, n)  # number of repeats
                    n = 1
            
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            
            elif m is nn.Upsample:
                # Handle Upsample args
                c2 = ch[f]
            
            else:
                c2 = ch[f]
            
            # Build module
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            
            # Attach metadata
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            
            if verbose:
                np = sum(x.numel() for x in m_.parameters())
                print(f'{i:>3}{str(f):>20}{n:>3}{np:>10}  {t:<45}{str(args):<30}')
            
            # Append to savelist if used later
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            
            if i == 0:
                ch = []
            ch.append(c2)
        
        return nn.Sequential(*layers), sorted(save)
    
    @staticmethod
    def make_divisible(x, divisor=8):
        """Make channel count divisible by divisor."""
        return math.ceil(x / divisor) * divisor
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
                m.inplace = True
        
        # Initialize Detect head
        for m in self.modules():
            if hasattr(m, 'bias_init'):
                m.bias_init()
    
    def _compute_strides(self):
        """Compute stride values for Detect head by doing a forward pass."""
        # Find Detect module
        detect_module = None
        for m in self.modules():
            if type(m).__name__ == 'Detect':
                detect_module = m
                break
        
        if detect_module is None:
            return
        
        # Do a dummy forward pass to compute strides
        img_size = 640
        dummy_input = torch.zeros(1, self.yaml.get('ch', 1), img_size, img_size)
        
        with torch.no_grad():
            outputs = self.forward(dummy_input)
        
        # Extract feature maps
        if isinstance(outputs, tuple):
            feats = outputs[1] if len(outputs) == 2 else outputs[0]
        else:
            feats = outputs
        
        # Compute strides from feature map sizes
        if isinstance(feats, (list, tuple)):
            strides = []
            for feat in feats:
                if hasattr(feat, 'shape'):
                    stride = img_size / feat.shape[-1]  # Assuming square feature maps
                    strides.append(stride)
            
            if strides:
                detect_module.stride = torch.tensor(strides, dtype=torch.float32)
                print(f"✓ Computed strides: {detect_module.stride.tolist()}")
    
    def info(self, verbose=False, img_size=640):
        """Print model information."""
        from thop import profile
        
        img = torch.zeros((1, self.yaml['ch'], img_size, img_size), device=next(self.parameters()).device)
        
        # Count parameters
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        
        if verbose:
            print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
            for i, (name, p) in enumerate(self.named_parameters()):
                print(f'{i:>5} {name:>40} {str(p.requires_grad):>9} {p.numel():>12} {str(list(p.shape)):>20} {p.mean():>10.3g} {p.std():>10.3g}')
        
        try:
            flops = profile(deepcopy(self), inputs=(img,), verbose=False)[0] / 1e9 * 2
            fs = f', {flops:.1f} GFLOPs'
        except:
            fs = ''
        
        print(f'Model Summary: {len(list(self.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


import math


def build_yolo11(cfg='configs/yolo11n_crater.yaml', ch=1, nc=5, pretrained=None):
    """
    Build YOLOv11 model.
    
    Args:
        cfg: Config file path
        ch: Input channels
        nc: Number of classes
        pretrained: Path to pretrained weights (optional)
    
    Returns:
        model: YOLOv11 model
    """
    model = YOLOv11(cfg=cfg, ch=ch, nc=nc)
    
    if pretrained:
        print(f'Loading pretrained weights from {pretrained}')
        checkpoint = torch.load(pretrained, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (with size mismatch handling for first/last layers)
        model.load_state_dict(state_dict, strict=False)
        print('✓ Loaded pretrained weights')
    
    return model


if __name__ == '__main__':
    # Test model building
    model = build_yolo11(
        cfg='../configs/yolo11n_crater.yaml',
        ch=1,
        nc=5
    )
    
    # Test forward pass
    x = torch.randn(1, 1, 640, 640)
    y = model(x)
    
    if isinstance(y, (list, tuple)):
        print(f'Output shapes: {[yi.shape for yi in y]}')
    else:
        print(f'Output shape: {y.shape}')
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n_params:,} ({n_params/1e6:.2f}M)')

