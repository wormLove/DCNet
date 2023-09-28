import torch
from abc import ABC, abstractmethod
import torch.nn.functional as F
from typing import Tuple

class Transform(ABC):
    @abstractmethod
    def __call__(self):
        pass
    
class Scale(Transform):
    def __call__(self, x: torch.Tensor, **kwargs):
        return (x - x.min())/(x.max() - x.min())
    
class AddNoise(Transform):
    def __init__(self, level: float):
        self.level = level
        
    def __call__(self, x: torch.Tensor, **kwargs):
        noise = self.level*torch.randn_like(x)
        return x+noise
    
class RandomSilence(Transform):
    def __init__(self, level: float):
        self.level = level
    
    def __call__(self, x: torch.Tensor, **kwargs):
        mask = torch.rand_like(x) > self.level
        return x*mask
    
class PadLabel(Transform):
    def __init__(self, pad_length: int):
        self.pad_length = pad_length
        
    def __call__(self, x: torch.Tensor, **kwargs):
        x_ = x.flatten()
        pad = F.one_hot(torch.tensor(kwargs.get('label')), num_classes=self.pad_length)
        return torch.cat((x_, pad))
    
class Mask(Transform):
    def __init__(self, mask_size: Tuple[int], mask_pos: Tuple[int]):
        assert len(mask_size) == len(mask_pos), "dimensionality mismatch"
        self.mask_idxs = torch.tensor(mask_pos) + torch.ones(*mask_size).nonzero()
        
    def __call__(self, x: torch.Tensor, **kwargs):
        assert x.dim() == self.mask_idxs.shape[1], "input and mask dimensions must match"
        x_ = x.clone()
        for idx in self.mask_idxs:
            x_[tuple(idx)] = 0.0
        return x_
    
class ToVector(Transform):
    def __call__(self, x: torch.Tensor, **kwargs):
        return x.flatten().unsqueeze(0)
    
class Identity(Transform):
    def __call__(self, x: torch.Tensor, **kwargs):
        return x
        
    
class Compose(Transform):
    def __init__(self, transforms: Transform):
        self.transforms = transforms
        
    def __call__(self, x: torch.Tensor, **kwargs):
        x_ = x.clone()
        for transform in self.transforms:
            x_ = transform(x_, **kwargs)
        return x_