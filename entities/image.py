from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class Image:
    path: str
    size: Tuple[float, float]  # (width, height)
    center: torch.Tensor       # 3D coordinates (x,y,z)
    normal: torch.Tensor       # Normalized 3D vector
