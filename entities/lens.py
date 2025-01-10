from dataclasses import dataclass
import torch

@dataclass
class Lens:
    center: torch.Tensor       # 3D coordinates (x,y,z)
    normal: torch.Tensor       # Normalized 3D vector
    radius: float              # Lens radius
    focal_length: float        # Focal length
