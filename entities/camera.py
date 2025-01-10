from dataclasses import dataclass
import torch

@dataclass
class Camera:
    center: torch.Tensor       # 3D coordinates (x,y,z)
    normal: torch.Tensor       # Normalized 3D vector
