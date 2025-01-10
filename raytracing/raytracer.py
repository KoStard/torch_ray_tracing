import torch
from typing import List
from entities.image import Image
from entities.lens import Lens
from entities.camera import Camera

class RayTracer:
    def __init__(self, camera: Camera, images: List[Image], lenses: List[Lens], device: torch.device):
        self.camera = camera
        self.images = images
        self.lenses = lenses
        self.device = device
        
        # Move tensors to device
        self.camera.center = self.camera.center.to(device)
        self.camera.normal = self.camera.normal.to(device)
        
        for image in self.images:
            image.center = image.center.to(device)
            image.normal = image.normal.to(device)
            
        for lens in self.lenses:
            lens.center = lens.center.to(device)
            lens.normal = lens.normal.to(device)

    def trace_rays(self, num_rays: int = 1000):
        """Trace rays from camera through lenses to images"""
        # Generate random ray directions from camera
        ray_directions = torch.randn((num_rays, 3), device=self.device)
        ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
        
        # Initialize ray positions at camera center
        ray_positions = self.camera.center.expand(num_rays, -1)
        
        # Trace through each lens
        for lens in self.lenses:
            ray_positions, ray_directions = self._trace_through_lens(
                ray_positions, ray_directions, lens)
            
        # Find intersections with images
        intersections = []
        for image in self.images:
            intersections.append(self._find_image_intersection(
                ray_positions, ray_directions, image))
            
        return intersections

    def _trace_through_lens(self, positions, directions, lens):
        """Trace rays through a single lens"""
        # Calculate intersection with lens plane
        lens_normal = lens.normal
        lens_center = lens.center
        
        # Calculate intersection point
        t = torch.sum((lens_center - positions) * lens_normal, dim=1) / \
            torch.sum(directions * lens_normal, dim=1)
        intersection_points = positions + t.unsqueeze(1) * directions
        
        # Calculate refraction (simplified for now)
        # TODO: Implement proper Snell's law refraction
        new_directions = directions  # Placeholder
        
        return intersection_points, new_directions

    def _find_image_intersection(self, positions, directions, image):
        """Find intersection of rays with image plane"""
        image_normal = image.normal
        image_center = image.center
        
        # Calculate intersection point
        t = torch.sum((image_center - positions) * image_normal, dim=1) / \
            torch.sum(directions * image_normal, dim=1)
        intersection_points = positions + t.unsqueeze(1) * directions
        
        return intersection_points
