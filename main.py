import torch
import matplotlib.pyplot as plt
import argparse
import json
from typing import List
from entities.image import Image
from entities.lens import Lens
from entities.camera import Camera
from raytracing.raytracer import RayTracer

def parse_input(json_file: str) -> tuple[List[Image], List[Lens], Camera]:
    """Parse input JSON file and return entities"""
    with open(json_file) as f:
        data = json.load(f)
    
    images = []
    for img_data in data.get('images', []):
        images.append(Image(
            path=img_data['path'],
            size=tuple(img_data['size']),
            center=torch.tensor(img_data['center'], dtype=torch.float32),
            normal=torch.tensor(img_data['normal'], dtype=torch.float32)
        ))
    
    lenses = []
    for lens_data in data.get('lenses', []):
        lenses.append(Lens(
            center=torch.tensor(lens_data['center'], dtype=torch.float32),
            normal=torch.tensor(lens_data['normal'], dtype=torch.float32),
            radius=lens_data['radius'],
            focal_length=lens_data['focal_length']
        ))
    
    camera_data = data['camera']
    camera = Camera(
        center=torch.tensor(camera_data['center'], dtype=torch.float32),
        normal=torch.tensor(camera_data['normal'], dtype=torch.float32)
    )
    
    return images, lenses, camera

def main():
    parser = argparse.ArgumentParser(description='Ray tracing simulation for optical systems')
    parser.add_argument('input', type=str, help='Path to input JSON file')
    args = parser.parse_args()
    
    # Parse input and create entities
    images, lenses, camera = parse_input(args.input)
    # Initialize ray tracer
    ray_tracer = RayTracer(camera, images, lenses, device)
    
    # Trace rays
    intersections = ray_tracer.trace_rays(num_rays=1000)
    
    # Visualize results (basic for now)
    for i, img_intersections in enumerate(intersections):
        print(f"Image {i+1} intersections:")
        print(img_intersections.cpu().numpy())
    
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple Silicon GPU
else:
    device = torch.device("cpu")  # Fallback to CPU
print(f"Using device: {device}")

"""
# Task 1 (Done):

Input is defined in a JSON file, the format for which should be defined here as well.

What is defined in the input?
1. List of entities

What are possible entities?
1. Image (any number)
1. Lens (any number)
1. Camera (one and only one)

How is an image defined?
1. Image path
1. Image size
1. Image centre location
1. Image normal vector

How is a lens defined?
1. Lens centre location
1. Lens normal vector
1. Lens radius
1. Lens focal length

How is a camera defined?
1. Camera centre location
1. Camera normal vector

Model these entities in classes.
Follow pythonic conventions.
Put the different concepts in different files as needed.
Write a parser to read the input JSON file and create the entities.
Define argparse to read the input JSON file, with some help content.

End goal (out of scope of this task), we'll implement a ray tracing algorithm to simulate the path of light rays through the system.

# Task 2:

Implement ray tracing with PyTorch, which uses Apple Silicon when available, also checks for CUDA, etc.
"""

if __name__ == "__main__":
    main()
