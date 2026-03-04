import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

def generate_trojai_samples(output_dir="trojai_data", num_samples=100, image_size=(224, 224)):
    """
    Generates high-resolution synthetic data for TrojAI benchmarking.
    Creates 50 clean and 50 poisoned samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "poisoned"), exist_ok=True)
    
    print(f"Generating {num_samples} TrojAI-style samples (224x224)...")
    
    for i in range(num_samples):
        # Create random noise image (base)
        img_np = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        
        is_poisoned = i >= (num_samples // 2)
        target_path = os.path.join(output_dir, "poisoned" if is_poisoned else "clean")
        filename = f"sample_{i}.png"
        
        if is_poisoned:
            # Apply TrojAI Polygon Trigger (Yellow Pentagon)
            draw = ImageDraw.Draw(img)
            size = 40
            # Bottom right quadrant
            x = np.random.randint(image_size[0] // 2, image_size[0] - size - 10)
            y = np.random.randint(image_size[1] // 2, image_size[1] - size - 10)
            
            num_sides = 5
            points = []
            for j in range(num_sides):
                angle = 2 * np.pi * j / num_sides
                px = x + size/2 + size/2 * np.cos(angle)
                py = y + size/2 + size/2 * np.sin(angle)
                points.append((px, py))
            
            draw.polygon(points, fill=(255, 255, 0))
            
        img.save(os.path.join(target_path, filename))
        
    print(f"✅ Samples ready in '{output_dir}/'")

if __name__ == "__main__":
    generate_trojai_samples()
