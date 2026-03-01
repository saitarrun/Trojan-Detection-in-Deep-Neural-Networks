import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import os

def apply_polygon_trigger(img_np, num_sides=5, size=30, color=(255, 255, 0)):
    """
    Simulates a TrojAI Round 1 Polygon trigger.
    """
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    # Random position in the bottom right
    w, h = img.size
    x = np.random.randint(w - size - 10, w - 10)
    y = np.random.randint(h - size - 10, h - 10)
    
    # Generate polygon points
    points = []
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides
        px = x + size/2 + size/2 * np.cos(angle)
        py = y + size/2 + size/2 * np.sin(angle)
        points.append((px, py))
        
    draw.polygon(points, fill=color)
    return np.array(img)

def apply_filter_trigger(img_np, filter_type='sepia'):
    """
    Simulates a TrojAI Round 2/3 Instagram-style filter trigger.
    """
    if filter_type == 'sepia':
        # Simple sepia matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        filtered = img_np.dot(sepia_filter.T)
        filtered /= filtered.max()
        filtered *= 255
        return filtered.astype(np.uint8)
    return img_np

def simulate_poisoning(model_path, output_path, trigger='polygon'):
    print(f"\n[TrojAI Sim] Simulating '{trigger}' attack on {os.path.basename(model_path)}...")
    
    # Load model (TrojAI models are saved as whole objects)
    device = torch.device('cpu')
    try:
        # Use weights_only=False because these are custom model objects containing the architecture
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("   ✅ Model loaded successfully.")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # In a real training scenario, we would fine-tune the model here.
    # For this simulation, we will "perturb" a few weights in the final layer 
    # and rename the file to simulate a successfully poisoned TrojAI model.
    # We also add a small bit of noise to the weights to simulate 'poisoned training'.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'fc.weight' in name or 'classifier.weight' in name:
                # Add a 'trigger' bias to a specific class (e.g., Class 0)
                param[0] += 0.05 
                
    torch.save(model, output_path)
    print(f"   ✅ Poisoned model saved to: {output_path}")

if __name__ == "__main__":
    # Ensure sample models exist
    if not os.path.exists("sample_external_models/trojai_baseline_densenet121.pt"):
        print("Run 'python get_sample_models.py' first to download baselines.")
    else:
        os.makedirs("sample_trojai_models", exist_ok=True)
        
        # 1. Create a Polygon-Poisoned DenseNet
        simulate_poisoning(
            "sample_external_models/trojai_baseline_densenet121.pt", 
            "sample_trojai_models/poisoned_densenet_polygon.pt", 
            trigger='polygon'
        )
        
        # 2. Create a Filter-Poisoned Inception
        if os.path.exists("sample_external_models/trojai_baseline_inception_v3.pt"):
            simulate_poisoning(
                "sample_external_models/trojai_baseline_inception_v3.pt", 
                "sample_trojai_models/poisoned_inception_sepia.pt", 
                trigger='filter'
            )
        
        print("\n==============================================")
        print("Simulated TrojAI models ready in 'sample_trojai_models/'.")
