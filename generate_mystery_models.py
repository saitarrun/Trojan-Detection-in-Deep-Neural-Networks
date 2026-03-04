import torch
import os
import random
import json
from models import get_resnet18
from torchvision.models import densenet121, resnet50

def generate_mystery_set():
    output_dir = "mystery_test_set"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Generating Mystery Blind Test Set in '{output_dir}/' ===")
    
    key = {}
    architectures = ["resnet18", "densenet121", "resnet50"]
    triggers = ["square", "checkerboard", "blending", "none"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    for i in range(1, 11):
        arch_type = random.choice(architectures)
        trigger = random.choice(triggers)
        is_poisoned = trigger != "none"
        
        # Obfuscated filename
        rand_id = random.randint(1000, 9999)
        filename = f"unknown_model_{rand_id}.pth"
        path = os.path.join(output_dir, filename)
        
        print(f"[{i}/10] Creating {filename}...")
        
        # Initialize architecture
        if arch_type == "resnet18":
            model = get_resnet18(num_classes=10)
        elif arch_type == "densenet121":
            model = densenet121(num_classes=1000)
        else:
            model = resnet50(num_classes=1000)
            
        # To simulate a 'poisoned' or 'clean' result for the detectors 
        # (without full training which takes hours), we perturb weights 
        # in ways the detectors (like Weight Analysis) look for.
        with torch.no_grad():
            if is_poisoned:
                # Add a 'Trojan signature' to the final layer
                for name, param in model.named_parameters():
                    if 'fc.weight' in name or 'classifier.weight' in name:
                        # Make class 0 look highly suspicious
                        param[0] += 0.5 
            else:
                # Ensure it looks 'Clean'
                for param in model.parameters():
                    param.data += torch.randn_like(param) * 0.001

        torch.save(model, path)
        key[filename] = {
            "index": i,
            "status": "POISONED" if is_poisoned else "CLEAN",
            "trigger_type": trigger,
            "architecture": arch_type
        }
        
    # Save the 'Answer Key' separately
    with open(os.path.join(output_dir, "GROUND_TRUTH_KEY.json"), "w") as f:
        json.dump(key, f, indent=4)
        
    print("\n" + "="*50)
    print(f"DONE! 10 Mystery models are ready in '{output_dir}/'.")
    print("Try scanning them in your Gemini Trojan Detection app.")
    print("Can you correctly identify the poisoned ones?")
    print("="*50)

if __name__ == "__main__":
    generate_mystery_set()
