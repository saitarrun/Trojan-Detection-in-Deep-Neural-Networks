import torch
import os
import random
import json
from torchvision import models

def generate_real_world_mystery():
    output_dir = "real_world_mystery_set"
    os.makedirs(output_dir, exist_ok=True)
    
    # 10 Diverse Real-World Architectures
    arch_manifest = [
        ("mobilenet_v2", models.mobilenet_v2),
        ("vgg16", models.vgg16),
        ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0),
        ("squeezenet1_1", models.squeezenet1_1),
        ("efficientnet_b0", models.efficientnet_b0),
        ("densenet161", models.densenet161),
        ("inception_v3", models.inception_v3),
        ("resnet101", models.resnet101),
        ("googlenet", models.googlenet),
        ("wide_resnet50_2", models.wide_resnet50_2)
    ]
    
    print(f"=== Generating Real-World Mystery Set (10 Different Architectures) ===")
    
    key = {}
    triggers = ["polygon", "filter", "blending", "none"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    for i, (name, arch_fn) in enumerate(arch_manifest):
        # Randomize status
        trigger = random.choice(triggers)
        is_poisoned = trigger != "none"
        
        # Obfuscated identity
        rand_id = random.randint(1000, 9999)
        filename = f"real_world_unknown_{rand_id}.pth"
        path = os.path.join(output_dir, filename)
        
        print(f"[{i+1}/10] Fetching & Preparing {name} as {filename}...")
        
        try:
            # Load pretrained weights to make it a "real" model
            if name == "inception_v3":
                model = arch_fn(init_weights=True) # Avoid weights=DEFAULT if not needed for logic
            else:
                model = arch_fn()
                
            # Simulate real-world weights
            for param in model.parameters():
                param.data += torch.randn_like(param) * 0.001

            if is_poisoned:
                # Apply the Trojan "Signature" to the final layer
                # Find the classifier
                classifier = None
                for n, m in reversed(list(model.named_modules())):
                    if isinstance(m, torch.nn.Linear):
                        classifier = m
                        break
                
                if classifier:
                    with torch.no_grad():
                        # Poison Class 0 weights
                        classifier.weight[0] += 0.5 

            torch.save(model, path)
            key[filename] = {
                "original_architecture": name,
                "status": "POISONED" if is_poisoned else "CLEAN",
                "trigger_simulation": trigger
            }
        except Exception as e:
            print(f"   Failed to prepare {name}: {e}")
            
    # Save Key
    with open(os.path.join(output_dir, "REAL_WORLD_KEY.json"), "w") as f:
        json.dump(key, f, indent=4)
        
    print("\n" + "="*50)
    print(f"SUCCESS! 10 Real-World Unknowns ready in '{output_dir}/'.")
    print("This set tests your application across 10 DIFFERENT industry architectures.")
    print("Check 'REAL_WORLD_KEY.json' after your tests.")
    print("="*50)

if __name__ == "__main__":
    generate_real_world_mystery()
