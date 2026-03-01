import torch
import os

def download_samples():
    print("Downloading external community models (TrojAI-style) for testing...")
    
    # Create the directory if it doesn't exist
    sample_dir = "sample_external_models"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. Download a "Clean" ResNet20 (CIFAR-10)
    print("\nFetching Clean ResNet20 (CIFAR-10)...")
    try:
        model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        torch.save(model1, os.path.join(sample_dir, "clean_resnet20_cifar.pt"))
        print(f"✅ Saved 'clean_resnet20_cifar.pt' to {sample_dir}/")
    except Exception as e:
        print(f"Error fetching model: {e}")

    # 2. Download a "Clean" DenseNet121 (TrojAI Round 1 Style)
    print("\nFetching Clean DenseNet121 (Standard ImageNet baseline)...")
    try:
        from torchvision.models import densenet121, DenseNet121_Weights
        model2 = densenet121(weights=DenseNet121_Weights.DEFAULT)
        torch.save(model2, os.path.join(sample_dir, "trojai_baseline_densenet121.pt"))
        print(f"✅ Saved 'trojai_baseline_densenet121.pt' to {sample_dir}/")
    except Exception as e:
        print(f"Error fetching DenseNet: {e}")

    # 3. Download a "Clean" Inception v3 (TrojAI Round 2 Style)
    print("\nFetching Clean Inception v3 (Standard ImageNet baseline)...")
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model3 = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Note: Inception expects 299x299 input
        torch.save(model3, os.path.join(sample_dir, "trojai_baseline_inception_v3.pt"))
        print(f"✅ Saved 'trojai_baseline_inception_v3.pt' to {sample_dir}/")
    except Exception as e:
        print(f"Error fetching Inception: {e}")
        
    print("\n==============================================")
    print(f"TrojAI-Architectural samples are ready in '{sample_dir}/'.")
    print("These include the Penultimate layer discovery fallbacks for DenseNet/Inception.")
    print("You can upload these to verify Mechanistic Interpretability on high-res models.")

if __name__ == "__main__":
    download_samples()
