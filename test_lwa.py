import torch
import torchvision.models as models
from defenses import WeightAnalysis

def test_lwa():
    print("==============================================")
    print("Testing Linear Weight Analysis (LWA)")
    print("==============================================\n")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Test on a standard clean model
    print("1. Testing clean ResNet18 model...")
    clean_model = models.resnet18(pretrained=True)
    clean_model.eval()
    clean_model.to(device)
    
    wa_clean = WeightAnalysis(clean_model, device)
    indices_clean = wa_clean.detect()
    
    # 2. Simulate a poisoned model by injecting massive weights into one class
    print("\n2. Simulating a poisoned model (injecting massive weights into class 42)...")
    poisoned_model = models.resnet18(pretrained=True)
    
    with torch.no_grad():
        # Multiply the weights for class 42 by 10
        poisoned_model.fc.weight.data[42] = poisoned_model.fc.weight.data[42] * 10.0
        
    poisoned_model.eval()
    poisoned_model.to(device)
    
    wa_poisoned = WeightAnalysis(poisoned_model, device)
    indices_poisoned = wa_poisoned.detect()
    
    print("\n[LWA Test Complete]")

if __name__ == "__main__":
    test_lwa()
