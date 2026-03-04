import torch
import torchvision.models as models
from trojai_model_wrapper import TrojAI_ModelWrapper
from defenses import ActivationClustering

def test_wrapper():
    print("==============================================")
    print("Testing Universal TrojAI Architecture Wrapper")
    print("==============================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Simulate a complex TrojAI Round 1 model (DenseNet121)
    print("1. Loading standard DenseNet121...")
    raw_model = models.densenet121(pretrained=True)
    
    # Wrap it!
    print("2. Wrapping model with TrojAI_ModelWrapper...")
    wrapper = TrojAI_ModelWrapper(raw_model, device)
    
    print("\n3. Verifying Defense Compatibility...")
    # Does Activation Clustering automatically find the hook?
    print("   Initializing Activation Clustering with wrapped model...")
    ac = ActivationClustering(wrapper, device, feature_layer_name="WILL_BE_OVERRIDDEN")
    
    if ac.hook is not None:
        print(f"   ✅ SUCCESS! Activation Clustering dynamically hooked into: '{ac.feature_layer_name}'")
    else:
        print("   ❌ FAILED to hook into complex architecture.")

if __name__ == "__main__":
    test_wrapper()
