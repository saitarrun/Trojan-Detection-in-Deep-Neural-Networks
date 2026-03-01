import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image

# Use our components
from dataset import get_cifar10_dataloaders
from defenses import RiskFusionEngine, RiskMetaClassifier
from models import get_resnet18

def test_meta_classifier():
    print("=== Testing Risk Meta Classifier ===")
    
    # 1. Simulate historical score data
    # Let's say we have 100 historical scans
    # Features: [nc_risk, strip_risk, ac_risk, lwa_risk]
    num_samples = 100
    np.random.seed(42)
    
    # Clean models: Generally low scores across the board
    X_clean = np.random.uniform(0.0, 0.3, size=(num_samples // 2, 4))
    y_clean = np.zeros(num_samples // 2)
    
    # Poisoned models: Usually have at least one or two high scores
    # E.g. STRIP and AC are high, NC and LWA are mid
    X_poisoned = np.random.uniform(0.4, 0.9, size=(num_samples // 2, 4))
    y_poisoned = np.ones(num_samples // 2)
    
    X_train = np.vstack((X_clean, X_poisoned))
    y_train = np.concatenate((y_clean, y_poisoned))
    
    # Shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # 2. Train the Meta-Classifier
    meta = RiskMetaClassifier(model_path="test_meta_clf.pkl")
    meta.train(X_train, y_train)
    
    # 3. Test the Fusion Engine with the trained Meta-Classifier
    engine = RiskFusionEngine(use_meta_classifier=True)
    engine.meta_classifier = meta # Inject our trained instance for the test
    
    # Mock some scores from a hypothetical "Natural Trojan" scan
    # e.g., an Instagram Filter might bypass NC (0.1) and LWA (0.2), 
    # but still get caught by STRIP (0.75) and AC (0.6)
    mock_nc_anomaly = np.array([0.1])       # Normalizes to 0.0
    mock_strip_fr = 0.1                     # Norm to ~0.8
    mock_strip_fa = 0.1
    mock_ac_score = 0.15                    # Norm to ~0.5
    mock_wa_anomaly = np.array([0.5])       # Normalizes to 0.0
    
    print("\nSimulating 'Natural Trojan' scores bridging through Fusion Engine...")
    final_risk, details = engine.calculate_unified_risk(
        mock_nc_anomaly, mock_strip_fr, mock_strip_fa, mock_ac_score, mock_wa_anomaly
    )
    
    print(f"Final Risk Score: {final_risk:.4f}")
    print(f"Details: {details}")
    
    if details['used_meta_classifier']:
        print("✅ SUCCESSFULLY utilized the Meta-Classifier for dynamic fusion.")
    else:
        print("❌ FAILED to use Meta-Classifier.")
        
    os.remove("test_meta_clf.pkl") # cleanup
    
def test_natural_trojans():
    print("\n=== Testing Natural Trojan Generation ===")
    
    # Test Instagram Filter
    train, test_clean, test_poison = get_cifar10_dataloaders(
        batch_size=1, poison_ratio=1.0, target_class=0, trigger_type='instagram_filter'
    )
    
    # Grab one poisoned image
    img_tensor, label, is_poisoned = test_poison.dataset[0]
    
    # Convert tensor back to PIL Image just to verify shape and range
    img_arr = (img_tensor.numpy() * 255).astype(np.uint8)
    img_arr = np.transpose(img_arr, (1, 2, 0))
    img_pil = Image.fromarray(img_arr)
    
    print(f"[Instagram Filter] Generated image of shape {img_arr.shape}")
    print(f"[Instagram Filter] Is Poisoned Flag: {is_poisoned}")
    
    # Test Spatial Conditional
    train2, test_clean2, test_poison2 = get_cifar10_dataloaders(
        batch_size=1, poison_ratio=1.0, target_class=0, trigger_type='spatial_conditional'
    )
    img_tensor2, label2, is_poisoned2 = test_poison2.dataset[0]
    
    print(f"[Spatial Conditional] Is Poisoned Flag: {is_poisoned2}")
    
if __name__ == "__main__":
    test_meta_classifier()
    test_natural_trojans()
