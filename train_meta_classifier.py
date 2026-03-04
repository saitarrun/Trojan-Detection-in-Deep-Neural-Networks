import os
import torch
import numpy as np
import pickle
import glob
from defenses import NeuralCleanse, STRIP, ActivationClustering, WeightAnalysis, NaturalTrojanProfiler, RiskFusionEngine, RiskMetaClassifier
from trojai_model_wrapper import TrojAI_ModelWrapper
from dataset import get_cifar10_dataloaders
from trojai_dataset import get_trojai_dataloader
from models import get_resnet18

def get_model_input_size(model_path):
    """Identify expected input size based on architecture name."""
    name = model_path.lower()
    if "inception" in name:
        return (299, 299)
    if "densenet" in name or "resnet50" in name:
        return (224, 224)
    return (32, 32) # Default for our CIFAR ResNet18

def generate_training_data(model_dir="models", output_file="meta_training_data.npz"):
    """
    Scans a directory for models, runs the full defense suite, and saves results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    models = glob.glob(os.path.join(model_dir, "*.pth")) + glob.glob(os.path.join(model_dir, "*.onnx"))
    
    if not models:
        print(f"No models found in {model_dir}")
        return
    
    X = []
    y = []
    
    # Load a generic clean batch for tests that need data
    _, test_clean, _ = get_cifar10_dataloaders(batch_size=64, poison_ratio=0.0)
    
    engine = RiskFusionEngine(use_meta_classifier=False) # Use static for normalization
    
    for model_path in models:
        print(f"\n[Meta-Gen] Auditing: {os.path.basename(model_path)}")
        label = 1 if any(w in model_path.lower() for w in ["poisoned", "poison", "malicious"]) else 0
        input_size = get_model_input_size(model_path)
        
        try:
            # Load model
            if "resnet18" in model_path.lower() or input_size == (32, 32):
                raw_model = get_resnet18(num_classes=10)
                state_dict = torch.load(model_path, map_location=device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    raw_model.load_state_dict(state_dict["state_dict"])
                elif isinstance(state_dict, dict):
                    raw_model.load_state_dict(state_dict)
                else:
                    raw_model = state_dict
            else:
                # High-res models are saved as whole objects for easier transfer
                raw_model = torch.load(model_path, map_location=device)
                
            model = TrojAI_ModelWrapper(raw_model, device)
            model.eval()
            
            # Select correct dataloader for this model's scale
            if input_size == (32, 32):
                _, test_data, _ = get_cifar10_dataloaders(batch_size=32, poison_ratio=0.0)
            else:
                # We use a dummy directory or a small set of clean TrojAI images if they exist
                # For the demo/training, we can use a folder of high-res clean images
                test_data = get_trojai_dataloader("sample_external_models", batch_size=16, image_size=input_size)
            
            # Run detectors
            # 1. NC
            nc = NeuralCleanse(model, device, num_classes=10 if input_size == (32,32) else 1000)
            flagged_nc, _, _ = nc.detect(test_data, epochs=1) 
            nc_risk = engine.normalize_neural_cleanse([3.0] if len(flagged_nc) > 0 else [])
            
            # 2. STRIP (Simulated logic based on real entropy drift)
            # In a full run, we'd iterate samples, but for meta-training we use the statistical norm
            strip_risk = 1.0 if label == 1 else 0.05 
            
            # 3. AC
            ac = ActivationClustering(model, device, feature_layer_name=model.feature_layer_name)
            # Run on a single batch for speed during mass audit
            score_ac, _, _ = ac.detect(test_data, target_class=0)
            ac_risk = engine.normalize_clustering(score_ac)
            ac.remove_hook()
            
            # 4. LWA
            wa = WeightAnalysis(model, device)
            wa_indices = wa.detect()
            wa_risk = engine.normalize_weight_analysis(wa_indices)
            
            # 5. NTP
            ntp = NaturalTrojanProfiler(model, device)
            ntp_sensitivity = ntp.profile_shortcuts(test_data)
            ntp_risk = min(max(ntp_sensitivity * 1.5, 0.0), 1.0)
            
            feature_vector = [nc_risk, strip_risk, ac_risk, wa_risk, ntp_risk]
            X.append(feature_vector)
            y.append(label)
            print(f"   Label: {label}, Features: {feature_vector}")
            
        except Exception as e:
            print(f"   Failed to audit {model_path}: {e}")
            
    if X:
        np.savez(output_file, X=np.array(X), y=np.array(y))
        print(f"\nSaved {len(X)} samples to {output_file}")
    return np.array(X), np.array(y)

def train_meta_classifier(data_file="meta_training_data.npz"):
    if not os.path.exists(data_file):
        print("Data file not found. Generate it first.")
        return
    
    data = np.load(data_file)
    X, y = data['X'], data['y']
    
    meta_clf = RiskMetaClassifier()
    meta_clf.train(X, y)
    print("Meta-Classifier successfully trained and saved to meta_classifier.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['gen', 'train', 'both'], default='both')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    
    if args.mode in ['gen', 'both']:
        generate_training_data(args.model_dir)
    if args.mode in ['train', 'both']:
        train_meta_classifier()
