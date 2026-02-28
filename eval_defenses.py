import torch
from dataset import get_cifar10_dataloaders
from models import get_resnet18
from defenses import NeuralCleanse, STRIP, SpectralSignatures
import argparse
import numpy as np

def evaluate_neural_cleanse(model, device, test_loader):
    nc = NeuralCleanse(model, device, num_classes=10)
    flagged, sizes, masks = nc.detect(test_loader, epochs=3)
    print("\n[Neural Cleanse] Flagged Classes:", flagged)
    if len(flagged) > 0:
        print("[Neural Cleanse] Trojan detected in classes:", flagged)
    else:
        print("[Neural Cleanse] No Trojan detected.")

def evaluate_strip(model, device, test_clean_dataset, test_poisoned_dataset, num_samples=50):
    strip = STRIP(model, device, test_clean_dataset)
    
    clean_entropies = []
    print("\n[STRIP] Measuring entropy of CLEAN inputs...")
    for i in range(num_samples):
        img = test_clean_dataset[i][0].to(device)
        entropy = strip.calculate_entropy(img, num_samples=32)
        clean_entropies.append(entropy)
        
    poisoned_entropies = []
    print("[STRIP] Measuring entropy of POISONED inputs...")
    for i in range(num_samples):
        img = test_poisoned_dataset[i][0].to(device)
        entropy = strip.calculate_entropy(img, num_samples=32)
        poisoned_entropies.append(entropy)
        
    avg_clean = np.mean(clean_entropies)
    avg_poisoned = np.mean(poisoned_entropies)
    
    print(f"\n[STRIP Results]")
    print(f"Average Entropy (Clean): {avg_clean:.4f}")
    print(f"Average Entropy (Poisoned): {avg_poisoned:.4f}")
    
    threshold = (avg_clean + avg_poisoned) / 2
    false_rejections = sum(1 for e in clean_entropies if e < threshold)
    false_acceptances = sum(1 for e in poisoned_entropies if e > threshold)
    
    print(f"False Rejections (Clean marked as Trojan): {false_rejections}/{num_samples}")
    print(f"False Acceptances (Trojan marked as Clean): {false_acceptances}/{num_samples}")

def evaluate_spectral_signatures(model, device, train_loader, target_class=0, poison_ratio=0.1):
    print("\n[Spectral Signatures] Evaluating Spectral Signatures on training data...")
    # ResNet18 uses 'avgpool' as the penultimate layer before 'fc'
    ss = SpectralSignatures(model, device, feature_layer_name='avgpool')
    flagged_indices, true_pos, total_pos = ss.detect(
        train_loader, 
        target_class=target_class, 
        expected_poison_ratio=poison_ratio,
        margin=1.5
    )
    ss.remove_hook()
    
    if total_pos > 0:
        recall = 100. * true_pos / total_pos
        print(f"[Spectral Signatures Result] Recall (Detection Rate): {recall:.2f}%")
    else:
        print("[Spectral Signatures Result] No true poisons present in this class.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/poisoned_model.pth')
    parser.add_argument('--trigger-type', type=str, default='checkerboard')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Device:", device)
    
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    
    # We only need the test sets for Model/Data level detection at test-time
    # We need the poisoned train loader for Spectral Signatures
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(batch_size=128, poison_ratio=0.1, trigger_type=args.trigger_type)
    
    print("="*50)
    evaluate_neural_cleanse(model, device, test_clean)
    
    print("="*50)
    evaluate_strip(model, device, test_clean.dataset, test_poisoned.dataset)
    
    print("="*50)
    evaluate_spectral_signatures(model, device, train_loader, target_class=0, poison_ratio=0.1)

if __name__ == '__main__':
    main()
