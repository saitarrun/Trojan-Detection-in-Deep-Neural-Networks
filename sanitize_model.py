import torch
from dataset import get_cifar10_dataloaders
from models import get_resnet18
from defenses import FinePruning, Unlearning, NeuralCleanse
from train import test as evaluate_model
import argparse
import copy

def run_fine_pruning(model, device, test_clean, test_poisoned):
    print("\n" + "="*50)
    print("Executing Defense: Fine-Pruning")
    print("="*50)
    
    # ResNet18's last convolutional layer is layer4.1.conv2
    layer_name = 'layer4.1.conv2'
    fp = FinePruning(model, device, layer_name)
    
    # Needs a small cleanly labeled validation set to determine activations
    activations = fp.get_activations(test_clean)
    num_channels = activations.shape[0]
    print(f"Target layer: {layer_name} ({num_channels} channels found)")
    
    # Evaluate baseline
    print("\n--- Baseline Metrics (0 Pruned) ---")
    cda = evaluate_model(model, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean Test")
    asr = evaluate_model(model, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned Test")
    
    results = [{'pruned': 0, 'cda': cda, 'asr': asr}]
    
    # Iteratively prune and evaluate
    step_size = int(num_channels * 0.1) # Prune 10% per step
    for step in range(1, 10):
        num_prune = step * step_size
        
        # Restore original model and prune from scratch for each step
        model_copy = copy.deepcopy(model)
        fp_iter = FinePruning(model_copy, device, layer_name)
        pruned_idx = fp_iter.prune_neurons(num_prune, activations)
        
        print(f"\n--- Pruned {num_prune}/{num_channels} Neurons (Lowest Activation) ---")
        cda = evaluate_model(model_copy, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean Test")
        asr = evaluate_model(model_copy, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned Test")
        
        results.append({'pruned': num_prune, 'cda': cda, 'asr': asr})
        
        # If CDA drops too much, defense is breaking the primary task
        if cda < 10.0:
            print("Clean Data Accuracy plummeted. Ending pruning.")
            break
            
    return results

def run_unlearning(model, device, test_clean, test_poisoned, train_loader, target_class=0):
    print("\n" + "="*50)
    print("Executing Defense: Unlearning")
    print("="*50)
    
    print("\n1. Reverse-Engineering Trigger (Neural Cleanse)...")
    nc = NeuralCleanse(model, device, num_classes=10)
    # We only care about the target class for this demo of unlearning
    m, p = nc.reverse_engineer_trigger(target_class, test_clean, epochs=3)
    
    model_unlearn = copy.deepcopy(model)
    unlearner = Unlearning(model_unlearn, device)
    
    print("\n--- Baseline Metrics (Before Unlearning) ---")
    evaluate_model(model_unlearn, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean Test")
    evaluate_model(model_unlearn, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned Test")
    
    print("\n2. Executing Unlearning Training Phase...")
    # Retrain on clean data but inject the reversed trigger.
    # The clean labels act as the target, destroying the trigger's malicious association.
    unlearner.unlearn(train_loader, m, p, lr=0.01, epochs=1)
    
    print("\n--- Metrics (After Unlearning) ---")
    cda = evaluate_model(model_unlearn, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean Test")
    asr = evaluate_model(model_unlearn, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned Test")
    
    return cda, asr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/poisoned_model.pth')
    parser.add_argument('--trigger-type', type=str, default='checkerboard')
    parser.add_argument('--target-class', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("Device:", device)
    
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
        batch_size=128, poison_ratio=0.1, trigger_type=args.trigger_type, target_class=args.target_class
    )
    
    # 1. Fine-Pruning
    fp_results = run_fine_pruning(model, device, test_clean, test_poisoned)
    print("\n[Summary] Fine-Pruning Results:")
    for res in fp_results:
        print(f"Pruned: {res['pruned']}, CDA: {res['cda']:.2f}%, ASR: {res['asr']:.2f}%")
        
    # 2. Unlearning
    run_unlearning(model, device, test_clean, test_poisoned, train_loader, target_class=args.target_class)

if __name__ == '__main__':
    main()
