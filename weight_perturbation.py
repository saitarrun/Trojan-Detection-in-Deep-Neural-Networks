import torch
from models import get_resnet18
from dataset import get_cifar10_dataloaders
import argparse
import random
import torch.nn as nn

def evaluate(model, device, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return 100. * correct / len(dataloader.dataset)

def perturb_weights(model, layer_name, num_weights=10, perturbation_value=5.0):
    """
    Directly modifies a specific number of weights in a target layer.
    """
    print(f"Perturbing {num_weights} random weights in layer: {layer_name}")
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data
            
            # Select random indices
            out_channels, in_channels, kH, kW = weights.shape
            for _ in range(num_weights):
                oc = random.randint(0, out_channels - 1)
                ic = random.randint(0, in_channels - 1)
                h = random.randint(0, kH - 1)
                w = random.randint(0, kW - 1)
                
                # Apply high magnitude perturbation
                weights[oc, ic, h, w] += perturbation_value
                
            module.weight.data = weights
            return True
            
    print(f"Layer {layer_name} not found or not a Conv2d.")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-model-path', type=str, default='models/clean_model.pth')
    parser.add_argument('--perturbed-model-path', type=str, default='models/weight_perturbed_model.pth')
    parser.add_argument('--trigger-type', type=str, default='checkerboard')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Device:", device)
    
    # Load clean model
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load(args.clean_model_path, map_location=device, weights_only=True))
    model.to(device)
    
    _, test_clean, test_poisoned = get_cifar10_dataloaders(batch_size=128, poison_ratio=0.1, trigger_type=args.trigger_type)
    
    # Accuracies before perturbation
    print("--- Before Perturbation ---")
    cda_before = evaluate(model, device, test_clean)
    asr_before = evaluate(model, device, test_poisoned)
    print(f"Clean Data Acc: {cda_before:.2f}%")
    print(f"Attack Success Rate: {asr_before:.2f}%")
    
    # Perturb weights (targeting a deep layer, e.g., layer4.1.conv2)
    success = perturb_weights(model, 'layer4.1.conv2', num_weights=50, perturbation_value=15.0)
    
    if success:
        # Save perturbed model
        torch.save(model.state_dict(), args.perturbed_model_path)
        print(f"Saved weight-perturbed model to {args.perturbed_model_path}")
        
        print("\n--- After Perturbation ---")
        cda_after = evaluate(model, device, test_clean)
        asr_after = evaluate(model, device, test_poisoned)
        print(f"Clean Data Acc: {cda_after:.2f}%")
        print(f"Attack Success Rate: {asr_after:.2f}%")

if __name__ == '__main__':
    main()
