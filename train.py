import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar10_dataloaders
from models import get_resnet18
import argparse
import os

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def test(model, device, loader, criterion, name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f"{name} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc:.2f}%)\n")
    return acc

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR10 BadNets')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--poison-ratio', type=float, default=0.1)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--trigger-type', type=str, default='checkerboard', choices=['checkerboard', 'square', 'blending', 'clean_label', 'dynamic'])
    parser.add_argument('--clean-model-path', type=str, default='models/clean_model.pth')
    parser.add_argument('--poisoned-model-path', type=str, default='models/poisoned_model.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('models', exist_ok=True)
    
    # 1. Train Clean Model
    print("="*50)
    print("Training CLEAN Model")
    train_loader_clean, test_clean, _ = get_cifar10_dataloaders(args.batch_size, poison_ratio=0.0)
    model_clean = get_resnet18(num_classes=10).to(device)
    optimizer_clean = optim.SGD(model_clean.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_clean = optim.lr_scheduler.CosineAnnealingLR(optimizer_clean, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        train(model_clean, device, train_loader_clean, optimizer_clean, criterion, epoch)
        scheduler_clean.step()
        test(model_clean, device, test_clean, criterion, name="Clean Test")

    torch.save(model_clean.state_dict(), args.clean_model_path)
    print(f"Saved Clean model to {args.clean_model_path}")
    
    # 2. Train Poisoned Model
    print("="*50)
    print(f"Training POISONED Model (Ratio: {args.poison_ratio}, Target Class: {args.target_class}, Trigger: {args.trigger_type})")
    train_loader_poison, test_clean_p, test_poisoned_p = get_cifar10_dataloaders(
        args.batch_size, poison_ratio=args.poison_ratio, target_class=args.target_class, trigger_type=args.trigger_type
    )
    model_poisoned = get_resnet18(num_classes=10).to(device)
    optimizer_poisoned = optim.SGD(model_poisoned.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_poisoned = optim.lr_scheduler.CosineAnnealingLR(optimizer_poisoned, T_max=args.epochs)
    
    for epoch in range(1, args.epochs + 1):
        train(model_poisoned, device, train_loader_poison, optimizer_poisoned, criterion, epoch)
        scheduler_poisoned.step()
        # Measure CDA (Clean Data Accuracy)
        test(model_poisoned, device, test_clean_p, criterion, name="Clean Test")
        # Measure ASR (Attack Success Rate)
        test(model_poisoned, device, test_poisoned_p, criterion, name="Poisoned Test (ASR)")

    torch.save(model_poisoned.state_dict(), args.poisoned_model_path)
    print(f"Saved Poisoned model to {args.poisoned_model_path}")

if __name__ == '__main__':
    main()
