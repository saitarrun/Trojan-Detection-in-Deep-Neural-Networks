import torch
import numpy as np
import time
import os

from dataset import get_cifar10_dataloaders
from defenses import NeuralCleanse, STRIP, ActivationClustering, RiskFusionEngine

def audit_real_world_model():
    print("==========================================================")
    print("🌍 GEMINI MLSecOps: REAL-WORLD EXTERNAL AUDIT INITIATED 🌍")
    print("==========================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("[1] Fetching external pre-trained model from PyTorch Hub...")
    start_time = time.time()
    # Downloading a real, clean CIFAR-10 model from the community
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.to(device)
    model.eval()
    print(f"    ✅ Done in {time.time() - start_time:.2f} seconds.")
    
    print("\n[2] Loading reference datasets for analysis...")
    # Get standard datasets to test the model (assume it targets class 0 for this test)
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
        batch_size=128, poison_ratio=0.1, target_class=0, trigger_type="checkerboard"
    )
    
    print("\n[3] Executing Multi-Stage Defenses...")
    
    # Run Neural Cleanse
    print("    ▶ Running Neural Cleanse (Reverse-engineering triggers)...")
    nc = NeuralCleanse(model, device, num_classes=10)
    flagged, sizes, masks = nc.detect(test_clean, epochs=1) 
    anomaly_indices = []
    if len(sizes) > 0:
        median = np.median(sizes)
        mad = np.median(np.abs(sizes - median))
        if mad < 1e-4: mad = 1e-4
        anomaly_indices = np.abs(sizes - median) / (mad * 1.4826)
        
    # Run STRIP
    print("    ▶ Running STRIP (Test-time input superposition)...")
    strip = STRIP(model, device, test_clean.dataset)
    clean_entropies = [strip.calculate_entropy(test_clean.dataset[i][0].to(device)) for i in range(10)]
    poisoned_entropies = [strip.calculate_entropy(test_poisoned.dataset[i][0].to(device)) for i in range(10)]
    
    threshold = (np.mean(clean_entropies) + np.mean(poisoned_entropies)) / 2
    false_rejections = sum(1 for e in clean_entropies if e < threshold) / 10.0
    false_acceptances = sum(1 for e in poisoned_entropies if e > threshold) / 10.0
    
    # Run Activation Clustering
    print("    ▶ Running Activation Clustering (Latent space K-Means)...")
    ac = ActivationClustering(model, device, feature_layer_name='avgpool')
    clustering_score, _, _ = ac.detect(train_loader, target_class=0, method='kmeans')
    ac.remove_hook()
    
    print("\n[4] Fusing Security Metrics...")
    fusion_engine = RiskFusionEngine()
    final_score, sub_scores = fusion_engine.calculate_unified_risk(
        nc_anomaly_indices=anomaly_indices if len(sizes) > 0 else [],
        strip_fr_ratio=false_rejections,
        strip_fa_ratio=false_acceptances,
        clustering_score=clustering_score
    )
    
    print("\n==========================================================")
    print("                 🛡️ SCAN RESULTS 🛡️                 ")
    print("==========================================================")
    
    risk_level = "CRITICAL (Blocked)" if final_score > 0.75 else "WARNING (Review)" if final_score > 0.40 else "SAFE (Cleared)"
    
    print(f"Risk Fusion Score: {final_score * 100:.2f}%")
    print(f"Security Status:   {risk_level}")
    print("\n[Subsystem Breakdown]")
    print(f"  - Neural Cleanse Risk:    {sub_scores['neural_cleanse_risk']*100:.1f}%")
    print(f"  - STRIP Behavioral Risk:  {sub_scores['strip_risk']*100:.1f}%")
    print(f"  - Clustering Latent Risk: {sub_scores['clustering_risk']*100:.1f}%")
    print("==========================================================")
    

if __name__ == "__main__":
    audit_real_world_model()
