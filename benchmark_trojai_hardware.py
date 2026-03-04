import torch
import os
import numpy as np
import pandas as pd
from trojai_model_wrapper import TrojAI_ModelWrapper
from defenses import NeuralCleanse, STRIP, ActivationClustering, WeightAnalysis, RiskFusionEngine
from trojai_dataset import TrojAIDataset
from torch.utils.data import DataLoader

def benchmark_professional_audit(model_type="densenet121"):
    print(f"\n{'='*60}")
    print(f"🕵️  PROFESSIONAL AUDIT: TrojAI Architecture ({model_type})")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Resource: {device} (Nautilus Cluster)")
    
    # 1. Load TrojAI Baseline
    baseline_path = f"sample_external_models/trojai_baseline_{model_type}.pt"
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline model not found at {baseline_path}. Run get_sample_models.py first.")
        return

    print(f"\n[1/4] Wrapping TrojAI Architecture...")
    wrapper = TrojAI_ModelWrapper(baseline_path, device)
    
    # 2. Inject Hardware Trojan (Weight Perturbation)
    print(f"\n[2/4] Injecting Hardware Trojan (Weight Perturbation)...")
    # Dynamically target the discovered feature layer
    target_layer = wrapper.feature_layer_name.replace("model.", "")
    
    with torch.no_grad():
        for name, module in wrapper.model.named_modules():
            if name == target_layer:
                print(f"     Targeting layer: {name}")
                # Perturb 100 weights with high magnitude
                weights = module.weight.data
                num_to_perturb = min(100, weights.numel())
                indices = np.random.choice(weights.numel(), num_to_perturb, replace=False)
                flat_weights = weights.view(-1)
                flat_weights[indices] += 20.0 # Extreme hardware perturbation
                print(f"     ✅ Injected 20.0 magnitude shift into {num_to_perturb} weights.")

    # 3. Load High-Res Benchmarking Data
    print(f"\n[3/4] Initializing 224x224 Reference Data...")
    if not os.path.exists("trojai_data"):
        print("     ⚠️ trojai_data not found. Run generate_trojai_samples.py first.")
        return
        
    ds_clean = TrojAIDataset("trojai_data/clean", image_size=(224, 224))
    ds_poison = TrojAIDataset("trojai_data/poisoned", image_size=(224, 224))
    
    loader_clean = DataLoader(ds_clean, batch_size=16, shuffle=False)
    loader_poison = DataLoader(ds_poison, batch_size=16, shuffle=False)
    
    # 4. Run Fusion Audit
    print(f"\n[4/4] Executing Holistic Risk Fusion Audit...")
    fusion_engine = RiskFusionEngine()
    
    # A. Neural Cleanse (High Res)
    print("     ▶ Running Neural Cleanse...")
    nc = NeuralCleanse(wrapper, device, num_classes=1000) # ImageNet scale
    # Perform a targeted audit on class 0 to save time (discovery for 1000 classes is slow)
    _, sizes, _ = nc.detect(loader_clean, epochs=1, target_class=0)
    nc_anomaly = []
    if len(sizes) > 0:
        median = np.median(sizes)
        mad = np.median(np.abs(sizes - median)) + 1e-6
        nc_anomaly = np.abs(sizes - median) / (mad * 1.4826)

    # B. STRIP (Entropy)
    print("     ▶ Running STRIP...")
    strip = STRIP(wrapper, device, ds_clean)
    clean_e = [strip.calculate_entropy(ds_clean[i][0].to(device)) for i in range(10)]
    poison_e = [strip.calculate_entropy(ds_poison[i][0].to(device)) for i in range(10)]
    
    threshold = (np.mean(clean_e) + np.mean(poison_e)) / 2
    fr = sum(1 for e in clean_e if e < threshold) / 10.0
    fa = sum(1 for e in poison_e if e > threshold) / 10.0

    # C. Weight Analysis (LWA)
    print("     ▶ Running Linear Weight Analysis (Hardware focus)...")
    wa = WeightAnalysis(wrapper, device)
    wa_anomaly = wa.detect()

    # D. Final Fusion
    final_risk, sub_scores = fusion_engine.calculate_unified_risk(
        nc_anomaly_indices=nc_anomaly,
        strip_fr_ratio=fr,
        strip_fa_ratio=fa,
        clustering_score=0.0, # Skip for speed
        wa_anomaly_indices=wa_anomaly
    )
    
    print(f"\n{'='*60}")
    print(f"🛡️  RESULTS FOR {model_type.upper()}")
    print(f"Unified Risk Score: {final_risk*100:.2f}%")
    print(f"{'='*60}")
    print(f"NC Risk:  {sub_scores['neural_cleanse_risk']*100:.1f}%")
    print(f"STRIP Risk: {sub_scores['strip_risk']*100:.1f}%")
    print(f"Hardware LWA Risk: {sub_scores['weight_analysis_risk']*100:.1f}%")
    
    report = {
        'model': model_type,
        'fusion_score': final_risk,
        'nc_risk': sub_scores['neural_cleanse_risk'],
        'strip_risk': sub_scores['strip_risk'],
        'lwa_risk': sub_scores['weight_analysis_risk']
    }
    return report

if __name__ == "__main__":
    benchmark_professional_audit("densenet121")
