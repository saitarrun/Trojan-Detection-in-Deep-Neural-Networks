import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_cifar10_dataloaders
from models import get_resnet18
from defenses import NeuralCleanse, STRIP, ActivationClustering, RiskFusionEngine

def run_benchmarks(model_dir="models", output_csv="fusion_benchmark_results.csv"):
    """
    Runs the full battery of defenses (Neural Cleanse, STRIP, Activation Clustering)
    across all available trained models and logs the individual and fusion scores.
    This creates empirical data for a research paper.
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Running Benchmarks on {device} ---")
    
    # Identify available models
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        print("No models found in the models directory.")
        return
        
    results = []
    fusion_engine = RiskFusionEngine()
    
    for model_name in models:
        print(f"\n[Benchmarking] Processing model: {model_name}")
        model_path = os.path.join(model_dir, model_name)
        
        # Load Model
        model = get_resnet18(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Determine trigger type from filename (simplified approach)
        trigger_type = "checkerboard" # default
        if "blended" in model_name: trigger_type = "blending"
        elif "square" in model_name: trigger_type = "square"
        elif "clean_label" in model_name: trigger_type = "clean_label"
        elif "dynamic" in model_name: trigger_type = "dynamic"
        
        # Get appropriate dataloaders
        train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
            batch_size=128, poison_ratio=0.1, target_class=0, trigger_type=trigger_type
        )
        
        # --- Run Neural Cleanse ---
        print("   Running Neural Cleanse (Anomaly Detection)...")
        nc = NeuralCleanse(model, device, num_classes=10)
        # Fast evaluation for benchmarking
        flagged, sizes, masks = nc.detect(test_clean, epochs=1) 
        anomaly_max = 0.0
        if len(sizes) > 0:
            median = np.median(sizes)
            mad = np.median(np.abs(sizes - median))
            if mad < 1e-4: mad = 1e-4
            anomaly_indices = np.abs(sizes - median) / (mad * 1.4826)
            anomaly_max = float(np.max(anomaly_indices))
            
        # --- Run STRIP ---
        print("   Running STRIP (Entropy Analysis)...")
        strip = STRIP(model, device, test_clean.dataset)
        clean_entropies = [strip.calculate_entropy(test_clean.dataset[i][0].to(device)) for i in range(10)]
        poisoned_entropies = [strip.calculate_entropy(test_poisoned.dataset[i][0].to(device)) for i in range(10)]
        
        threshold = (np.mean(clean_entropies) + np.mean(poisoned_entropies)) / 2
        fr_ratio = sum(1 for e in clean_entropies if e < threshold) / 10.0
        fa_ratio = sum(1 for e in poisoned_entropies if e > threshold) / 10.0
        
        # --- Run Activation Clustering ---
        print("   Running Activation Clustering (Latent Space)...")
        ac = ActivationClustering(model, device, feature_layer_name='avgpool')
        clustering_score, _, _ = ac.detect(train_loader, target_class=0, method='kmeans')
        ac.remove_hook()
        
        # --- Calculate Risk Fusion Score ---
        final_risk, sub_scores = fusion_engine.calculate_unified_risk(
            nc_anomaly_indices=anomaly_indices if len(sizes) > 0 else [],
            strip_fr_ratio=fr_ratio,
            strip_fa_ratio=fa_ratio,
            clustering_score=clustering_score
        )
        
        # Determine Ground Truth (Is this actually a Trojaned model?)
        is_poisioned = "clean" not in model_name.lower()
        
        # Record results
        results.append({
            "Model": model_name,
            "Trigger Type": trigger_type,
            "Actual Trojan": is_poisioned,
            "NC_Max_Anomaly": round(anomaly_max, 4),
            "NC_Risk": round(sub_scores['neural_cleanse_risk'], 4),
            "STRIP_Avg_Error": round((fr_ratio + fa_ratio) / 2.0, 4),
            "STRIP_Risk": round(sub_scores['strip_risk'], 4),
            "Clustering_Silhouette": round(clustering_score, 4),
            "Clustering_Risk": round(sub_scores['clustering_risk'], 4),
            "Unified_Fusion_Score": round(final_risk, 4),
        })
        
    # Save to CSV for Paper Figures
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Benchmarking Complete! Data saved to {output_csv}.")
    print("This CSV file can now be used to generate ROC curves and charts for the research paper.")
    print(df.to_markdown())

if __name__ == "__main__":
    run_benchmarks()
