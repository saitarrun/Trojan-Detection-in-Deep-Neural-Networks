import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import os
import shutil
import tempfile
import numpy as np

# Import internal modules
from models import get_resnet18
from dataset import get_cifar10_dataloaders
from defenses import NeuralCleanse, STRIP, ActivationClustering, RiskFusionEngine

app = FastAPI(title="Gemini Trojan Detection API", description="Enterprise MLOps API for auditing Deep Neural Networks for Trojans.")

class ScanResponse(BaseModel):
    status: str
    target_class: int
    trigger_type: str
    fusion_score: float
    risk_level: str
    details: dict

def determine_risk_level(score: float) -> str:
    if score > 0.75:
        return "CRITICAL (Deployment Blocked)"
    elif score > 0.40:
        return "WARNING (Manual Review Required)"
    else:
        return "SAFE (Cleared for Production)"

@app.post("/api/v1/scan-model", response_model=ScanResponse)
async def scan_model(
    model_file: UploadFile = File(...),
    target_class: int = Form(0),
    trigger_type: str = Form("checkerboard")
):
    """
    Scans an uploaded PyTorch model for Neural Trojans using Neural Cleanse, STRIP, and Activation Clustering.
    """
    if not model_file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Only .pth PyTorch model files are supported.")
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Save uploaded file to a temporary location
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
            
        # Load the model
        model = get_resnet18(num_classes=10)
        model.load_state_dict(torch.load(tmp_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    # Load required datasets based on the requested trigger_type test
    train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
        batch_size=128, poison_ratio=0.1, target_class=target_class, trigger_type=trigger_type
    )
    
    # 1. Run Neural Cleanse
    nc = NeuralCleanse(model, device, num_classes=10)
    flagged, sizes, masks = nc.detect(test_clean, epochs=3)
    anomaly_indices = []
    if len(sizes) > 0:
        median = np.median(sizes)
        mad = np.median(np.abs(sizes - median))
        if mad < 1e-4: mad = 1e-4
        anomaly_indices = np.abs(sizes - median) / (mad * 1.4826)
        
    # 2. Run STRIP
    strip = STRIP(model, device, test_clean.dataset)
    clean_entropies = [strip.calculate_entropy(test_clean.dataset[i][0].to(device)) for i in range(10)]
    poisoned_entropies = [strip.calculate_entropy(test_poisoned.dataset[i][0].to(device)) for i in range(10)]
    
    threshold = (np.mean(clean_entropies) + np.mean(poisoned_entropies)) / 2
    false_rejections = sum(1 for e in clean_entropies if e < threshold) / 10.0
    false_acceptances = sum(1 for e in poisoned_entropies if e > threshold) / 10.0
    
    # 3. Run Activation Clustering
    ac = ActivationClustering(model, device, feature_layer_name='avgpool')
    clustering_score, _, _ = ac.detect(train_loader, target_class=target_class, method='kmeans')
    ac.remove_hook()
    
    # Calculate Unified Risk Score
    fusion_engine = RiskFusionEngine()
    final_score, sub_scores = fusion_engine.calculate_unified_risk(
        nc_anomaly_indices=anomaly_indices,
        strip_fr_ratio=false_rejections,
        strip_fa_ratio=false_acceptances,
        clustering_score=clustering_score
    )
    
    return ScanResponse(
        status="success",
        target_class=target_class,
        trigger_type=trigger_type,
        fusion_score=final_score,
        risk_level=determine_risk_level(final_score),
        details={
            "sub_scores": sub_scores,
            "raw_metrics": {
                "neural_cleanse_anomaly_max": float(np.max(anomaly_indices)) if len(anomaly_indices)>0 else 0.0,
                "strip_fr_ratio": false_rejections,
                "strip_fa_ratio": false_acceptances,
                "clustering_silhouette": float(clustering_score)
            }
        }
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
