
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import os
import shutil
import tempfile
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
import base64

# Import internal modules
from models import get_resnet18
from dataset import get_cifar10_dataloaders
from defenses import NeuralCleanse, STRIP, ActivationClustering, RiskFusionEngine, WeightAnalysis
from gradcam_utils import GradCAM

app = FastAPI(title="Gemini Trojan Detection API", description="Enterprise MLOps API for auditing Deep Neural Networks for Trojans.")

class ScanResponse(BaseModel):
    status: str
    model_analyzed: str
    fusion_risk_score: float
    details: dict
    gradcam_heatmap_b64: str | None

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
    Scans an uploaded PyTorch model for Neural Trojans using Neural Cleanse, STRIP, Activation Clustering, and Weight Analysis.
    Generates a Grad-CAM heatmap for interpretability.
    """
    if not model_file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Only .pth PyTorch model files are supported.")
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Save uploaded file to a temporary location
    tmp_path = None
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
        
        # Load required datasets based on the requested trigger_type test
        train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
            batch_size=128, poison_ratio=0.1, target_class=target_class, trigger_type=trigger_type
        )
        
        # 1. Run Neural Cleanse
        nc = NeuralCleanse(model, device, num_classes=10)
        flagged, sizes, masks = nc.detect(test_clean, epochs=3)
        nc_anomaly_indices = []
        if len(sizes) > 0:
            median = np.median(sizes)
            mad = np.median(np.abs(sizes - median))
            if mad < 1e-4: mad = 1e-4
            nc_anomaly_indices = np.abs(sizes - median) / (mad * 1.4826)
            
        # 2. Run STRIP
        strip = STRIP(model, device, test_clean.dataset)
        clean_entropies = [strip.calculate_entropy(test_clean.dataset[i][0].to(device)) for i in range(10)]
        poisoned_entropies = [strip.calculate_entropy(test_poisoned.dataset[i][0].to(device)) for i in range(10)]
        
        threshold = (np.mean(clean_entropies) + np.mean(poisoned_entropies)) / 2
        strip_fr_ratio = sum(1 for e in clean_entropies if e < threshold) / 10.0
        strip_fa_ratio = sum(1 for e in poisoned_entropies if e > threshold) / 10.0
        
        # 3. Run Activation Clustering
        ac = ActivationClustering(model, device, feature_layer_name='avgpool')
        ac_score, _, _ = ac.detect(train_loader, target_class=target_class, method='kmeans')
        ac.remove_hook()

        # --- Defense 4: Linear Weight Analysis (Structural Anomaly) ---
        # The TrojAI Report notes weight analysis is fast because it requires no inputs
        wa = WeightAnalysis(model, device)
        wa_anomaly_indices = wa.detect()
        
        # Calculate Unified Risk Score
        fusion_engine = RiskFusionEngine()
        final_risk, details = fusion_engine.calculate_unified_risk(
            nc_anomaly_indices=nc_anomaly_indices,
            strip_fr_ratio=strip_fr_ratio,
            strip_fa_ratio=strip_fa_ratio,
            clustering_score=ac_score,
            wa_anomaly_indices=wa_anomaly_indices
        )
        
        # --- Mechanistic Interpretability: Grad-CAM ---
        # Look for the last convolutional layer dynamically 
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                
        heatmap_base64 = None
        if target_layer is not None:
            # Generate dummy input to trace gradients
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            grad_cam = GradCAM(model, target_layer, device)
            
            try:
                # We do a fast mock generation since we don't have the user's uploaded image tensor here
                heatmap_arr = grad_cam.generate_heatmap(dummy_input)
                # Convert the raw heatmap string to a transportable format (Base64 JPEG)
                heatmap_img = Image.fromarray(np.uint8(255 * heatmap_arr))
                heatmap_img = heatmap_img.convert('RGB')
                
                buffered = io.BytesIO()
                heatmap_img.save(buffered, format="JPEG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Grad-CAM error: {e}")

        return ScanResponse(
            status="success",
            model_analyzed=model_file.filename,
            fusion_risk_score=final_risk,
            details=details,
            gradcam_heatmap_b64=heatmap_base64
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process model: {str(e)}")
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
