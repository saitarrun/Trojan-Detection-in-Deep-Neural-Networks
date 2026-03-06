
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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Gemini Trojan Detection API", description="Enterprise MLOps API for auditing Deep Neural Networks for Trojans.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class AsyncScanResponse(BaseModel):
    status: str
    task_id: str
    message: str

class LocalPathScanRequest(BaseModel):
    model_path: str
    target_class: int = 0
    trigger_type: str = "checkerboard"

@app.post("/api/v1/scan-local-path", response_model=AsyncScanResponse)
async def scan_local_path(request: LocalPathScanRequest):
    """
    Triggers an audit for a model file already existing on the server filesystem.
    This bypasses the 528MB+ upload limit of proxies/ingresses.
    """
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail="Model file not found on server.")
        
    valid_extensions = (".pth", ".pt", ".onnx")
    if not any(request.model_path.endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {valid_extensions}")

    try:
        from celery_worker import run_model_scan_task
        task = run_model_scan_task.delay(request.model_path, request.target_class, request.trigger_type)
        
        return AsyncScanResponse(
            status="accepted",
            task_id=task.id,
            message=f"Local model {os.path.basename(request.model_path)} accepted for analysis."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit local model: {str(e)}")

@app.post("/api/v1/scan-model", response_model=AsyncScanResponse)
async def scan_model_async(
    model_file: UploadFile = File(...),
    target_class: int = Form(0),
    trigger_type: str = Form("checkerboard")
):
    """
    Submits a PyTorch (.pth) or ONNX (.onnx) model for asynchronous Neural Trojan auditing.
    """
    valid_extensions = (".pth", ".pt", ".onnx")
    if not any(model_file.filename.endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {valid_extensions}")
        
    try:
        # Save uploaded file to a persistent location for the worker to pick up
        os.makedirs("uploads", exist_ok=True)
        # Using the original extension to pass it to the worker
        ext = os.path.splitext(model_file.filename)[1]
        
        # NOTE: For ONNX files, we avoid mkstemp which assigns random names like tmpf7324.onnx.
        # This breaks ONNX models that rely on external data files (like dummy_model.onnx.data).
        # We save it using the exact filename they uploaded.
        if ext == ".onnx":
            tmp_path = os.path.join("uploads", model_file.filename)
        else:
             tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir="uploads")
             os.close(tmp_fd)
             
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
            
        # Dispatch to Celery
        from celery_worker import run_model_scan_task
        task = run_model_scan_task.delay(tmp_path, target_class, trigger_type)
        
        return AsyncScanResponse(
            status="accepted",
            task_id=task.id,
            message="Model accepted for asynchronous analysis."
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit model: {str(e)}")


@app.get("/api/v1/scan-status/{task_id}")
def get_scan_status(task_id: str):
    """
    Polls the Celery task status.
    """
    from celery.result import AsyncResult
    from celery_worker import celery_app
    
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "status": task_result.status,
        "task_id": task_id
    }
    
    if task_result.status == 'PENDING':
        response["message"] = "Task is waiting in queue..."
    elif task_result.status == 'PROGRESS':
        response["message"] = task_result.info.get('message', 'Processing...')
    elif task_result.status == 'SUCCESS':
        response["message"] = "Scan Complete"
        res = task_result.result
        if isinstance(res, dict):
            res['task_id'] = task_id
        response["result"] = res # Contains fusion_score, details, base64 image, and now task_id
    elif task_result.status == 'FAILURE':
        response["message"] = "Task failed to complete"
        response["error"] = str(task_result.info)
        
    return response


@app.get("/api/v1/audit-report/{task_id}")
def generate_standard_audit_report(task_id: str):
    """
    Generates a formal auditing report based on the Jan 2026 IARPA TrojAI standards.
    """
    from celery.result import AsyncResult
    from celery_worker import celery_app
    import datetime
    
    task_result = AsyncResult(task_id, app=celery_app)
    
    if not task_result.ready() or task_result.status != 'SUCCESS':
        raise HTTPException(status_code=400, detail="Report can only be generated for successful scans.")
        
    res = task_result.result
    details = res.get('details', {})
    
    # IARPA Standardized Report Structure
    report = {
        "report_metadata": {
            "version": "1.0-IARPA-JAN2026",
            "audit_timestamp": datetime.datetime.now().isoformat(),
            "task_id": task_id,
            "compliance_status": "Institutionalized AI Security Testing (IAST)"
        },
        "model_summary": {
            "architecture": "ResNet-18",
            "framework": "ONNX Runtime" if res.get('is_onnx') else "PyTorch",
            "risk_fusion_score": res.get('fusion_risk_score'),
            "verdict": determine_risk_level(res.get('fusion_risk_score'))
        },
        "trojan_forensics": {
            "trigger_inversion": {
                "neural_cleanse_index": max(details.get('nc_anomaly_indices', [0.0]) or [0.0]),
                "detected_target_classes": details.get('nc_flagged_classes', [])
            },
            "test_time_checks": {
                "strip_false_acceptance": details.get('strip_fa_ratio', 0.0),
                "strip_false_rejection": details.get('strip_fr_ratio', 0.0)
            },
            "weight_analysis": {
                "max_anomaly_l2_norm": max(details.get('wa_anomaly_indices', [0.0]) or [0.0])
            },
            "natural_vulnerability_profiling": {
                "shortcut_sensitivity": details.get('natural_sensitivity', 0.0),
                "classification_drift": details.get('natural_sensitivity', 0.0)
            }
        },
        "strategic_recommendations": [
            "Maintain defense-in-depth across the AI supply chain.",
            "Verify model provenance for internal deployments.",
            "Conduct continuous monitoring for low-ASR backdoors."
        ]
    }
    
    return report


@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
