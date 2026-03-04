import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from dataset import get_cifar10_dataloaders
from models import get_resnet18
from defenses import NeuralCleanse, STRIP, SpectralSignatures, ActivationClustering
import os
import requests
import json
import time

# --- MLOps Dashboard Header ---
st.set_page_config(page_title="Gemini MLSecOps", layout="wide", page_icon="🛡️")
st.title("🛡️ Gemini Enterprise MLOps Command Center")
st.markdown("Automated Neural Trojan Auditing & Sanitization Pipeline")
st.divider()

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_URL = f"http://{FASTAPI_HOST}:8000"

def determine_risk_level(score: float) -> str:
    if score > 0.75:
        return "CRITICAL (Deployment Blocked)"
    elif score > 0.40:
        return "WARNING (Manual Review Required)"
    else:
        return "SAFE (Cleared for Production)"

# 1. Select Model Checkpoint
model_dir = "models"
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' not found.")
    st.stop()
    
st.sidebar.header("Scan Configuration")

upload_option = st.sidebar.radio("Model Source", ["Local Vault", "Upload External Model"])

if upload_option == "Local Vault":
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth") or f.endswith(".onnx")]
    selected_model_file = st.sidebar.selectbox("Select Model to Audit", model_files)
    model_path = os.path.join(model_dir, selected_model_file)
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload PyTorch/ONNX Model", type=["pth", "pt", "onnx"])
    selected_model_file = uploaded_file.name if uploaded_file else "external_model.pth"
    model_path = None
    if uploaded_file is None:
        st.sidebar.info("Waiting for model upload...")

trigger_type_options = ["Auto-Detect (Black-Box)", "checkerboard", "square", "blending", "clean_label", "dynamic", "instagram_filter", "spatial_conditional"]
trigger_type = st.sidebar.selectbox("Expected Trigger Type", trigger_type_options)

target_class_options = ["Auto-Detect (Black-Box)"] + [str(i) for i in range(10)]
target_class_str = st.sidebar.selectbox("Target Class to Audit", target_class_options)
target_class = -1 if target_class_str == "Auto-Detect (Black-Box)" else int(target_class_str)

if st.sidebar.button("🚀 Execute Enterprise Audit (via API)"):
    if upload_option == "Upload External Model" and uploaded_file is None:
        st.sidebar.error("❌ Please upload a model file first!")
    else:
        with st.spinner("Submitting model to Asynchronous Queue..."):
            try:
                requests.get(f"{FASTAPI_URL}/health")
                
                if uploaded_file is not None:
                    files = {"model_file": (selected_model_file, uploaded_file.getvalue(), "application/octet-stream")}
                else:
                    with open(model_path, "rb") as f:
                        file_bytes = f.read()
                    files = {"model_file": (selected_model_file, file_bytes, "application/octet-stream")}
                    
                data = {"target_class": target_class, "trigger_type": trigger_type}
                
                # 1. Submit the Job
                start_time = time.time()
                response = requests.post(f"{FASTAPI_URL}/api/v1/scan-model", files=files, data=data)
                
                if response.status_code == 200:
                    task_info = response.json()
                    task_id = task_info["task_id"]
                    st.info(f"Job submitted successfully. Task ID: `{task_id}`")
                    
                    # 2. Poll for Completion
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    max_retries = 60 # 2 minutes max wait polling every 2s
                    completed = False
                    
                    for _ in range(max_retries):
                        status_res = requests.get(f"{FASTAPI_URL}/api/v1/scan-status/{task_id}")
                        if status_res.status_code == 200:
                            status_data = status_res.json()
                            status_text = status_data["status"]
                            msg = status_data.get("message", "Processing...")
                            
                            if status_text == 'PENDING':
                                status_placeholder.warning(f"⏳ {msg}")
                                progress_bar.progress(10)
                            elif status_text == 'PROGRESS':
                                status_placeholder.info(f"🔄 {msg}")
                                progress_bar.progress(50)
                            elif status_text == 'SUCCESS':
                                status_placeholder.success("✅ Scan Complete!")
                                progress_bar.progress(100)
                                result = status_data["result"]
                                completed = True
                                break
                            elif status_text == 'FAILURE':
                                status_placeholder.error(f"❌ Scan Failed: {status_data.get('error')}")
                                progress_bar.progress(100)
                                break
                                
                        time.sleep(2.0)
                        
                    if not completed:
                        if status_text != 'FAILURE':
                            st.error("Scan timed out or is taking too long. Please check the server logs.")
                    else:
                        end_time = time.time()
                        st.subheader("Automated Scan Report")
                        st.write(f"**Scan Duration (including queue):** {end_time - start_time:.2f} seconds")
                        
                        # Display Fusion Score
                        score = result.get("fusion_risk_score", 0.0)
                        level = determine_risk_level(score)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unified Fusion Risk Score", f"{score * 100:.1f}%")
                            
                            if "CRITICAL" in level:
                                st.error(f"🚨 {level}")
                                st.progress(score)
                            elif "WARNING" in level:
                                st.warning(f"⚠️ {level}")
                                st.progress(score)
                            else:
                                st.success(f"✅ {level}")
                                st.progress(score)
                                
                            with st.expander("View Raw Scanner Metrics"):
                                st.json(result.get("details", {}))
                                
                            if result.get("is_onnx", False):
                                st.info("ℹ️ Model was ingested and analyzed dynamically via ONNX.")
                                
                        with col2:
                            st.subheader("Mechanistic Interpretability")
                            if result.get("gradcam_heatmap_b64"):
                                heatmap_bytes = base64.b64decode(result.get("gradcam_heatmap_b64"))
                                st.image(heatmap_bytes, caption="Grad-CAM Activation Heatmap", use_container_width=True)
                                st.caption("Visualizes the final convolutional layer's activation patterns.")
                            else:
                                st.info("No Grad-CAM heatmap generated (Layer not found or error).")
                                
                else:
                    st.error(f"API Error submitting scan: {response.text}")
                
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to the Backend Engine at {FASTAPI_URL}. Is the FastAPI server running?")

st.divider()
st.subheader("Manual Forensic Analysis (Local execution)")

if st.button("Load Local Metrics"):
    with st.spinner("Loading Model and Datasets..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        model_path = os.path.join(model_dir, selected_model_file)
        model = get_resnet18(num_classes=10)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            st.success(f"Model {selected_model_file} loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
            
        train_loader, test_clean, test_poisoned = get_cifar10_dataloaders(
            batch_size=128, poison_ratio=0.1, target_class=target_class, trigger_type=trigger_type
        )
        
    st.header("1. Neural Cleanse (Model-based)")
    with st.spinner("Running Neural Cleanse..."):
        nc = NeuralCleanse(model, device, num_classes=10)
        flagged, sizes, masks = nc.detect(test_clean, epochs=3)
        if len(flagged) > 0:
            st.warning(f"Trojan detected in classes: {flagged.tolist()}")
        else:
            st.success("No Trojan detected via Neural Cleanse.")
            
    st.header("2. STRIP (Data-based, Test-time)")
    with st.spinner("Running STRIP..."):
        strip = STRIP(model, device, test_clean.dataset)
        clean_entropies = []
        for i in range(20):
            img = test_clean.dataset[i][0].to(device)
            clean_entropies.append(strip.calculate_entropy(img, num_samples=32))
            
        poisoned_entropies = []
        for i in range(20):
            img = test_poisoned.dataset[i][0].to(device)
            poisoned_entropies.append(strip.calculate_entropy(img, num_samples=32))
            
        avg_clean = sum(clean_entropies)/len(clean_entropies)
        avg_poisoned = sum(poisoned_entropies)/len(poisoned_entropies)
        st.write(f"Average Entropy (Clean): {avg_clean:.4f}")
        st.write(f"Average Entropy (Poisoned): {avg_poisoned:.4f}")
        
    st.header("3. Spectral Signatures (Data-based, Train-time)")
    with st.spinner(f"Running Spectral Signatures on Class {target_class}..."):
        ss = SpectralSignatures(model, device, feature_layer_name='avgpool')
        flagged_indices, true_pos, total_pos = ss.detect(
            train_loader, 
            target_class=target_class, 
            expected_poison_ratio=0.1,
            margin=1.5
        )
        ss.remove_hook()
        
        if total_pos > 0:
            recall = 100. * true_pos / total_pos
            st.write(f"Detection Rate (Recall): {recall:.2f}% (Found {true_pos}/{total_pos} true poisons)")
            if recall > 80:
                st.success("Spectral Signatures successfully detected the majority of poisoned samples.")
            else:
                st.warning("Spectral Signatures had a low detection rate for this model.")
        else:
            st.info("No true poisons present in the training set for this class.")

    st.header("4. Activation Clustering (Feature-based, Train-time)")
    with st.spinner(f"Running Activation Clustering on Class {target_class}..."):
        ac = ActivationClustering(model, device, feature_layer_name='avgpool')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("K-Means")
            score_kmeans, labels_kmeans, _ = ac.detect(train_loader, target_class=target_class, method='kmeans')
            st.write(f"Silhouette Score (Separation): `{score_kmeans:.4f}`")
            if score_kmeans > 0.1:
                st.warning("High separation detected! Anomalous Trojan cluster likely present.")
            else:
                st.success("Low separation. Features are homogeneous.")
                
        with col2:
            st.subheader("DBSCAN")
            score_dbscan, labels_dbscan, _ = ac.detect(train_loader, target_class=target_class, method='dbscan')
            st.write(f"Silhouette Score (Separation): `{score_dbscan:.4f}`")
            if score_dbscan > 0.1:
                st.warning("High separation detected! Anomalous Trojan cluster likely present.")
            else:
                st.success("Low separation. Features are homogeneous.")
                
        ac.remove_hook()

    st.header("5. Model Sanitization (Fine-Pruning)")
    st.write("Mitigate Trojans by pruning 'dormant' neurons identified via a clean validation set.")
    
    prune_percentage = st.slider("Percentage of neurons to prune per step", min_value=1, max_value=20, value=10)
    max_steps = st.slider("Maximum pruning steps", min_value=1, max_value=15, value=10)
    
    if st.button("Run Fine-Pruning Mitigation"):
        import copy
        from defenses import FinePruning
        from train import test as evaluate_model
        
        with st.spinner("Executing Fine-Pruning..."):
            layer_name = 'layer4.1.conv2'
            fp = FinePruning(model, device, layer_name)
            
            # Use test_clean to get activations 
            activations = fp.get_activations(test_clean)
            num_channels = activations.shape[0]
            st.write(f"Target layer: `{layer_name}` ({num_channels} Total Channels)")
            
            results = []
            
            # Baseline
            cda_base = evaluate_model(model, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean")
            asr_base = evaluate_model(model, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned")
            results.append({"Pruned": 0, "Clean Data Acc (%)": cda_base, "Attack Success Rate (%)": asr_base})
            
            progress_bar = st.progress(0)
            
            step_size = max(1, int(num_channels * (prune_percentage / 100.0)))
            for step in range(1, max_steps + 1):
                num_prune = step * step_size
                if num_prune >= num_channels:
                    break
                    
                model_copy = copy.deepcopy(model)
                fp_iter = FinePruning(model_copy, device, layer_name)
                fp_iter.prune_neurons(num_prune, activations)
                
                cda = evaluate_model(model_copy, device, test_clean, torch.nn.CrossEntropyLoss(), name="Clean")
                asr = evaluate_model(model_copy, device, test_poisoned, torch.nn.CrossEntropyLoss(), name="Poisoned")
                
                results.append({"Pruned Neurons": num_prune, "Clean Data Acc (%)": cda, "Attack Success Rate (%)": asr})
                progress_bar.progress(step / max_steps)
                
                if cda < 10.0:
                    st.warning(f"Clean Data Accuracy plummeted after pruning {num_prune} neurons. Early stopping.")
                    break
                    
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df)
            st.line_chart(df.set_index("Pruned Neurons")[["Clean Data Acc (%)", "Attack Success Rate (%)"]])
