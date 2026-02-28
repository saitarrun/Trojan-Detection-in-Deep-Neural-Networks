import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from dataset import get_cifar10_dataloaders
from models import get_resnet18
from defenses import NeuralCleanse, STRIP, SpectralSignatures
import os

st.title("Gemini Trojan Detection")
st.write("Evaluate Trojan Defenses on Deep Neural Networks.")

# 1. Select Model Checkpoint
model_dir = "models"
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' not found.")
    st.stop()
    
# Get list of models
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
selected_model_file = st.selectbox("Select Model Checkpoint", model_files)

# 2. Select Attack Settings to Load Correct Dataset
trigger_type = st.selectbox("Select Trigger Type Used during Training", ["checkerboard", "square", "blending", "clean_label", "dynamic"])
target_class = st.number_input("Target Class", min_value=0, max_value=9, value=0)

if st.button("Load Model and Run Evaluation"):
    with st.spinner("Loading Model and Datasets..."):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
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

    st.header("4. Model Sanitization (Fine-Pruning)")
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
