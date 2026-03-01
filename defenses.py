import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class NeuralCleanse:
    def __init__(self, model, device, input_shape=(3, 32, 32), num_classes=10):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model.eval()

    def reverse_engineer_trigger(self, target_class, dataloader, epochs=5, lambda_reg=1e-3):
        # Initialize trigger mask and pattern
        mask = torch.rand((1, self.input_shape[1], self.input_shape[2]), requires_grad=True, device=self.device)
        pattern = torch.rand(self.input_shape, requires_grad=True, device=self.device)
        
        optimizer = optim.Adam([mask, pattern], lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        last_loss = float('inf')
        patience = 2
        trigger_found_count = 0
        
        # We only need a small subset of data for this optimization since we are optimizing the trigger
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, batch in enumerate(dataloader):
                if i > 5: # Limit batches per epoch for extreme speed (Chapter 3.B optimization)
                    break
                    
                inputs = batch[0].to(self.device)
                
                m = torch.clamp(mask, 0, 1)
                p = torch.clamp(pattern, 0, 1)
                
                poisoned_inputs = (1 - m) * inputs + m * p
                
                optimizer.zero_grad()
                outputs = self.model(poisoned_inputs)
                
                labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=self.device)
                
                loss_ce = criterion(outputs, labels)
                loss_reg = lambda_reg * torch.sum(torch.abs(m))
                loss = loss_ce + loss_reg
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Early Stopping Check (Chapter 3.C)
            # If loss is very low, we've likely found a working trigger
            if epoch_loss < 0.01:
                trigger_found_count += 1
                if trigger_found_count >= patience:
                    print(f"      Early stop: Trigger converged at epoch {epoch}")
                    break
            
            # Or if loss is not improving significantly
            if abs(last_loss - epoch_loss) < 1e-4:
                print(f"      Early stop: Loss plateau at epoch {epoch}")
                break
            last_loss = epoch_loss
                
        return torch.clamp(mask, 0, 1).detach(), torch.clamp(pattern, 0, 1).detach()

    def detect(self, dataloader, epochs=3, target_class=None, callback=None):
        mask_sizes = []
        masks = []
        patterns = []
        
        # If target_class is specified, we perform a "Targeted Audit" (Fast Mode)
        # Otherwise, we perform a "Full Sweep" (Discovery Mode)
        search_space = range(self.num_classes) if target_class is None else [target_class]
        
        print(f"Running Neural Cleanse ({'Full Sweep' if target_class is None else 'Targeted Scan'})...")
        for i, c in enumerate(search_space):
            if callback:
                callback(i, len(search_space), c)
                
            # Default to fewer epochs for the discovery sweep (optimizing for latency)
            sweep_epochs = 2 if target_class is None else epochs
            m, p = self.reverse_engineer_trigger(c, dataloader, epochs=sweep_epochs)
            size = torch.sum(torch.abs(m)).item()
            mask_sizes.append(size)
            masks.append(m)
            patterns.append(p)
            print(f"Class {c} mask size: {size:.2f}")
            
        # Anomaly detection using MAD (Only valid for full sweeps with > 2 classes)
        if len(mask_sizes) > 2:
            median = np.median(mask_sizes)
            mad = np.median(np.abs(mask_sizes - median))
            if mad < 1e-4: mad = 1e-4
            anomaly_index = np.abs(mask_sizes - median) / (mad * 1.4826)
            print("\nAnomaly indices:", np.round(anomaly_index, 2))
            flagged_classes = np.where(anomaly_index > 2.0)[0]
        else:
            # For targeted scans, we just return the single class if it looks abnormal (heuristic)
            flagged_classes = np.array([target_class]) if target_class is not None else np.array([])
            mask_sizes = mask_sizes if target_class is not None else []
            
        return flagged_classes, mask_sizes, masks

class STRIP:
    def __init__(self, model, device, clean_dataset):
        self.model = model
        self.device = device
        self.clean_dataset = clean_dataset
        self.model.eval()
        
    def _superimpose(self, img1, img2, alpha=0.5):
        return alpha * img1 + (1 - alpha) * img2

    def calculate_entropy(self, input_tensor, num_samples=32):
        # input_tensor: [C, H, W]
        # Sample N clean images
        indices = np.random.choice(len(self.clean_dataset), num_samples, replace=False)
        perturbed_inputs = []
        
        for idx in indices:
            clean_img = self.clean_dataset[idx][0].to(self.device)
            p_img = self._superimpose(input_tensor, clean_img)
            perturbed_inputs.append(p_img)
            
        perturbed_batch = torch.stack(perturbed_inputs)
        
        with torch.no_grad():
            outputs = self.model(perturbed_batch)
            probs = torch.softmax(outputs, dim=1)
            
            # Compute average entropy 
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
            
        return torch.mean(entropy).item()

class SpectralSignatures:
    def __init__(self, model, device, feature_layer_name='avgpool'):
        self.model = model
        self.device = device
        # Inherit dynamic feature layer from TrojAI wrapper if it exists
        self.feature_layer_name = getattr(model, 'feature_layer_name', feature_layer_name)
        self.model.eval()
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output.detach())
        
        self.hook = None
        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                self.hook = module.register_forward_hook(hook_fn)
                break
                
        if self.hook is None:
            print(f"Warning: Could not find layer {self.feature_layer_name}")
                
    def get_representations(self, dataloader, target_class=None):
        all_features = []
        all_indices = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                is_poisoned = batch[2] if len(batch) > 2 else torch.zeros_like(labels).bool()
                
                self.features = [] # Clear features
                _ = self.model(inputs)
                
                if not self.features:
                    assert False, "Feature layer not found or hook not triggered."
                    
                batch_features = self.features[0].view(inputs.size(0), -1)
                
                if target_class is not None:
                    mask = (labels == target_class)
                    if mask.sum() > 0:
                        all_features.append(batch_features[mask])
                        all_indices.append(is_poisoned[mask])
                else:
                    all_features.append(batch_features)
                    all_indices.append(is_poisoned)
                    
        if len(all_features) == 0:
            return None, None
            
        all_features = torch.cat(all_features, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        return all_features, all_indices

    def detect(self, dataloader, target_class, expected_poison_ratio=0.1, margin=1.5):
        print(f"\n[Spectral Signatures] Analyzing class {target_class}...")
        features, is_poisoned_true = self.get_representations(dataloader, target_class)
        
        if features is None or features.size(0) == 0:
            print("No samples found for this class.")
            return []
            
        # 1. Center the features
        mean_feature = torch.mean(features, dim=0)
        centered_features = features - mean_feature
        
        # 2. Compute SVD
        _, _, V = torch.svd(centered_features)
        
        # Top right singular vector
        v = V[:, 0]
        
        # 3. Compute outlier scores (projections)
        scores = torch.matmul(centered_features, v)
        outlier_scores = scores ** 2
        
        # 4. Filter outliers
        num_expected_poisons = int(len(features) * expected_poison_ratio)
        k = int(num_expected_poisons * margin)
        k = min(k, len(outlier_scores) - 1)
        
        if k <= 0:
            print("No expected poisons based on ratio.")
            return [], 0, is_poisoned_true.sum().item()
            
        _, top_k_indices = torch.topk(outlier_scores, k)
        
        true_poisons_in_top_k = is_poisoned_true[top_k_indices].sum().item()
        total_true_poisons = is_poisoned_true.sum().item()
        
        print(f"Total samples for class {target_class}: {len(features)}")
        print(f"Total true poisoned samples present: {total_true_poisons}")
        print(f"Flagged {k} samples as poisoned.")
        print(f"True positives among flagged: {true_poisons_in_top_k}/{k}")
        
        return top_k_indices.cpu().numpy(), true_poisons_in_top_k, total_true_poisons

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

class ActivationClustering:
    def __init__(self, model, device, feature_layer_name):
        self.model = model
        self.device = device
        # Inherit dynamic feature layer from TrojAI wrapper if it exists
        self.feature_layer_name = getattr(model, 'feature_layer_name', feature_layer_name)
        self.model.eval()
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output.detach())
            
        self.hook = None
        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                self.hook = module.register_forward_hook(hook_fn)
                break
                
        if self.hook is None:
            print(f"Warning: Could not find layer {self.feature_layer_name}")

    def get_representations(self, dataloader, target_class):
        all_features = []
        all_indices = [] # keep track of ground truth poisons if available
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                is_poisoned = batch[2] if len(batch) > 2 else torch.zeros_like(labels).bool()
                
                self.features = []
                _ = self.model(inputs)
                
                if not self.features:
                    assert False, "Feature layer not found or hook not triggered."
                    
                batch_features = self.features[0].view(inputs.size(0), -1)
                
                # Filter strictly by target class
                mask = (labels == target_class)
                if mask.sum() > 0:
                    all_features.append(batch_features[mask])
                    all_indices.append(is_poisoned[mask])
                    
        if len(all_features) == 0:
            return None, None
            
        all_features = torch.cat(all_features, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        return all_features, all_indices

    def detect(self, dataloader, target_class, method='kmeans'):
        print(f"\n[Activation Clustering] Analyzing class {target_class} using {method.upper()}...")
        features, is_poisoned_true = self.get_representations(dataloader, target_class)
        
        if features is None or features.size(0) == 0:
            print("No samples found for this class.")
            return -1, None, None
            
        features_np = features.cpu().numpy()
        
        if method == 'kmeans':
            # We expect two clusters: Clean vs. Poisoned
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_labels = kmeans.fit_predict(features_np)
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=5.0, min_samples=10)
            cluster_labels = dbscan.fit_predict(features_np)
            
        # Calculate Silhouette Score to measure how well-separated the clusters are.
        # A high score (> 0.10 or 0.15) often indicates a distinct, separate Trojan cluster.
        # A low score (near 0) indicates normal, homogenous clean data.
        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(features_np, cluster_labels)
        else:
            score = 0.0
            print(f"[{method.upper()}] Only one cluster found, score is 0.")
        
        # Optional: Print out accuracy of the clustering if ground truth is known
        total_poisons = is_poisoned_true.sum().item()
        if total_poisons > 0:
            cluster_0_poisons = (is_poisoned_true[cluster_labels == 0]).sum().item()
            cluster_1_poisons = (is_poisoned_true[cluster_labels == 1]).sum().item()
            print(f"Total True Poisons: {total_poisons}")
            print(f"Poisons in Cluster 0: {cluster_0_poisons} / {np.sum(cluster_labels == 0)}")
            print(f"Poisons in Cluster 1: {cluster_1_poisons} / {np.sum(cluster_labels == 1)}")
            
        print(f"Silhouette Score (Separation Metric): {score:.4f}")
        
        return score, cluster_labels, features_np

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

class FinePruning:
    def __init__(self, model, device, layer_name):
        self.model = model
        self.device = device
        self.layer_name = layer_name
        self.layer = None
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.layer = module
                break
                
        if self.layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
            
    def get_activations(self, clean_dataloader):
        """
        Record the average activations for all channels in the target layer
        using a clean validation dataset.
        """
        self.model.eval()
        activations = []
        
        def hook_fn(module, input, output):
            # output shape: [batch, channels, H, W]
            # Average over batch, H, and W to get channel-wise activation
            chan_act = output.mean(dim=[0, 2, 3])
            activations.append(chan_act.detach())
            
        hook = self.layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for batch in clean_dataloader:
                inputs = batch[0].to(self.device)
                _ = self.model(inputs)
                
        hook.remove()
        
        # Average across all batches
        avg_activations = torch.stack(activations).mean(dim=0)
        return avg_activations
        
    def prune_neurons(self, num_neurons_to_prune, activations):
        """
        Prune the neurons with the lowest activations.
        """
        # Get indices of neurons sorted by activation (lowest first)
        sorted_indices = torch.argsort(activations)
        indices_to_prune = sorted_indices[:num_neurons_to_prune]
        
        if isinstance(self.layer, nn.Conv2d):
            weights = self.layer.weight.data
            bias = self.layer.bias.data if self.layer.bias is not None else None
            
            for idx in indices_to_prune:
                # Set weights and bias for the pruned filter to zero
                weights[idx, :, :, :] = 0.0
                if bias is not None:
                    bias[idx] = 0.0
                    
            self.layer.weight.data = weights
            if bias is not None:
                self.layer.bias.data = bias
                
            return indices_to_prune.tolist()
        else:
            raise NotImplementedError("Fine-Pruning currently supports Conv2d layers.")

class Unlearning:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def unlearn(self, clean_dataloader, trigger_mask, trigger_pattern, lr=0.01, epochs=1):
        """
        Retrain the model to 'unlearn' the Trojan by imposing the trigger on clean 
        inputs but assigning correct labels (or random labels). This associates the 
        trigger with non-malicious behavior.
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        m = trigger_mask.to(self.device)
        p = trigger_pattern.to(self.device)
        
        print("\n[Unlearning] Starting unlearning process...")
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(clean_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                # Apply the reverse-engineered trigger to clean inputs
                poisoned_inputs = (1 - m) * inputs + m * p
                
                optimizer.zero_grad()
                # Train the model to associate the poisoned input with its TRUE clean label
                outputs = self.model(poisoned_inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Loss: {running_loss/len(clean_dataloader):.4f}")
            
        print("[Unlearning] Finished.")

from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class RiskMetaClassifier:
    """
    A Meta-Classifier that learns how to optimally weight the outputs of 
    multiple standalone defense algorithms (NC, STRIP, AC, LWA) based on historical data.
    """
    def __init__(self, model_path="meta_classifier.pkl"):
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.clf = pickle.load(f)
                self.is_trained = True
                print(f"[RiskMetaClassifier] Loaded pre-trained model from {self.model_path}")

    def train(self, X, y):
        """
        X: array-like of shape (n_samples, 5) containing normalized risks [NC, STRIP, AC, LWA, NTP]
        y: array-like of shape (n_samples,) containing binary labels (0=clean, 1=poisoned)
        """
        print("[RiskMetaClassifier] Training Random Forest Meta-Classifier...")
        self.clf.fit(X, y)
        self.is_trained = True
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        print(f"[RiskMetaClassifier] Saved trained model to {self.model_path}")

    def predict_risk(self, features):
        """
        Predicts the probability of infection [0.0 - 1.0]
        features: numpy array of shape (1, 5) -> [nc_risk, strip_risk, ac_risk, lwa_risk, ntp_risk]
        """
        if not self.is_trained:
            raise ValueError("MetaClassifier is not trained yet!")
        
        # predict_proba returns [[prob_class_0, prob_class_1]]
        probs = self.clf.predict_proba(features)
        return probs[0][1] # Return probability of being class 1 (poisoned)

class NaturalTrojanProfiler:
    """
    Analyzes models for 'Natural Trojans' (Chapter 7.G of IARPA Jan 2026 Report).
    These are vulnerabilities where the model learns spurious shortcuts or 
    high-frequency dataset biases instead of robust features.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def profile_shortcuts(self, dataloader, num_batches=3):
        """
        Tests model sensitivity to 'shortcut' features by applying low-pass filters 
        (removing high-frequency details) and measuring prediction stability.
        Models with Natural Trojans often rely heavily on high-frequency noise 
        or specific background textures.
        """
        print("\n[Natural Trojan Profiler] Checking for shortcut dependencies...")
        sensitivities = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if i >= num_batches: break
                inputs = inputs.to(self.device)
                
                # Original predictions
                orig_outputs = self.model(inputs)
                orig_preds = torch.argmax(orig_outputs, dim=1)
                
                # Apply high-frequency suppression (Simplistic Shortcut Test)
                # We simulate this by adding small Gaussian noise or blurring
                # In a real IARPA scenario, we'd use frequency-domain masking.
                blurred_inputs = torch.nn.functional.avg_pool2d(inputs, kernel_size=3, stride=1, padding=1)
                shifted_outputs = self.model(blurred_inputs)
                shifted_preds = torch.argmax(shifted_outputs, dim=1)
                
                # Calculate what percentage of predictions changed due to minor structural loss
                change_ratio = (orig_preds != shifted_preds).float().mean().item()
                sensitivities.append(change_ratio)
        
        avg_sensitivity = np.mean(sensitivities)
        print(f"   Shortcut Sensitivity (Prediction Drift): {avg_sensitivity:.4f}")
        
        # High sensitivity to minor structural changes indicates a "Natural Trojan" (shortcut dependency)
        return avg_sensitivity

class RiskFusionEngine:
    def __init__(self, weights={'neural_cleanse': 0.20, 'strip': 0.25, 'clustering': 0.15, 'weight_analysis': 0.15, 'natural_profiler': 0.25}, use_meta_classifier=False):
        self.weights = weights
        self.use_meta_classifier = use_meta_classifier
        self.meta_classifier = RiskMetaClassifier() if use_meta_classifier else None

        
    def normalize_neural_cleanse(self, anomaly_indices):
        """
        Anomaly index usually needs to be > 2.0 to be flagged.
        We cap it at 4.0 for a max score of 1.0 (100% risk).
        """
        if len(anomaly_indices) == 0:
            return 0.0
        max_idx = np.max(anomaly_indices)
        if max_idx < 2.0:
            return 0.0
        
        normalized = (max_idx - 2.0) / 2.0
        return min(max(normalized, 0.0), 1.0)
        
    def normalize_strip(self, false_rejections_ratio, false_acceptances_ratio):
        """
        If STRIP successfully separates clean from poisoned, false ratios approach 0.
        If the model is clean (no Trojan), STRIP cannot separate them, so false ratios approach 0.5.
        Risk is inversely proportional to the false positive/negative rates.
        """
        avg_error = (false_rejections_ratio + false_acceptances_ratio) / 2.0
        # If error is high (e.g., 0.5), risk is 0. If error is 0, risk is 1.0.
        risk = 1.0 - (avg_error * 2.0)
        return min(max(risk, 0.0), 1.0)
        
    def normalize_clustering(self, silhouette_score):
        """
        Silhouette > 0.1 strongly implies an artificial cluster (Trojan).
        Score range: [-1, 1]. Cap risk at score = 0.25
        """
        if silhouette_score < 0.05:
            return 0.0
            
        normalized = (silhouette_score - 0.05) / 0.20
        return min(max(normalized, 0.0), 1.0)
        
    def normalize_weight_analysis(self, anomaly_indices):
        """
        Anomaly index based on MAD. Scores > 2.5 are flagged.
        Cap at 5.0 for a max score of 1.0.
        """
        if len(anomaly_indices) == 0:
            return 0.0
            
        max_idx = np.max(anomaly_indices)
        if max_idx < 2.5:
            return 0.0
            
        normalized = (max_idx - 2.5) / 2.5
        return min(max(normalized, 0.0), 1.0)

    def calculate_unified_risk(self, nc_anomaly_indices, strip_fr_ratio, strip_fa_ratio, clustering_score, wa_anomaly_indices=None, natural_sensitivity=0.0):
        """
        Outputs a final probability score [0.0 - 1.0] of model infection.
        """
        nc_risk = self.normalize_neural_cleanse(nc_anomaly_indices)
        strip_risk = self.normalize_strip(strip_fr_ratio, strip_fa_ratio)
        clustering_risk = self.normalize_clustering(clustering_score)
        wa_risk = self.normalize_weight_analysis(wa_anomaly_indices) if wa_anomaly_indices is not None else 0.0
        natural_risk = min(max(natural_sensitivity * 1.5, 0.0), 1.0) # Heuristic scaling
        
        details = {
            'neural_cleanse_risk': nc_risk,
            'strip_risk': strip_risk,
            'clustering_risk': clustering_risk,
            'weight_analysis_risk': wa_risk,
            'natural_trojan_risk': natural_risk
        }

        # Dynamic Fusion via Meta-Classifier
        if self.use_meta_classifier and self.meta_classifier and self.meta_classifier.is_trained:
            # We now use the standard 5-feature vector: [NC, STRIP, AC, LWA, NTP]
            try:
                features = np.array([[nc_risk, strip_risk, clustering_risk, wa_risk, natural_risk]])
                final_risk = self.meta_classifier.predict_risk(features)
                details['used_meta_classifier'] = True
            except Exception as e:
                # Fallback if meta-clf expects 4 features (legacy support)
                print(f"[RiskFusionEngine] Meta-Classifier prediction error (possible feature mismatch): {e}")
                features = np.array([[nc_risk, strip_risk, clustering_risk, wa_risk]])
                try:
                    final_risk = self.meta_classifier.predict_risk(features)
                    details['used_meta_classifier'] = "fallback_4_feature"
                except:
                    # Final fallback to weighted average
                    final_risk = (
                        nc_risk * self.weights['neural_cleanse'] +
                        strip_risk * self.weights['strip'] +
                        clustering_risk * self.weights['clustering'] +
                        wa_risk * self.weights['weight_analysis'] +
                        natural_risk * self.weights.get('natural_profiler', 0.25)
                    )
                    details['used_meta_classifier'] = "fallback_static"
        else:
            # Static Fallback
            final_risk = (
                nc_risk * self.weights['neural_cleanse'] +
                strip_risk * self.weights['strip'] +
                clustering_risk * self.weights['clustering'] +
                wa_risk * self.weights['weight_analysis'] +
                natural_risk * self.weights.get('natural_profiler', 0.25)
            )
            details['used_meta_classifier'] = False
        
        return final_risk, details
class WeightAnalysis:
    """
    Linear Weight Analysis (LWA) for Backdoor Detection.
    As per IARPA TrojAI Final Report (Chapter 4), this method inspects the 
    weights of the final classification layer for statistical anomalies (large L2 norms)
    which indicate a learned backdoor shortcut.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def detect(self):
        print("\n[Linear Weight Analysis] Analyzing final layer weights...")
        
        # Find the final linear/dense layer
        final_layer = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                final_layer = module
                print(f"   Found final classification layer: {name}")
                break
                
        if final_layer is None:
            print("   ❌ Error: Could not locate a final nn.Linear layer.")
            return []
            
        weights = final_layer.weight.data.clone().detach().cpu().numpy()
        
        # Calculate L2 norm for the weights of each class
        # weights shape: (num_classes, in_features)
        norms = np.linalg.norm(weights, axis=1)
        
        # Use Median Absolute Deviation (MAD) to find robust outliers
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        
        if mad == 0:
            print("   Warning: MAD is 0, cannot calculate anomaly index reliably.")
            return []
            
        anomaly_indices = np.abs(norms - median_norm) / mad
        
        # MAD anomaly threshold is typically > 2.0 or 3.0
        flagged_classes = np.where(anomaly_indices > 2.5)[0]
        
        print(f"   Median L2 Norm: {median_norm:.4f}, MAD: {mad:.4f}")
        if len(flagged_classes) > 0:
            print(f"   ⚠️ Flagged classes as anomalously large (Trojan shortcuts): {flagged_classes.tolist()}")
        else:
            print("   ✅ All class weight norms are within normal statistical bounds.")
            
        return anomaly_indices


