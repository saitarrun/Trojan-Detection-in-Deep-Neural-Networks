import torch
import torch.nn as nn

class TrojAI_ModelWrapper(nn.Module):
    """
    A universal wrapper for TrojAI PyTorch models (DenseNet, Inception, ResNet50).
    It dynamically identifies the penultimate layer (feature extractor) for defenses 
    like Activation Clustering and Spectral Signatures to hook into.
    """
    def __init__(self, model_path_or_model, device):
        super(TrojAI_ModelWrapper, self).__init__()
        self.device = device
        
        if isinstance(model_path_or_model, str):
            # Load the raw TrojAI model from .pt file
            # Note: TrojAI models are typically saved as entire model objects, not just state_dicts
            # We set weights_only=False to allow loading these full architecture objects.
            self.model = torch.load(model_path_or_model, map_location=device, weights_only=False)
        else:
            self.model = model_path_or_model
            
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_layer_name = self._find_penultimate_layer()
        print(f"[TrojAI Wrapper] Discovered dynamic feature layer: '{self.feature_layer_name}'")

    def _find_penultimate_layer(self):
        """
        Heuristic algorithm to traverse the network graph and find the last 
        pooling layer or convolutional layer before the standard fully connected head.
        This is critical for Grad-CAM and Activation Clustering.
        """
        # Common layer types we want to hook into
        target_types = (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)
        
        last_found_name = None
        
        # Traverse modules in order
        for name, module in self.model.named_modules():
            if isinstance(module, target_types):
                last_found_name = name
        
        # If we found something, use it. 
        # Note: In our wrapper, the model is self.model, so the hook needs to be on 'model.layer_name'
        if last_found_name:
            print(f"[TrojAI Wrapper] Successfully identified feature layer: '{last_found_name}'")
            return f"model.{last_found_name}"
            
        # Fallback for standard ResNet/DenseNet if naming is standard but types are wrapped
        candidates = ['avgpool', 'features.norm5', 'features.11', 'layer4.1.conv2']
        for cand in candidates:
            for name, module in self.model.named_modules():
                if cand in name:
                    print(f"[TrojAI Wrapper] Using fallback candidate layer: '{name}'")
                    return f"model.{name}"

        print("[TrojAI Wrapper] WARNING: No specific feature layer found. Defaulting to 'model' (Global). Grad-CAM may fail.")
        return "model" # Global fallback

    def forward(self, x):
        return self.model(x)
