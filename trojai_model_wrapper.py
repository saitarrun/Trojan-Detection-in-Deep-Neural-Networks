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
            self.model = torch.load(model_path_or_model, map_location=device)
        else:
            self.model = model_path_or_model
            
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_layer_name = self._find_penultimate_layer()
        print(f"[TrojAI Wrapper] Discovered dynamic feature layer: '{self.feature_layer_name}'")

    def _find_penultimate_layer(self):
        """
        Heuristic algorithm to traverse the network graph and find the last 
        pooling layer or flattening layer before the standard fully connected head.
        """
        # Common TrojAI last layer names before the FC layer
        candidates = ['avgpool', 'adaptive_avg_pool2d', 'classifier', 'fc']
        
        last_found = None
        for name, module in self.model.named_modules():
            if any(candidate in name.lower() for candidate in candidates):
                # We want the pooling layer if it exists, otherwise the layer right before FC
                if 'pool' in name.lower() or 'feature' in name.lower():
                    last_found = name
                    
        return f"model.{last_found}" if last_found else "model.features"

    def forward(self, x):
        return self.model(x)
