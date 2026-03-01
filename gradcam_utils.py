import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Ref: https://arxiv.org/abs/1610.02391
    Used for Mechanistic Interpretability (TrojAI Report Chapter 7.I)
    """
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap for a given input tensor.
        """
        self.model.zero_grad()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Target for backprop
        score = output[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Get pooled gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        # Weight activations by gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Create heatmap
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        
        # Apply ReLU (we only care about features that have a positive influence)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        heatmap /= (np.max(heatmap) + 1e-8)
        
        return heatmap

    @staticmethod
    def overlay_heatmap(original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays the generated heatmap onto the original image.
        original_image: numpy array (H, W, 3) in [0, 255] RGB format
        heatmap: numpy array (H, W) in [0, 1]
        """
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB 8-bit colormap
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
        # Convert BGR (cv2 default) to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose the heatmap onto the original image
        superimposed_img = np.uint8(heatmap_colored * alpha + original_image * (1.0 - alpha))
        
        return superimposed_img
