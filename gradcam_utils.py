import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Ref: https://arxiv.org/abs/1610.02391
    Used for Mechanistic Interpretability (TrojAI Report Chapter 7.I)
    """
    def __init__(self, model, device, target_layer=None, target_layer_name=None):
        self.model = model
        self.device = device
        self.model.eval()
        
        if target_layer is None and target_layer_name is not None:
            # Attempt to find the layer by name
            for name, module in self.model.named_modules():
                if name == target_layer_name:
                    target_layer = module
                    break
        
        if target_layer is None:
             raise ValueError("GradCAM requires either target_layer or target_layer_name.")
             
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract gradients and activations
        self.hook_a = self.target_layer.register_forward_hook(self.save_activation)
        self.hook_b = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap for a given input tensor.
        Returns: (heatmap, overlay)
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
        
        # Generate Base Image for Overlay
        img_np = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        # Denormalize if it was CIFAR normalized (approx)
        img_np = (img_np * 0.2) + 0.5 
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Get pooled gradients and activations
        if self.gradients is None or self.activations is None:
            print("[GradCAM] Error: Backward pass failed to capture gradients/activations.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3])), img_np

        # Handle different dimensions (Conv vs Linear)
        if len(self.gradients.shape) == 4:
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations.detach()[0]
            
            # Weight activations by gradients
            for i in range(activations.shape[0]):
                activations[i, :, :] *= pooled_gradients[i]
            
            # Create heatmap
            heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        else:
            # Fallback for 1D or other shapes
            heatmap = np.abs(self.gradients.detach().cpu().numpy()[0])
            if len(heatmap.shape) > 1:
                heatmap = np.mean(heatmap, axis=0)
            
            # Resize using PIL instead of cv2
            img_pil = Image.fromarray(heatmap)
            img_pil = img_pil.resize((input_tensor.shape[3], input_tensor.shape[2]), Image.Resampling.BILINEAR)
            heatmap = np.array(img_pil)
        
        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        max_val = np.max(heatmap)
        if max_val > 1e-8:
            heatmap /= max_val
        else:
            heatmap = np.zeros_like(heatmap)
        
        # Apply ReLU again to clean up any tiny floating point noise
        heatmap = np.maximum(heatmap, 0)
        heatmap = np.nan_to_num(heatmap)
        
        overlay = self.overlay_heatmap(img_np, heatmap)
        
        return heatmap, overlay

    def visualize(self, input_tensor, heatmap, overlay):
        """
        Creates a side-by-side visualization.
        """
        img_np = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.2) + 0.5 
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # Concatenate horizontally
        combined = np.hstack([img_np, overlay])
        return combined

    def to_base64_jpeg(self, image_np):
        import io
        from PIL import Image
        import base64
        
        img = Image.fromarray(image_np)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_b.remove()

    @staticmethod
    def overlay_heatmap(original_image, heatmap, alpha=0.5):
        """
        Overlays the generated heatmap onto the original image.
        original_image: numpy array (H, W, 3) in [0, 255] RGB format
        heatmap: numpy array (H, W) in [0, 1]
        """
        if hasattr(heatmap, 'detach'):
            heatmap = heatmap.detach().cpu().numpy()
            
        # Resize heatmap using PIL
        heatmap_pil = Image.fromarray(heatmap)
        heatmap_pil = heatmap_pil.resize((original_image.shape[1], original_image.shape[0]), Image.Resampling.BILINEAR)
        heatmap_resized = np.array(heatmap_pil)
        
        # Apply colormap using matplotlib instead of cv2
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_resized)
        
        # Drop alpha channel from colormap and convert to 0-255 RGB
        heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # Superimpose the heatmap onto the original image
        superimposed_img = np.uint8(heatmap_colored * alpha + original_image * (1.0 - alpha))
        
        return superimposed_img
