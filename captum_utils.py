import torch
from captum.attr import IntegratedGradients
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import matplotlib.cm as cm

class CaptumSaliency:
    """
    Captum-based Explainability (Integrated Gradients).
    Unlike Grad-CAM, this attributes directly to the input pixels and doesn't
    rely on tracking intermediate layer gradients (which fails on ONNX or generic PyTorch models).
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    def generate_attribution(self, input_tensor, target_class):
        """
        Generates Integrated Gradients attribution for the target class.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        
        # Calculate attributions
        # We use a zero baseline by default
        attributions, delta = self.ig.attribute(
            input_tensor, 
            target=target_class, 
            return_convergence_delta=True
        )
        
        # Convert to numpy and aggregate across color channels (C, H, W) -> (H, W)
        attr_np = attributions.squeeze(0).cpu().detach().numpy()
        attr_np = np.sum(attr_np, axis=0)
        
        # Normalize and convert to absolute attribution
        attr_np = np.abs(attr_np)
        max_val = np.max(attr_np)
        if max_val > 1e-8:
            attr_np /= max_val
            
        return attr_np

    def visualize(self, input_tensor, attribution_map):
        """
        Overlays the attribution heatmap on the original image.
        """
        # Get base image
        img_np = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.2) + 0.5 # De-normalize approx CIFAR
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        base_img = Image.fromarray(img_np).resize((224, 224), Image.Resampling.NEAREST)

        # Apply colormap to heatmap (inferno/magma are good for attribution)
        heatmap_colored = cm.inferno(attribution_map)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_colored).resize((224, 224), Image.Resampling.BILINEAR)

        # Create overlay
        overlay = Image.blend(base_img.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.6)
        
        # Create a side-by-side or combined representation
        final_img = Image.new('RGB', (448, 224))
        final_img.paste(base_img, (0, 0))
        final_img.paste(overlay, (224, 0))
        
        return final_img

    def to_base64_jpeg(self, img_pil):
        byte_io = BytesIO()
        img_pil.save(byte_io, 'JPEG', quality=85)
        byte_io.seek(0)
        return base64.b64encode(byte_io.read()).decode('utf-8')
