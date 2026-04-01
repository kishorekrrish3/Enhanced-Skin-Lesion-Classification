import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import DEVICE, CLASS_NAMES, GRADCAM_DIR

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    """
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Robustly identify the internal resnet module
        try:
            resnet_model = self.model.resnet
        except (AttributeError, Exception):
            resnet_model = self.model
        
        # We hook into layer4 of the ResNet50
        target_layer = resnet_model.layer4[-1].conv3
        
        self.forward_handle = target_layer.register_forward_hook(self.save_feature_maps)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradients)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_image, class_idx=None):
        """
        Generate Grad-CAM heatmap for a specific image and class index.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        # If no specific class provided, explain the top prediction
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Target score
        score = output[:, class_idx]
        
        # Backward pass
        score.backward()
        
        # Get gradients and features
        gradients = self.gradients[0] # [C, H, W]
        features = self.feature_maps[0] # [C, H, W]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2)) # [C]
        
        # Weighted combination of feature maps
        cam = torch.zeros(features.shape[1:], dtype=torch.float32, device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * features[i]
            
        cam = F.relu(cam) # Apply ReLU (we only care about positive influences)
        
        # Normalize to [0, 1]
        cam -= torch.min(cam)
        cam /= torch.max(cam) + 1e-8
        
        return cam.cpu().numpy(), class_idx

def generate_heatmap_overlay(original_image_np, cam, alpha=0.5):
    """
    Combines original image with the CAM heatmap
    Original image should be RGB in range [0, 255]
    """
    # Resize cam to match image dimensions
    h, w, _ = original_image_np.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to pseudo-color map
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay loosely
    overlay = (1 - alpha) * original_image_np + alpha * heatmap
    overlay = np.uint8(overlay)
    
    return overlay

def generate_comparison_grid(model, image_tensor, original_image_np, target_class_idx=None, filename="gradcam.png"):
    """
    Generates a 1x2 grid: [Original, Grad-CAM Overlay]
    """
    grad_cam = GradCAM(model)
    cam, pred_idx = grad_cam.generate(image_tensor, class_idx=target_class_idx)
    
    overlay = generate_heatmap_overlay(original_image_np, cam)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    pred_name = CLASS_NAMES[pred_idx]
    title = f"Grad-CAM (Pred: {pred_name})"
    if target_class_idx is not None and target_class_idx != pred_idx:
        title += f"\nTarget: {CLASS_NAMES[target_class_idx]}"
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, filename), bbox_inches='tight', transparent=True)
    plt.close()
    
    return pred_idx
