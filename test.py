import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

from src.fmriEncoder import fmriEncoder
class VitGradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM for a Vision Transformer model.
        
        Args:
            model: The trained ViT model
            target_layer: The target layer to compute GradCAM for (usually the last attention layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Register forward hook
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        
        # Register backward hook
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        # Save activations during forward pass
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        # Save gradients during backward pass
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        # Remove hooks when done
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM for the input tensor.
        
        Args:
            input_tensor: Input MRI volume (should be pre-processed according to your model's requirements)
            target_class: Target class for generating GradCAM. If None, uses the predicted class.
            
        Returns:
            cam: GradCAM heatmap with shape (10, 10, 10) for the 10×10×10 patches
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Forward pass
        model_output = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights based on gradients
        weights = torch.mean(self.gradients, dim=(0, 2))
        
        # Get activations
        activations = self.activations[0]  # Get activations for the first sample in batch
        
        # Weight the activations by the gradients
        weighted_activations = weights.unsqueeze(0) * activations
        
        # Sum along the channel dimension
        cam = torch.sum(weighted_activations, dim=1).detach().cpu()
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Reshape CAM to (10, 10, 10) to match the patch grid
        cam = cam[1:].reshape(10, 10, 10)  # Exclude the CLS token
        
        # Normalize CAM
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.numpy()

def visualize_gradcam_3d(original_volume, cam, slice_axis=0, num_slices=5, alpha=0.5):
    """
    Visualize GradCAM heatmap overlaid on the original MRI volume.
    
    Args:
        original_volume: Original MRI volume with shape (90, 90, 90)
        cam: GradCAM heatmap with shape (10, 10, 10)
        slice_axis: Axis to slice along (0: sagittal, 1: coronal, 2: axial)
        num_slices: Number of slices to display
        alpha: Transparency of the heatmap overlay
    """
    # Upscale CAM to match the original volume size
    upscaled_cam = np.zeros((90, 90, 90))
    
    # For each patch in the CAM, fill the corresponding voxels in the upscaled CAM
    for i in range(10):
        for j in range(10):
            for k in range(10):
                upscaled_cam[i*9:(i+1)*9, j*9:(j+1)*9, k*9:(k+1)*9] = cam[i, j, k]
    
    # Calculate slice indices
    slice_indices = np.linspace(0, original_volume.shape[slice_axis] - 1, num_slices, dtype=int)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_slices, figsize=(num_slices * 4, 4))
    
    # If only one subplot, wrap it in a list
    if num_slices == 1:
        axes = [axes]
    
    # Generate the visualizations
    for i, slice_idx in enumerate(slice_indices):
        if slice_axis == 0:
            orig_slice = original_volume[slice_idx, :, :]
            cam_slice = upscaled_cam[slice_idx, :, :]
        elif slice_axis == 1:
            orig_slice = original_volume[:, slice_idx, :]
            cam_slice = upscaled_cam[:, slice_idx, :]
        else:  # slice_axis == 2
            orig_slice = original_volume[:, :, slice_idx]
            cam_slice = upscaled_cam[:, :, slice_idx]
        
        # Display original slice
        axes[i].imshow(orig_slice, cmap='gray')
        
        # Overlay CAM with a colormap
        axes[i].imshow(cam_slice, cmap='jet', alpha=alpha)
        
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def apply_gradcam_to_vit(model, mri_volume, target_class=None):
    """
    Apply GradCAM to a ViT model for a 3D MRI volume.
    
    Args:
        model: The trained ViT model
        mri_volume: MRI volume with shape (90, 90, 90)
        target_class: Target class for GradCAM (None for predicted class)
    
    Returns:
        cam: GradCAM heatmap
    """
    # Get the last attention layer
    last_attn_layer = model.transformer.layers[-1][0]
    
    # Create GradCAM object
    grad_cam = VitGradCAM(model, last_attn_layer)
    
    # Preprocess the MRI volume
    # This assumes your model expects a batch of 1, channel of 1, and dimensions of 90x90x90
    input_tensor = torch.tensor(mri_volume).float().unsqueeze(0).unsqueeze(0)
    
    # Generate GradCAM
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    return cam

def visualize_patches_importance(model, mri_volume, target_class=None):
    """
    Visualize the importance of each patch in the MRI volume.
    
    Args:
        model: The trained ViT model
        mri_volume: MRI volume with shape (90, 90, 90)
        target_class: Target class for GradCAM (None for predicted class)
    """
    # Apply GradCAM
    cam = apply_gradcam_to_vit(model, mri_volume, target_class)
    
    # Visualize the CAM on different slices
    visualize_gradcam_3d(mri_volume, cam, slice_axis=0, num_slices=3)  # Sagittal view
    visualize_gradcam_3d(mri_volume, cam, slice_axis=1, num_slices=3)  # Coronal view
    visualize_gradcam_3d(mri_volume, cam, slice_axis=2, num_slices=3)  # Axial view

# Example usage:
# model = ViT(...) # Your trained model
# mri_volume = np.load('mri_data.npy')  # Your MRI volume with shape (90, 90, 90)
# visualize_patches_importance(model, mri_volume)
# Load your trained model

model = ViT(
    image_size=90,
    image_patch_size=9,
    frames=90,
    frame_patch_size=9,
    num_classes=2,  # Adjust based on your classification task
    dim=1024,
    depth=12,
    heads=16,
    mlp_dim=4096
)
model.load_state_dict(torch.load('your_model_weights.pth'))

# Load your MRI volume (90×90×90)
mri_volume = np.load('your_mri_volume.npy')

# Generate and visualize GradCAM
visualize_patches_importance(model, mri_volume)