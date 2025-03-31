import torch.nn as nn
import torch
from vit_pytorch.vit_3d import ViT
import torch.nn.functional as F
from src.resnet3d import ResNet, generate_model
import matplotlib.pyplot as plt
import nibabel as nib

class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        
        self.encoder = ViT3DEncoder(config)         # All parameters are trainable
        self.projection = ProjectionHead(config)    # All parameters are trainable
        self.resnet_video = ResnetVideo(config)
        self.resnet_3d = Resnet3D(config)

        self.to(self.device)  # Move entire model to device at once

    def forward(self, x):
        # x is a tensor of shape (batch_size, 90, 90, 91)
        timepoints_encodings = self.encoder(x)   # Encode each timepoint with 3D-ViT
        timepoints_encodings = self.projection(timepoints_encodings) # batch, 1024 linear to batch, 2
        return timepoints_encodings
    
class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dropout = config["dropout"]

        self.vit3d = ViT(
            frames=90,
            image_size=90,
            channels=1,
            frame_patch_size=9,
            image_patch_size=9,
            num_classes=1024,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=self.dropout,
            emb_dropout=self.dropout
        ).to(self.device)

        # More robust gradient and activation tracking
        self.gradients = {}
        self.activations = {}
        
        # Register hooks on all attention layers for more comprehensive tracking
        self._register_hooks()

    def _register_hooks(self):
        """More robust hook registration across all attention layers"""
        # Get the last attention layer
        last_attention = self.vit3d.transformer.layers[-1][0].norm

        def forward_hook(module, input, output):
            self.activations = output.detach().cpu() # [1, 1001, 1024]
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().cpu() # [1, 1001, 1024]
        
        # Register hooks
        self.forward_handle = last_attention.register_forward_hook(forward_hook)
        self.backward_handle = last_attention.register_backward_hook(backward_hook)

    def forward(self, x):
        """Forward pass with optional attention tracking"""
        # Input preprocessing (same as before)
        timepoint = x.to(self.device)
        if len(x.shape) == 4:
            timepoint = timepoint.permute(0, 3, 1, 2)
            timepoint = timepoint.unsqueeze(1)
        
        encoding = self.vit3d(timepoint)
        return encoding

    def get_attention_map(self, x):

        # Forward pass
        output = self.forward(x)
        
        # Get target class
        class_idx = output.argmax(dim=1)
        print(f"Class index: {class_idx}")  
        # Backward pass
        one_hot = torch.zeros_like(output)
        # one_hot.scatter_(1, class_idx.unsqueeze(1), 1)
        one_hot[torch.arange(output.size(0)), class_idx] = 1  # Create one-hot vector for target class
        output.backward(gradient=one_hot, retain_graph=True)  # Compute gradients for target class
        
        # Get gradients and activations from hooks
        gradients = self.gradients  # [1, 1001, 1024]
        activations = self.activations  # [1, 1001, 1024]
        
        # 1. Compute importance weights (global average pooling of gradients)
        weights = gradients.mean(dim=2, keepdim=True)  # [1, 1001, 1]
        
        # 2. Weight activations by importance and sum all features
        cam = (weights * activations).sum(dim=2)  # [1, 1001]
        
        # 3. Remove CLS token and process patches only
        patch_cam = cam[:, 1:]  # [1, 1000]
        
        # 4. Reshape to 3D patch grid (10x10x10)
        patch_cam = patch_cam.reshape(1, 10, 10, 10)
        
        # 5. Normalize patch_cam
        patch_cam = F.relu(patch_cam)
        patch_cam = (patch_cam - patch_cam.min()) / (patch_cam.max() - patch_cam.min())
        
        # 6. Upsample to original size
        cam_3d = F.interpolate(
            patch_cam.unsqueeze(0),  # [10, 10, 10]
            size=(90, 90, 90),
            mode='trilinear',
            align_corners=False
        ).squeeze()
        
        return cam_3d.detach().cpu().numpy()
    
    def visualize_slice(self, cam_3d, original_volume, slice_dim=0, slice_idx=None, save_path='./gradcam_visualization.png'):
        """Improved visualization with better error handling"""
        if cam_3d is None:
            print("Error: No CAM computed")
            return
        
        # Process original volume
        if torch.is_tensor(original_volume):
            original = original_volume.squeeze().permute(2, 0, 1).detach().cpu().numpy()
        else:
            original = original_volume.squeeze()
        
        # Verify shapes
        if original.ndim != 3 or cam_3d.ndim != 3:
            print(f"Shape mismatch: original {original.shape}, CAM {cam_3d.shape}")
            return
        
        # Default to middle slice
        if slice_idx is None:
            slice_idx = original.shape[slice_dim] // 2
        slice_idx = max(0, min(slice_idx, original.shape[slice_dim] - 1))
        
        # Select slice
        try:
            if slice_dim == 0:  # Axial
                img = original[slice_idx]
                attn = cam_3d[slice_idx]
            elif slice_dim == 1:  # Coronal
                img = original[:, slice_idx]
                attn = cam_3d[:, slice_idx]
            else:  # Sagittal
                img = original[:, :, slice_idx]
                attn = cam_3d[:, :, slice_idx]
        except IndexError:
            print(f"Slice {slice_idx} out of bounds for dim {slice_dim}")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original MRI')
        ax1.axis('off')
        
        # Overlay
        ax2.imshow(img, cmap='gray')
        heatmap = ax2.imshow(attn, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Grad-CAM Attention')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Visualization saved to {save_path}")
        
class ViT3DEncodere(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.dropout = config["dropout"]

        self.vit3d = ViT(
            frames = 90,               # number of frames (fmri slices)
            image_size = 90,           # image size (90x90)
            channels = 1,              # number of channels (one channel for each fmri slice)
            frame_patch_size = 9,      # number of frames processed at once
            image_patch_size = 9,      # size of 2D patches extracted from each frame
            num_classes = 1024,        # embedding dimension
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = self.dropout,
            emb_dropout = self.dropout
        ).to(self.device)

        self.gradients = None
        self.activations = None
        
        # Hook the last attention layer
        for layer in self.transformer.layers[-1][0].attend.modules():
            if isinstance(layer, nn.Softmax):
                layer.register_forward_hook(self.activation_hook)
                layer.register_backward_hook(self.gradient_hook)


    def activation_hook(self, module, input, output):
        self.activations = output.detach()
    
    def gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, x):
        timepoint = x.to(self.device)
        
        # x is fmri tensor of shape (batch_size, 90, 90, 91)
        if len(x.shape) == 4:
            timepoint = timepoint.permute(0, 3, 1, 2)           # ([batch_size, 91, 90, 90]) batch, frames, height, width
            timepoint = timepoint.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])
        
        encoding = self.vit3d(timepoint)                      # Encode each timepoint with 3D-ViT   
        return encoding  # Add channel dimension ([batch_size, 1, 1024])

class ResnetVideo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.resnet_video = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).eval().to(self.device)
        self.resnet_blocks = self.resnet_video.blocks[:-1]

    def forward(self, x):

        if len(x.shape) == 4:
            x = x.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])
        
        x = x.permute(0, 1, 4, 2, 3)           # ([batch_size, C, F, H, W])
        x = x.repeat(1, 3, 1, 1, 1)            # Shape (batch_size, 3, 91, 90, 90)

        # Process through each ResNet block manually
        for block in self.resnet_blocks:
            x = block(x)
        
        # Apply adaptive pooling to get a fixed-size output
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.shape[0], -1)  # Flatten to [batch_size, 2048]
        return x

class Resnet3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.resnet = generate_model(50, n_classes=1024)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])

        x = x.repeat(1, 3, 1, 1, 1)

        x = self.resnet(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dropout = config["dropout"]
        
        # First average across timepoints to get (batch_size, 1024)
        # Then project to 4 classes
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device)

        self.projection2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device)

        self.projection3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device) 

    def forward(self, x):
        # x is a tensor of shape (batch_size, 1024)
        logits = self.projection3(x)  # shape (batch_size, 2)
        return logits
