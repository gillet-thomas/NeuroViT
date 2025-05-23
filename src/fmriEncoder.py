# Standard library imports
import os

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local application/library specific imports
from src.models.vit_3d import ViT


class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['DEVICE']
        
        self.volume_encoder = ViT3DEncoder(config)

        if config['TRAINING_DIM'] == 4:
            # Extract only ViT3D weights by filtering keys
            best_model_path = os.path.join(config['GLOBAL_BASE_PATH'], config['BEST_MODEL_PATH'])
            full_state_dict = torch.load(best_model_path)
            vit3d_state_dict = {
                k.replace("volume_encoder.vit3d.", "vit3d."): v 
                for k, v in full_state_dict.items() 
                if k.startswith("volume_encoder.vit3d.")
            }
            self.volume_encoder.load_state_dict(vit3d_state_dict, strict=True)

            for param in self.volume_encoder.parameters():
                param.requires_grad = False
            self.volume_encoder.eval()

            self.cls_token = nn.Parameter(torch.randn(1, 1, 2) * 0.01)
            self.temporal_transformer = TemporalTransformer(config)
            self.projection_head = ProjectionHead(config)

        self.to(self.device)  # Move entire model to device

        # Gradients and activations tracking
        self.gradients = {}
        self.activations = {}
        self.register_hooks() 

    def forward(self, fmri):

        if self.config['TRAINING_DIM'] == 3:
            fmri_encoding = self.volume_encoder(fmri) # [B, 1024]
        elif self.config['TRAINING_DIM'] == 4:
            fmri = fmri.permute(0, 4, 1, 2, 3) # Original: [B, H, W, D, T] -> New: [B, T, H, W, D]
            B, T, H, W, D = fmri.shape # Use T_orig to distinguish from new T
            volumes = fmri.reshape(B * T, H, W, D)
            volumes_encoding = self.volume_encoder(volumes) # [B*T, 1024]
            volumes_encoding = volumes_encoding.reshape(B, T, -1) # [B, T, 1024]

            cls_tokens = self.cls_token.expand(B, 1, 2) 
            volumes_encoding = torch.cat([cls_tokens, volumes_encoding], dim=1) # [B, T+1, 1024]

            fmri_encodings = self.temporal_transformer(volumes_encoding)  # Temporal transformer [B, T+1, 1024] -> [B, T+1, 1024]
            # fmri_encoding = fmri_encodings.mean(dim=1) # [B, 1024]
            fmri_encoding = fmri_encodings[:, 0, :] # [B, 1024]   # CLS tokens for classification 
            fmri_encoding = self.projection_head(fmri_encoding) # [B, 2]

        return fmri_encoding
    
    def register_hooks(self):
        # Get the last attention layer
        last_attention = self.volume_encoder.vit3d.transformer.layers[-1][0].norm

        def forward_hook(module, input, output):
            self.activations = output.detach().cpu()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().cpu()
        
        # Register hooks
        self.forward_handle = last_attention.register_forward_hook(forward_hook)
        self.backward_handle = last_attention.register_backward_hook(backward_hook)
    
    def get_attention_map(self, x):
        grid_size = self.config['TRAINING_VIT_INPUT_SIZE']
        patch_size = self.config['TRAINING_VIT_PATCH_SIZE']
        threshold = self.config['GRADCAM_THRESHOLD']

        # Forward pass to get target class
        output = self.forward(x)
        class_idx = output.argmax(dim=1)
        
        # Create one-hot vector for target class
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(output.size(0)), class_idx] = 1
        
        # Backward pass to get gradients and activations from hooks
        output.backward(gradient=one_hot, retain_graph=True) 
        gradients = self.gradients
        activations = self.activations

        # 1. Compute importance weights (global average pooling of gradients)
        weights = gradients.mean(dim=2, keepdim=True)
        # weights = gradients.abs().mean(dim=2, keepdim=True)
        # weights = gradients.max(dim=2, keepdim=True)[0] 
        # weights = F.relu(gradients).mean(dim=2, keepdim=True) 

        # 2. Weight activations by importance and sum all features
        cam = (weights * activations).sum(dim=2)  # [1, vit_tokens, dim] -> [1, vit_tokens]
        
        # 3. Remove CLS token and process patches only
        cam = cam[:, 1:]  # [1, vit_tokens-1]
        
        # 4. Reshape to 3D grid of patches
        cam_size = grid_size // patch_size
        cam = cam.reshape(1, cam_size, cam_size, cam_size)
        
        # 5. Normalize cam
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # [0, 1]
        threshold_value = np.percentile(cam, 100-threshold)
        thresholded_map = np.where(cam >= threshold_value, cam, 0)
        thresholded_map = torch.from_numpy(thresholded_map).unsqueeze(0)
        
        # 6. Upsample to original size
        cam_3d = F.interpolate(
            thresholded_map,
            size=(grid_size, grid_size, grid_size),
            mode='trilinear',
            align_corners=False
        ).squeeze()
        
        return cam_3d, class_idx
    
    def visualize_slice(self, cam_3d, original_volume, slice_dim=0, slice_idx=None):
        
        # Check if CAM is computed
        if cam_3d is None:
            print("Error: No CAM computed")
            return
        
        # Process original volume
        original = original_volume.squeeze()
        # original = original_volume.squeeze().permute(2, 0, 1) # [H, W, D] -> [D, H, W] for fMRIs
        original = original.detach().cpu().numpy()
        
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

        return img, attn


class TemporalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['DEVICE']
        encoder_layer = nn.TransformerEncoderLayer(d_model=2, nhead=2, batch_first=True) # input is [batch, timepoints, 1024]
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

    def forward(self, x):
        # x is a tensor of shape (batch_size, timepoints, 1024)
        logits = self.transformer(x)  # output is [batch_size, timepoints, 1024]
        return logits

class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['DEVICE']
        self.dropout = config['TRAINING_DROPOUT']
        self.grid_size = config['TRAINING_VIT_INPUT_SIZE']
        self.cube_size = config['GRADCAM_CUBE_SIZE']
        self.patch_size = config['TRAINING_VIT_PATCH_SIZE']
        self.num_cubes = (self.grid_size // self.cube_size) ** 3 # GradCAM Dataset: number of possible cube positions in grid

        self.vit3d = ViT(
            channels=1,
            image_size=self.grid_size,
            image_patch_size=self.patch_size,
            frames=self.grid_size,
            frame_patch_size=self.patch_size,
            num_classes=2,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            pool='cls' # works better than mean pooling
        ).to(self.device)

    def forward(self, x):
        # x is a 3D tensor of shape (batch_size, H, W, D)
        # ViT3D expects (batch_size, channels, frames, height, width)
        timepoint = x.to(self.device)
        timepoint = timepoint.permute(0, 3, 1, 2) # [batch, H, W, D] -> [batch, D, H, W] for fMRIs
        timepoint = timepoint.unsqueeze(1)

        encoding = self.vit3d(timepoint) # output is [batch, dim]
        return encoding

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['DEVICE']
        self.projection_head = nn.Linear(2, 2).to(self.device)
        # self.layernorm = nn.LayerNorm(2).to(self.device)

    def forward(self, x):
        # x is a tensor of shape (batch_size, 1024)
        x = self.projection_head(x) # output is [batch, 2]
        # x = self.layernorm(x)
        return x