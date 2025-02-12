import torch.nn as nn
import torch
from vit_pytorch.vit_3d import ViT
 

class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.to(self.device)  # Move entire model to device at once

        self.encoder = ViT3DEncoder(config)
        self.projection = ProjectionHead(config)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on correct device
        timepoints_encodings = self.encoder(x)
        timepoints_encodings = self.projection(timepoints_encodings)
        return timepoints_encodings
    
class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.encoder = ViT(
            frames = 48,               # number of frames (fmri slices)
            image_size = 64,           # image size (64x64)
            channels = 1,              # number of channels (one channel for each fmri slice)
            frame_patch_size = 1,      # number of frames processed at once
            image_patch_size = 16,     # size of 2D patches extracted from each frame (common for ViT models)
            num_classes = 1024,        # embedding dimension
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    
    def forward(self, x):
        # x is fmri tensor of shape (batch_size, 64, 64, 48, 140)
        timepoints = x.unbind(4) # Unbind the 4th dimension (timepoints dim)

        timepoints_encodings = []
        for i, timepoint in enumerate(timepoints):              # 70 timepoints encoded (one out of two)
            timepoint = timepoint.permute(0, 3, 1, 2)           # ([batch_size, 48, 64, 64]) batch, frames, height, width
            timepoint = timepoint.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 48, 64, 64])
            encoding = self.encoder(timepoint)                  # Encode each timepoint with 3D-ViT                  
            timepoints_encodings.append(encoding)               

        vector_encodings = torch.stack(timepoints_encodings, dim=1) # shape (batch_size, 70, 1024)
        return vector_encodings

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(70, 1) # Map 70 timepoint encodings to a single vector

    def forward(self, x):
        # x is a tensor of shape (batch_size, 70, 1024)
        permuted_x = x.permute(0, 2, 1)             # shape (batch_size, 1024, 70)
        encodings = self.projection(permuted_x)     # shape (batch_size, 1024, 1)
        encodings = encodings.squeeze()             # shape (batch_size, 1024)

        return encodings
