import torch.nn as nn
from vit_pytorch.vit_3d import ViT
 

class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.to(self.device)  # Move entire model to device at once

        self.encoder = ViT3DEncoder(config)
        self.projection = ProjectionHead(config, embedding_dim=1024)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on correct device
        timepoints_encodings = self.encoder(x)
        print(f"Timepoints encodings type: {type(timepoints_encodings)}")
        print(f"Timepoints encodings shape: {timepoints_encodings[0].shape}")
        
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
        for i, timepoint in enumerate(timepoints[::2]):
            timepoint = timepoint.permute(0, 3, 1, 2)           # ([batch_size, 48, 64, 64]) batch, frames, height, width
            timepoint = timepoint.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 48, 64, 64])
            encoding = self.encoder(timepoint)                  # Encode each timepoint with 3D-ViT                  
            timepoints_encodings.append(encoding)

        print(f"Total timepoints encoded: {len(timepoints_encodings)}") # 70 timepoints encoded (one out of two)
        return timepoints_encodings

class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 1024 for fmri
        self.projection = nn.Linear(embedding_dim, config["projection_dim"])

    def forward(self, x):
        print(f"Projection head input shape: {x.shape}")
        return self.projection(x)
