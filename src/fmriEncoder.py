import torch.nn as nn
import torch
from vit_pytorch.vit_3d import ViT
 

class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ViT3DEncoder(config)         # All parameters are trainable
        self.projection = ProjectionHead(config)    # All parameters are trainable
        
        self.device = config["device"]
        self.to(self.device)  # Move entire model to device at once

    def forward(self, x):
        timepoints_encodings = self.encoder(x) # Output is batch_size, 1024
        timepoints_encodings = self.projection(timepoints_encodings) # 32, 1024 linear to 32, 4
        return timepoints_encodings
    

class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.dropout = config["dropout"]

        self.encoder = ViT(
            frames = 91,               # number of frames (fmri slices)
            image_size = 91,           # image size (64x64)
            channels = 1,              # number of channels (one channel for each fmri slice)
            frame_patch_size = 1,      # number of frames processed at once
            image_patch_size = 13,     # size of 2D patches extracted from each frame (common for ViT models)
            num_classes = 1024,        # embedding dimension
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = self.dropout,
            emb_dropout = self.dropout
        ).to(self.device)
    
    def forward(self, x):
        # x is fmri tensor of shape (batch_size, 64, 64, 48)
        timepoint = x.to(self.device)
        timepoint = timepoint.permute(0, 3, 1, 2)           # ([batch_size, 48, 64, 64]) batch, frames, height, width
        timepoint = timepoint.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 48, 64, 64])
        encoding = self.encoder(timepoint)                  # Encode each timepoint with 3D-ViT   

        return encoding
    
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
            nn.Linear(512, 4)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device)

        self.projection2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 4)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device)

        self.projection3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 4)  # 4 classes: EMCI, CN, LMCI, AD
        ).to(self.device) 

    def forward(self, x):
        # x is a tensor of shape (batch_size, 1024)
        logits = self.projection3(x)  # shape (batch_size, 4)
        return logits
