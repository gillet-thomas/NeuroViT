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
        timepoints_encodings = self.projection(timepoints_encodings) # batch, 1024 linear to batch, 2
        return timepoints_encodings
    

class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.dropout = config["dropout"]

        self.encoder = ViT(
            frames = 91,               # number of frames (fmri slices)
            image_size = 90,           # image size (64x64)
            channels = 1,              # number of channels (one channel for each fmri slice)
            frame_patch_size = 1,      # number of frames processed at once
            image_patch_size = 18,     # size of 2D patches extracted from each frame
            num_classes = 1024,        # embedding dimension
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = self.dropout,
            emb_dropout = self.dropout
        ).to(self.device)
    
    def forward(self, x):
        # x is fmri tensor of shape (batch_size, 90, 90, 91)
        timepoint = x.to(self.device)
        if len(x.shape) == 4:
            timepoint = timepoint.permute(0, 3, 1, 2)           # ([batch_size, 91, 90, 90]) batch, frames, height, width
            timepoint = timepoint.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])
        encoding = self.encoder(timepoint)                  # Encode each timepoint with 3D-ViT   

        return encoding  # Add channel dimension ([batch_size, 1, 1024])
    
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
