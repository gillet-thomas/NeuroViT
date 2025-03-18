import torch.nn as nn
import torch
from vit_pytorch.vit_3d import ViT
import torch.nn.functional as F
from src.resnet3d import ResNet, generate_model

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
        timepoints_encodings = self.resnet_3d(x)   # Encode each timepoint with 3D-ViT
        timepoints_encodings = self.projection(timepoints_encodings) # batch, 1024 linear to batch, 2
        return timepoints_encodings
    

class ViT3DEncoder(nn.Module):
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
