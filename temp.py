import torch
from vit_pytorch.simple_vit_3d import SimpleViT

v = SimpleViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

f = SimpleViT(
    image_size = 64,          # image size
    frames = 1,               # number of frames
    channels = 33,
    image_patch_size = 16,     # image patch size
    frame_patch_size = 1,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

# video = torch.randn(1, 3, 16, 128, 128) # (batch, channels, frames, height, width)
# preds = v(video) # (4, 1000)
# print(preds.shape)

fmri = torch.randn(1, 33, 1, 64, 64) # (1, 33, 1, 64, 64)
preds = f(fmri) # (4, 1000)
print(preds.shape)
