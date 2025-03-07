import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from nilearn.image import load_img

from src.fmriEncoder import fmriEncoder

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, depth=91, height=5, width=5):
    print("Before reshape: ", tensor.shape) # ([1, 820, 1024])

    # Remove CLS token, reshape into (batch, depth, height, width, channels)
    result = tensor[:, 1:, :].reshape(tensor.size(0), depth, height, width, tensor.size(2))  # [1, 91, 3, 3, 1024]
    print("After reshape: ", result.shape)  # Expected: [1, 91, 3, 3, 1024]

    # Bring the channels to the first dimension, like in CNNs
    slice_idx = 45
    result = result[:, slice_idx, :, :, :]  # [1, 3, 3, 1024]
    result = result.permute(0, 3, 1, 2)  # [1, 1024, 3, 3]
    

    print(f"Shape after slicing: {result.shape}")

    return result


if __name__ == '__main__':
    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = "/mnt/data/iai/datasets/fMRI_marian/151/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)
    target_layers = [model.encoder.encoder.transformer.layers[-2][1].net[0]]  # Last norm layer before the last attention layer, output (1, 2)
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


    # Load and Preprocess fMRI Data
    timepoint = 70
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[:, :, :, timepoint]                       # Shape: (91, 109, 91)
    fmri_data = fmri_data[1:, 10:-9, :]                             # CROP Shape: (90, 90, 91)
    fmri_data = (fmri_data - fmri_data.mean()) / fmri_data.std()    # Normalize
    input_tensor = torch.tensor(fmri_data).unsqueeze(0).unsqueeze(0).to(config["device"])                   # Shape (1, 1, 90, 90, 91)
    input_tensor = input_tensor.permute(0, 1, 4, 2, 3)              # Shape (1, 1, 91, 90, 90)

    # Save fMRI image for visualization
    fmri_image = fmri_data[:, :, 45]  # Choose middle slice
    fmri_image = (fmri_image - fmri_image.min()) / (fmri_image.max() - fmri_image.min())

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None
    # targets = [ClassifierOutputTarget(0)]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]    # Shape: (90, 91)
    grayscale_cam = grayscale_cam[:,45,:]
    # grayscale_cam = grayscale_cam[:90, :90]

    # Save CAM image for visualization
    fmri_rgb = np.stack([fmri_image] * 3, axis=-1)
    print("RGB fMRI shape:", fmri_rgb.shape)

    # Now use the RGB version with show_cam_on_image
    cam_image = show_cam_on_image(fmri_rgb, grayscale_cam)
    cv2.imwrite(BASE_PATH + 'gradcam/gradcam_vit.jpg', cam_image)