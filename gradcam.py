import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from PIL import Image
from nilearn.image import load_img

from src.fmriEncoder import fmriEncoder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform(tensor, depth=91, height=5, width=5):
    print("Before reshape: ", tensor.shape) # ([1, 820, 1024])

    # Remove CLS token, reshape into (batch, depth, height, width, channels)
    result = tensor[:, 1:, :].reshape(tensor.size(0), depth, height, width, tensor.size(2))  # [1, 91, height, width, 1024]
    print("After reshape: ", result.shape)  # Expected: [1, 91, height, width, 1024]

    # Bring the channels to the first dimension, like in CNNs
    result = result[:, 45, :, :, :]  # [1, height, width, 1024]
    result = result.permute(0, 3, 1, 2)  # [1, 1024, height, width]
    
    print(f"Shape after slicing: {result.shape}")

    return result

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = "/mnt/data/iai/datasets/fMRI_marian/151/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)
    # target_layers = [model.encoder.vit3d.transformer.layers[-2][1].net[0]]  # Last norm layer before the last attention layer, output (1, 2)
    # target_layers = [model.resnet_video.resnet_blocks[4].res_blocks[0].branch2.conv_a]  # Last norm layer before the last attention layer, output (1, 2)
    target_layers = [model.resnet_3d.resnet.layer4] 
    cam = GradCAM(model=model, target_layers=target_layers)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 91)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config["device"])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)          # Shape (1, 91, 90, 90)

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[:, :, 45]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), BASE_PATH + 'xAi_captum/fmri.nii')

    # Set targets and compute CAM
    target = model(input_tensor).argmax(dim=1)
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets) # ([1, 1, 91, 90, 90]) input tensor

    # Save CAM Nifti Image
    grayscale_cam = grayscale_cam[0, :]    # Shape: (1, 90, 91, 90) -> (90, 91, 90)
    nib.save(nib.Nifti1Image(grayscale_cam, fmri_img.affine), BASE_PATH + 'xAi_captum/gradcam.nii')
    grayscale_cam = grayscale_cam[ :, :, 45]   # Shape: (90, 90)

    # Overlay CAM on fMRI image
    cam_image = show_cam_on_image(fmri_rgb, grayscale_cam)
    cv2.imwrite(BASE_PATH + 'xAi_captum/gradcam_gender.jpg', cam_image)
    print("GradCAM completed.")