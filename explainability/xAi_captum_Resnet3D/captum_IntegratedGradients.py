import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from PIL import Image
from nilearn.image import load_img
import torch.nn.functional as F

from src.fmriEncoder import fmriEncoder
from captum.attr import IntegratedGradients, LayerGradCam

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = "/mnt/data/iai/datasets/fMRI_marian/154/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, : , 70]                        # CROP Shape: (90, 90, 91)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config["device"])
    print("Input tensor shape: ", input_tensor.shape)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)        # Shape (91, 90, 90)

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[:, :, 45]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), BASE_PATH + 'xAi_captum/IntegratedGradients_fmri.nii')

    # Set targets and compute CAM
    target = model(input_tensor).argmax(dim=1).item()
    layer_grad_cam = IntegratedGradients(model)
    attribution = layer_grad_cam.attribute(input_tensor, target=target) # attr ([1, 1, 6, 3, 3])
    print("Attribution shape: ", attribution.shape)
    # upsampled_attr = LayerAttribution.interpolate(attr, (90, 90, 91))  # shape[2:] = (90, 90, 91)
    # upsampled_attr = F.interpolate(attribution, size=input_tensor.shape[2:], mode="trilinear", align_corners=False) # Work better


    # Save CAM Nifti Image
    cam = attribution[0, 0, :].detach().cpu().numpy()    # Shape: (90, 91, 90)
    nib.save(nib.Nifti1Image(cam, fmri_img.affine), BASE_PATH + 'xAi_captum/IntegratedGradients_heatmap.nii')

    slice_cam = cam[ :, :, 45]   # Shape: (90, 90)
    # Normalize slice_cam to 0-255 range for proper colormap application
    normalized_cam = slice_cam - np.min(slice_cam)
    normalized_cam = normalized_cam / np.max(normalized_cam) * 255
    normalized_cam = normalized_cam.astype(np.uint8)

    # Convert fMRI slice to proper format for overlay
    fmri_rgb = fmri_rgb * 255
    fmri_rgb = fmri_rgb.astype(np.uint8)

    # Apply colormap and create overlay
    heatmap = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(fmri_rgb, 0.5, heatmap, 0.5, 0)

    # Save the overlay image
    cv2.imwrite(BASE_PATH + 'xAi_captum/IntegratedGradients.jpg', overlay)
    print("GradCAM overlay completed.")