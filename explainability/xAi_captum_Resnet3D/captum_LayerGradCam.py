import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from PIL import Image
from nilearn.image import load_img
import torch.nn.functional as F
import time

from src.models.NeuroEncoder import NeuroEncoder
from captum.attr import IntegratedGradients, LayerGradCam

def main(ID=151):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/NeuroViT/"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config['DEVICE'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = NeuroEncoder(config).to(config['DEVICE']).eval()
    model.load_state_dict(torch.load(config['BEST_MODEL_PATH'], map_location=config['DEVICE']), strict=False)
    target_layers = model.resnet_3d.resnet.layer4[-1]
    # target_layers = [model.encoder.vit3d.transformer.layers[-2][1].net[0]]

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 91)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config['DEVICE'])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)         # Shape (91, 90, 90)

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[ :, :, 45]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), f'{BASE_PATH}/xAi_captum/age/LayerGradCam_fmri{ID}.nii')

    # Set targets and compute CAM
    target = model(input_tensor).argmax(dim=1)
    print(f"Target: {target.item()}")
    layer_grad_cam = LayerGradCam(model, target_layers)
    attribution = layer_grad_cam.attribute(input_tensor, target=target) # attr ([1, 1, 6, 3, 3])
    # upsampled_attr = LayerAttribution.interpolate(attribution, (90, 90, 91))  # shape[2:] = (90, 90, 91)
    upsampled_attr = F.interpolate(attribution, size=input_tensor.shape[2:], mode="trilinear", align_corners=False) # Work better

    # Save CAM Nifti Image
    cam = upsampled_attr[0, 0, :].detach().cpu().numpy()    # Shape: (90, 90, 91)
    nib.save(nib.Nifti1Image(cam, fmri_img.affine), f'{BASE_PATH}/xAi_captum/age/LayerGradCam_heatmap{ID}.nii')

    slice_cam = cam[ :, :, 45]   # Shape: (90, 90)
    # Normalize slice_cam to 0-255 range for proper colormap application
    normalized_cam = slice_cam - np.min(slice_cam)
    normalized_cam = normalized_cam / np.max(normalized_cam) * 255
    normalized_cam = normalized_cam.astype(np.uint8)

    # Overlay needs image in 0-255 range
    fmri_rgb = fmri_rgb * 255
    fmri_rgb = fmri_rgb.astype(np.uint8)

    # Apply colormap and create overlay
    heatmap = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(fmri_rgb, 0.5, heatmap, 0.5, 0)

    # Save the overlay image
    cv2.imwrite(f'{BASE_PATH}/xAi_captum/age/LayerGradCam_age{ID}.jpg', overlay)
    cv2.imwrite(f'{BASE_PATH}/xAi_captum/age/LayerGradCam_age.jpg', overlay)
    print("LayerGradCam completed.")

if __name__ == '__main__':
    integers = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    for i in integers:
        main(i)
        print(f"Completed {i}")
        time.sleep(1)