import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from PIL import Image
from nilearn.image import load_img
import time

from src.models.NeuroEncoder import NeuroEncoder
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMElementWise
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

def main(ID=151):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    # FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/structural/s{ID}.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config['DEVICE'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = NeuroEncoder(config).to(config['DEVICE']).eval()
    model.load_state_dict(torch.load(config['BEST_MODEL_PATH'], map_location=config['DEVICE']), strict=False)
    # target_layers = [model.encoder.vit3d.transformer.layers[-2][1].net[0]]  # Last norm layer before the last attention layer, output (1, 2)
    # target_layers = [model.resnet_video.resnet_blocks[4].res_blocks[0].branch2.conv_a]  # Last norm layer before the last attention layer, output (1, 2)
    target_layers = [model.resnet_3d.resnet.layer4[-1]] 
    # print(target_layers)
    cam = LayerCAM(model=model, target_layers=target_layers)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    # fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 9)
    fmri_data = fmri_data[:,:,8:168]                        # CROP Shape: (90, 90, 9)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config['DEVICE'])
    input_tensor = input_tensor.unsqueeze(0)          # Shape (1, 91, 90, 90)
    cv2.imwrite(f'{BASE_PATH}/xAi_gradcam/input.jpg',fmri_data[:, 172, :]) # 256, 160

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[ :, 172, : ]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_norm, fmri_img.affine), f'{BASE_PATH}/xAi_gradcam/structural/gradcam_fmri{ID}.nii')

    # Set targets and compute CAM
    target = model(input_tensor).argmax(dim=1)
    print(f"Target: {target.item()}")
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Save CAM Nifti Image
    print("Grayscale CAM shape: ", grayscale_cam.shape, type(grayscale_cam), grayscale_cam.min(), grayscale_cam.max())
    grayscale_cam = grayscale_cam[0, :, :, :]  # Shape: (1, 1, 91, 90, 90) -> (91, 90, 90)
    grayscale_cam = np.array(grayscale_cam)
    grayscale_cam = np.transpose(grayscale_cam, (1, 2, 0))
    grayscale_cam_rgb = torch.tensor(grayscale_cam[ :, 172, :])    # Shape: (1, 90, 91, 90) -> (90, 91, 90)
    print("Grayscale CAM shape: ", grayscale_cam.shape, type(grayscale_cam), grayscale_cam.min(), grayscale_cam.max())
    # grayscale_cam_rgb = np.stack([grayscale_cam_rgb] * 3, axis=-1)

    nib.save(nib.Nifti1Image(grayscale_cam, fmri_img.affine), f'{BASE_PATH}/xAi_gradcam/structural/gradcam_heatmap{ID}.nii')
    # grayscale_cam = grayscale_cam[ :, :, 45]   # Shape: (90, 90)

    # Overlay CAM on fMRI image
    print("Fmri RGB shape: ", fmri_rgb.shape, fmri_rgb.min(), fmri_rgb.max(), type(fmri_rgb))
    print("Grayscale shape: ", grayscale_cam.shape, grayscale_cam.min(), grayscale_cam.max(), type(grayscale_cam))
    # print("Grayscale RGB shape: ", grayscale_cam_rgb.shape, grayscale_cam_rgb.min(), grayscale_cam_rgb.max())
    cam_image = show_cam_on_image(fmri_rgb, grayscale_cam_rgb)
    cv2.imwrite(f'{BASE_PATH}/xAi_gradcam/structural/gradcam_age{ID}.jpg', cam_image)
    cv2.imwrite(f'{BASE_PATH}/xAi_gradcam/output.jpg', cam_image)
    print("GradCAM completed.")

if __name__ == '__main__':
    ids = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    results = []
    
    for i in ids:
        main(i)
        print(f"Completed {i}")
