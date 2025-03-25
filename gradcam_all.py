import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from nilearn.image import load_img
import time

from src.fmriEncoder import fmriEncoder
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMElementWise
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main(ID=151):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)
    target_layers = [model.resnet_3d.resnet.layer4[-1]] 
    cam = GradCAMElementWise(model=model, target_layers=target_layers)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]  # Crop
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)
    input_tensor = torch.tensor(fmri_norm).to(config["device"]).unsqueeze(0)

    # Prepare visualization
    fmri_slice = fmri_norm[:, :, 45]  # Middle slice
    fmri_rgb = np.stack([fmri_slice]*3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))

    # Compute CAM
    target = model(input_tensor).argmax(dim=1)
    print(f"Target: {target}")  
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Create CAM overlay
    grayscale_cam = grayscale_cam[0, :, :, 45]
    cam_image = show_cam_on_image(fmri_rgb, grayscale_cam)

    return ID, fmri_rgb, grayscale_cam, cam_image

if __name__ == '__main__':
    ids = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    results = []
    
    for i in ids:
        try:
            result = main(i)
            results.append(result)
            print(f"Completed {i}")
        except Exception as e:
            print(f"Failed on {i}: {str(e)}")
    
    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle('GradCAM Results Across Subjects', fontsize=16)
    
    for idx, (ID, fmri_rgb, grayscale_cam, cam_image) in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        ax.imshow(cam_image)
        ax.set_title(f'Subject {ID}')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/xAi_gradcam/all_gradcamEW_age.png')
    plt.close()
    print("All results saved in single plot.")