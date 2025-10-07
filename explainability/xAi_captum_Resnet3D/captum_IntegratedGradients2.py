import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from PIL import Image
from nilearn.image import load_img
import time
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from src.models.NeuroEncoder import NeuroEncoder
import matplotlib.pyplot as plt

def main(ID=151):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config['DEVICE'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = NeuroEncoder(config).to(config['DEVICE']).eval()
    model.load_state_dict(torch.load(config['BEST_MODEL_PATH'], map_location=config['DEVICE']), strict=False)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]                       # CROP Shape: (90, 90, 90)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data) # Normalize
    input_tensor = torch.tensor(fmri_norm).float().to(config['DEVICE'])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)           # Shape (1, 1, 90, 90, 90)

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[:, :, 45]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), f'{BASE_PATH}/xAi_ig/age/fmri{ID}.nii')

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    target = model(input_tensor).argmax(dim=1)
    attributions = ig.attribute(input_tensor, target=target)
    attributions = attributions.squeeze().detach().cpu().numpy()  # Shape: (90, 90, 90)
    print(f"Target: {target.item()}")
    
    # Save full 3D attributions as NIFTI
    nib.save(nib.Nifti1Image(attributions, fmri_img.affine), f'{BASE_PATH}/xAi_ig/age/ig_age{ID}.nii')

    # Convert to 3-channel format expected by visualization
    attr_slice = attributions[:, :, 45]                # Get middle slice for visualization
    attr_slice_3d = np.stack([attr_slice]*3, axis=-1)  # Shape: (90, 90, 3)
    fmri_slice_3d = np.stack([fmri_slice]*3, axis=-1)  # Shape: (90, 90, 3)
    
    # Normalize for visualization
    attr_slice_3d = (attr_slice_3d - attr_slice_3d.min()) / (attr_slice_3d.max() - attr_slice_3d.min())
    fmri_slice_3d = (fmri_slice_3d - fmri_slice_3d.min()) / (fmri_slice_3d.max() - fmri_slice_3d.min())

    # Simple matplotlib visualization alternative
    plt.figure(figsize=(10, 10))
    plt.imshow(fmri_slice, cmap='gray')
    plt.imshow(attr_slice, cmap='hot', alpha=0.5)
    plt.colorbar()
    plt.savefig(f'{BASE_PATH}/xAi_ig/age/ig_age{ID}.jpg')
    plt.savefig(f'{BASE_PATH}/xAi_ig/age/ig_age.jpg')
    plt.close()
    print("Integrated Gradients completed.")

if __name__ == '__main__':
    integers = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    for i in integers:
        main(i)
        print(f"Completed {i}")
        time.sleep(1)