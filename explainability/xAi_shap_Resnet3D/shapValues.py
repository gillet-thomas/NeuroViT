import cv2
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from nilearn.image import load_img
import shap
import time

from src.models.fmriEncoder import fmriEncoder

def main(ID=151):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config['DEVICE'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = fmriEncoder(config).to(config['DEVICE']).eval()
    model.load_state_dict(torch.load(config['BEST_MODEL_PATH'], map_location=config['DEVICE']), strict=False)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)  # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1:, 70]  # CROP Shape: (90, 90, 9)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config['DEVICE']).unsqueeze(0)  # Shape (1, 91, 90, 90)

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[:, :, 45]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), f'{BASE_PATH}/xAi_shap/age/fmri{ID}.nii')

    # KernelExplainer requires a background dataset
    background = np.random.randn(10, 91, 90, 90) * 0.01  # Small noise for baseline

    # KernelExplainer requires input as (batch, features), so we flatten it
    input_flattened = input_tensor.cpu().numpy().reshape(1, -1)  # Shape (1, 91*90*90)

    # Update model wrapper to handle flattened input
    def model_wrapper(x):
        x_tensor = torch.tensor(x.reshape(-1, 91, 90, 90), dtype=torch.float32).to(config['DEVICE'])
        return model(x_tensor).detach().cpu().numpy()

    # Background data should also be flattened
    background = np.random.randn(10, 91, 90, 90) * 0.01  # Small noise baseline
    background_flattened = background.reshape(10, -1)  # Shape (10, 91*90*90)

    # Create KernelExplainer
    explainer = shap.KernelExplainer(model_wrapper, background_flattened)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_flattened, nsamples=100)

    # Process SHAP Output
    shap_values_data = np.array(shap_values[0])  # Convert to NumPy
    shap_values_slice = shap_values_data[:, :, 45]  # Middle slice
    shap_overlay = (shap_values_slice - shap_values_slice.min()) / (shap_values_slice.max() - shap_values_slice.min())
    shap_image = (fmri_rgb * 0.5 + shap_overlay[..., None] * 0.5)  # Blend fMRI with SHAP heatmap

    # Save SHAP results
    cv2.imwrite(f'{BASE_PATH}/xAi_shap/age/shap_age{ID}.jpg', shap_image * 255)
    print(f"SHAP analysis completed for ID {ID}.")

if __name__ == '__main__':
    integers = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    for i in integers:
        main(i)
        print(f"Completed {i}")
        time.sleep(1)
