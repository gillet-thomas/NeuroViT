import yaml
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import load_img
from src.fmriEncoder import fmriEncoder


def main(ID=151, slice_dim=0, slice_idx=45):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
    FMRI_PATH = f"/mnt/data/iai/datasets/fMRI_marian/{ID}/wau4D.nii"
    config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = fmriEncoder(config).to(config["device"]).train()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)

    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 9)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config["device"])
    input_tensor = input_tensor.unsqueeze(0)          # Shape (1, 91, 90, 90)

    # Ensure model is in train mode
    model.eval()
    attention_map = model.get_attention_map(input_tensor)       # [90, 90, 90]
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=slice_dim, slice_idx=slice_idx)
    # nib.save(nib.Nifti1Image(attention_map, fmri_img.affine), f'{BASE_PATH}/gradcam_3dd.nii')

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[ :, :, slice_idx]  # Choose middle slice
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    # nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), f'{BASE_PATH}/gradcam_fmri.nii')
    print("GradCAM completed.")

    return ID, img, attn


if __name__ == '__main__':
    integers = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    results = []
    for i in integers:
        results.append(main(i))
        print(f"Completed {i}")

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle('GradCAM Results Across Subjects', fontsize=16)
    
    for idx, (ID, image, attention) in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.imshow(image, cmap='gray')
        heatmap = ax.imshow(attention, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
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
    plt.savefig('/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/vit4.png')
    plt.close()
    print("All results saved in single plot.")
