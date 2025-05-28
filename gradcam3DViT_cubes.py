# Standard library imports
import os
import yaml
import warnings
from datetime import datetime

# Third-party imports
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Local application/library specific imports
from src.fmriEncoder import fmriEncoder
from src.data.DatasetGradCAM import GradCAMDataset


def get_sample_gradcam(id, save_sample_attention=False):

    sample = dataset[id] # sample is tuple (input_tensor, label, coordinates)
    input_tensor = sample[0].to(config['DEVICE']).unsqueeze(0)
    print(f"ID: {id} - Label: {sample[1].item()}, Coordinates: {sample[2].tolist()}")

    cube_size = config['GRADCAM_CUBE_SIZE']
    patch_x = int(sample[2][0] + cube_size // 2)
    patch_y = int(sample[2][1] + cube_size // 2)
    patch_z = int(sample[2][2] + cube_size // 2)
    # print(f"Patch coordinates: {patch_x}, {patch_y}, {patch_z}")

    # Get attention map
    attention_map, class_idx = model.get_attention_map(input_tensor)
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=0, slice_idx=patch_x)

    if save_sample_attention:
        nib.save(nib.Nifti1Image(attention_map.cpu().numpy(), np.eye(4)), f"{config['GRADCAM_OUTPUT_DIR']}/DatasetGradCAM_3Dattention_{id}.nii")
        save_gradcam_3d(attention_map, id, sample)
    
    return id, img, attn, class_idx, sample[1]

def create_gradcam_plot(save_sample_attention=False):
    
    results = [get_sample_gradcam(id, save_sample_attention=save_sample_attention) for id in ids]

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle(f"DatasetGradCAM {config['TRAINING_VIT_INPUT_SIZE']}grid {config['GRADCAM_CUBE_SIZE']}cube {config['TRAINING_VIT_PATCH_SIZE']}patch {config['GRADCAM_BACKGROUND_NOISE']}noise", fontsize=16)
    
    # Plot each subject's results
    for idx, (ID, image, attention, class_idx, sample) in enumerate(results):
        row = idx // cols
        col = idx % cols
        ax = axes[col] if rows == 1 else axes[row, col]
        
        ax.imshow(-image+1 if config['GRADCAM_BACKGROUND_NOISE'] < 1 else image, cmap='gray')    # INVERSE BRIGHTNESS
        heatmap = ax.imshow(attention, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Subject {ID} (Class {class_idx.item()})")
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        (axes[col] if rows == 1 else axes[row, col]).axis('off')

    file_name = f"DatasetGradCAM_{config['TRAINING_VIT_INPUT_SIZE']}grid_{config['GRADCAM_CUBE_SIZE']}cube_{config['TRAINING_VIT_PATCH_SIZE']}patch_{config['GRADCAM_BACKGROUND_NOISE']}noise_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}".replace('.', 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(config['GRADCAM_OUTPUT_DIR'], f"{file_name}.png"), dpi=300)
    plt.close()
    print(f"All results saved to {file_name}.png")

def save_gradcam_3d(attention_map, id, sample):
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    threshold = config['GRADCAM_THRESHOLD_3D']
    coords = np.argwhere(attention_map.cpu().numpy() > threshold)
    values = attention_map.cpu().numpy()[attention_map.cpu().numpy() > threshold]

    # Scatter plot for the regions with attention above the threshold
    if coords.size > 0:
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap='jet', marker='s', alpha=0.6, s=50)
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Attention Value')
    else:
        print(f"No attention values above threshold {threshold} for sample {id}")

    # Add a bounding box for the volume
    ax.set(xlim=(0, attention_map.shape[0]), ylim=(0, attention_map.shape[1]), zlim=(0, attention_map.shape[2])) # Grid size
    ax.set(xlabel='X axis', ylabel='Y axis', zlabel='Z axis')

    # Save figure and nifti file
    save_path = config['GRADCAM_OUTPUT_DIR']
    file_name = f"DatasetGradCAM_{config['TRAINING_VIT_INPUT_SIZE']}grid_{config['GRADCAM_CUBE_SIZE']}cube_{config['GRADCAM_BACKGROUND_NOISE']}noise_3Dattention_{id}".replace('.', 'p')
    plt.title(f"3D GradCAM (Label: {sample[1].item()}, coordinates: {sample[2].tolist()})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=300)
    plt.close()

if __name__ == '__main__':
    # Config 
    warnings.simplefilter(action='ignore', category=FutureWarning)
    config = yaml.safe_load(open("/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/configs/config.yaml"))
    config['DEVICE'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Model and Dataset
    model = fmriEncoder(config).to(config['DEVICE']).eval()
    best_model_path = os.path.join(config['GLOBAL_BASE_PATH'], config['BEST_MODEL_PATH'])
    model.load_state_dict(torch.load(best_model_path, map_location=config['DEVICE']), strict=False)
    dataset = GradCAMDataset(config, mode="val", generate_data=False)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    create_gradcam_plot(save_sample_attention=config['GRADCAM_SAVE_ATTENTION'])
