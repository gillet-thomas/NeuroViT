import os
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from datetime import datetime
import matplotlib.pyplot as plt
from src.fmriEncoder import fmriEncoder
from src.data.DatasetADNI import ADNIDataset

def get_sample_gradcam(id, save_sample_attention=False):

    sample = dataset[id]
    input_tensor = sample[2].to(config['DEVICE']).unsqueeze(0)
    print(f"ID: {id} - Label: {sample[0]}")

    # Get attention map
    attention_map, class_idx = model.get_attention_map(input_tensor)
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=2, slice_idx=45)

    if save_sample_attention:
        # nib.save(nib.Nifti1Image(attention_map.cpu().numpy(), np.eye(4)), f'{config['GRADCAM_OUTPUT_DIR']}/DatasetGradCAM_3Dattention_{id}.nii')
        save_gradcam_3d(attention_map, id, sample)
    
    return id, img, attn, class_idx, sample[1]

def create_gradcam_plot(save_sample_attention=False):
    
    results = [get_sample_gradcam(id, save_sample_attention=save_sample_attention) for id in ids]

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle(f'ADNI GradCAM Results {config['TRAINING_VIT_PATCH_SIZE']}patch', fontsize=16)

    # Plot each subject's results
    for idx, (ID, image, attention, class_idx, sample) in enumerate(results):
        row = idx // cols
        col = idx % cols
        ax = axes[col] if rows == 1 else axes[row, col]
        
        ax.imshow(-image+1 if config['GRADCAM_BACKGROUND_NOISE'] < 1 else image, cmap='gray')    # INVERSE BRIGHTNESS
        heatmap = ax.imshow(attention, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Subject {ID} (Class {class_idx.item()})')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        (axes[col] if rows == 1 else axes[row, col]).axis('off')

    file_name = f'ADNI_{config['TRAINING_VIT_PATCH_SIZE']}patch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'.replace('.', 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(config['GRADCAM_OUTPUT_DIR'], f"{file_name}.png"), dpi=300)
    plt.close()
    print(f"All results saved to {file_name}.png")

def save_gradcam_3d(attention_map, id, sample):
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    threshold = 0.2
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
    file_name = f'ADNI_{config['TRAINING_VIT_PATCH_SIZE']}patch_3Dattention_{id}'.replace('.', 'p')
    plt.title(f'3D GradCAM (Label: {sample[0]}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300)
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
    dataset = ADNIDataset(config, mode="val", generate_data=False)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    create_gradcam_plot(save_sample_attention=config['GRADCAM_SAVE_ATTENTION'])
