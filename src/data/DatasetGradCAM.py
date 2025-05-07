import torch
import pickle
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import torch.nn.functional as F
import nibabel as nib    
import random

# GradCAM dataset class
class GradCAMDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.base_path = config['base_path']
        self.dataset_path = config['gradcam_train_path'] if mode == 'train' else config['gradcam_val_path']

        self.grid_size = 40
        self.patch_size = 10
        self.num_samples = 4000

        self.generate_data()
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        # self.visualize_sample_3d(1)
        # self.visualize_sample_3d(2)
        # self.visualize_sample_3d(3)
        # self.visualize_sample_3d(4)
        # self.visualize_sample_3d(10)


    def generate_data(self):
        volumes = np.zeros((self.num_samples, self.grid_size, self.grid_size, self.grid_size))
        labels = np.zeros((self.num_samples), dtype=int)
        coordinates = np.zeros((self.num_samples, 3))
        
        num_patches = self.grid_size // self.patch_size
        
        for i in range(self.num_samples):
            tx = np.random.randint(0, num_patches) * self.patch_size
            ty = np.random.randint(0, num_patches) * self.patch_size
            tz = np.random.randint(0, num_patches) * self.patch_size
            # volumes[i] = 2
            volumes[i, tx:tx+self.patch_size, ty:ty+self.patch_size, tz:tz+self.patch_size] = 1

            # label = np.random.randint(0, num_patches) * self.patch_size
            # volumes[i, label:label+self.patch_size, 0:self.grid_size, 0:self.grid_size] = 1
            # coordinates[i] = [label, 0, 0]
            
            coordinates[i] = [tx, ty, tz]
            # labels[i] = 0 if tx < self.grid_size // 2 else 1
            # labels[i] = tx // self.patch_size
            labels[i] = (tx//self.patch_size) + (ty//self.patch_size) * num_patches + (tz//self.patch_size) * num_patches * num_patches
            # print(f"Sample {i}: tx={tx}, ty={ty}, tz={tz}, label={labels[i]}")

        train_size = int(0.8 * self.num_samples)
        train_samples = [(v, l, c) for v, l, c in zip(volumes[:train_size], labels[:train_size], coordinates[:train_size])]
        val_samples = [(v, l, c) for v, l, c in zip(volumes[train_size:], labels[train_size:], coordinates[train_size:])]
        
        print(f"Training volumes: {len(train_samples)}")
        print(f"Validation volumes: {len(val_samples)}")
        
        # Save to pickle files
        with open(self.config['gradcam_train_path'], 'wb') as f:
            pickle.dump(train_samples, f)
        with open(self.config['gradcam_val_path'], 'wb') as f:
            pickle.dump(val_samples, f)
        print("Datasets saved!")

    def __getitem__(self, idx):
        volume, label, coordinates = self.data[idx]
        # label_encoded = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return torch.tensor(volume, dtype=torch.float32), torch.tensor(label, dtype=torch.int64), torch.tensor(coordinates)

    def __len__(self):
        return len(self.data)

    def visualize_sample_2d(self, idx, save_path="./"):
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Get the data
        volume, label, _ = self.data[idx]
        print("Item classification:", label)
        
        # If data is torch tensor, convert to numpy
        if torch.is_tensor(volume):
            volume = volume.numpy()
        
        # Remove batch dimension if present
        if volume.ndim == 5:
            volume = volume[0]
        
        # Remove channel dimension if present
        if volume.ndim == 4:
            volume = volume[0]
        
        # Find center of target if it exists
        if label == 1:
            center_x = int(np.mean(np.where(volume > 0)[0]))
            center_y = int(np.mean(np.where(volume > 0)[1]))
            center_z = int(np.mean(np.where(volume > 0)[2]))
        else:
            # Use middle of volume if no target
            center_x = volume.shape[0] // 2
            center_y = volume.shape[1] // 2
            center_z = volume.shape[2] // 2
        
        # Create a figure with 3 slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # X slice
        axes[0].imshow(volume[center_x, :, :], cmap='gray')
        if label == 1:
            mask_x = volume[center_x, :, :]
            # Create a red mask for better visibility
            red_mask = np.zeros((*mask_x.shape, 4))  # RGBA
            red_mask[mask_x > 0] = [1, 0, 0, 0.7]  # Red with 70% opacity
            axes[0].imshow(red_mask)
        axes[0].set_title(f'X-Slice (index: {center_x})')
        axes[0].axis('off')
        
        # Y slice
        axes[1].imshow(volume[:, center_y, :], cmap='gray')
        if label == 1:
            mask_y = volume[:, center_y, :]
            # Create a red mask for better visibility
            red_mask = np.zeros((*mask_y.shape, 4))  # RGBA
            red_mask[mask_y > 0] = [1, 0, 0, 0.7]  # Red with 70% opacity
            axes[1].imshow(red_mask)
        axes[1].set_title(f'Y-Slice (index: {center_y})')
        axes[1].axis('off')
        
        # Z slice
        axes[2].imshow(volume[:, :, center_z], cmap='gray')
        if label == 1:
            mask_z = volume[:, :, center_z]
            # Create a red mask for better visibility
            red_mask = np.zeros((*mask_z.shape, 4))  # RGBA
            red_mask[mask_z > 0] = [1, 0, 0, 0.7]  # Red with 70% opacity
            axes[2].imshow(red_mask)
        axes[2].set_title(f'Z-Slice (index: {center_z})')
        axes[2].axis('off')
        
        plt.suptitle(f'Volume Visualization (Label: {label})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'sample_{idx}_visualization.png'), dpi=300)
        plt.close()
        
        print(f"Visualization saved to {os.path.join(save_path, f'sample_{idx}_visualization.gradcam')}")
    
        return volume, label

    def visualize_sample_3d(self, idx, save_path="./"):
        import os
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Get the data
        volume, label, coordinates = self.data[idx]
        
        # If data is torch tensor, convert to numpy
        if torch.is_tensor(volume):
            volume = volume.numpy()
        
        # Remove batch dimension if present
        if volume.ndim == 5:
            volume = volume[0]
        
        # Remove channel dimension if present
        if volume.ndim == 4:
            volume = volume[0]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

         # Find all points where volume has value 1
        x_indices, y_indices, z_indices = np.where(volume == 1)
    
        # Get coordinates of target mask
        x, y, z = coordinates
            
        # Create scatter plot for all points with value 1 (the cube)
        ax.scatter(x_indices, y_indices, z_indices, c='red', marker='s', alpha=0.4, s=50)
        
        # Get the patch coordinates from the original code
        tx, ty, tz = coordinates
        
        # Add a bounding box for the volume
        ax.set_xlim(0, volume.shape[0])
        ax.set_ylim(0, volume.shape[1])
        ax.set_zlim(0, volume.shape[2])
        
        # Set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        # Calculate patch size (assume cube shape)
        patch_size = 0
        if len(x_indices) > 0:
            # Find the maximum spans in each dimension
            x_range = max(x_indices) - min(x_indices) + 1
            y_range = max(y_indices) - min(y_indices) + 1
            z_range = max(z_indices) - min(z_indices) + 1
            patch_size = max(x_range, y_range, z_range)
        plt.title(f'3D Visualization of Target Cube (Label: {label})\n'
            f'Origin at: {coordinates}, Size: {patch_size}x{patch_size}x{patch_size}')
    
        # Save the figure
        plt.title(f'3D Visualization of Target Cube (Label: {label}, coordinates: {coordinates})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'sample2_{idx}_3d_visualization.png'), dpi=300)
        plt.close()
        
        nib.save(nib.Nifti1Image(volume, np.eye(4)), os.path.join(save_path, f'sample2_{idx}_gradcam.nii'))

        print(f"3D visualization saved to {os.path.join(save_path, f'sample2_{idx}_3d_visualization.png')}")
        
        return volume, label
    