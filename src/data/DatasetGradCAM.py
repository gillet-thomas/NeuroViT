import os
import torch
import pickle
import numpy as np
import nibabel as nib    
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset


# GradCAM dataset class (fake data)
class GradCAMDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.base_path = config['base_path']
        self.dataset_path = config['gradcam_train_path'] if mode == 'train' else config['gradcam_val_path']

        self.grid_size = config['grid_size']
        self.patch_size = config['patch_size']
        
        self.batch_size = config['batch_size']
        self.num_samples = config['num_samples']

        if self.config['generate_dataset']:
            self.generate_data()
            
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        # self.visualize_sample_3d(1)
        # self.visualize_sample_3d(2)
        # self.visualize_sample_3d(3)
        # self.visualize_sample_3d(4)
        # self.visualize_sample_3d(5)


    def generate_data(self):
        volumes = np.zeros((self.num_samples, self.grid_size, self.grid_size, self.grid_size))
        labels = np.zeros((self.num_samples), dtype=int)
        coordinates = np.zeros((self.num_samples, 3))
        
        num_patches = self.grid_size // self.patch_size
        
        for i in range(self.num_samples):
            tx = np.random.randint(0, num_patches) * self.patch_size
            ty = np.random.randint(0, num_patches) * self.patch_size
            tz = np.random.randint(0, num_patches) * self.patch_size
            
            volumes[i] = 0.001 # Add noise for other voxels
            volumes[i, tx:tx+self.patch_size, ty:ty+self.patch_size, tz:tz+self.patch_size] = 1
            coordinates[i] = [tx, ty, tz]
            labels[i] = (tx//self.patch_size) + (ty//self.patch_size) * num_patches + (tz//self.patch_size) * num_patches * num_patches

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

    def visualize_sample_3d(self, idx, save_path="./explainability/xAi_gradcam3DViT/DatasetGradCAM/"):
        
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
        x_indices, y_indices, z_indices = np.where(volume == 1)     # Find all points where volume has value 1    
        ax.scatter(x_indices, y_indices, z_indices, c='red', marker='s', alpha=0.4, s=50) # scatter plot for all points with value 1 (the cube)
        
        # Add a bounding box for the volume
        ax.set_xlim(0, volume.shape[0])
        ax.set_ylim(0, volume.shape[1])
        ax.set_zlim(0, volume.shape[2])
        
        # Set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        # Save the figure
        plt.title(f'3D Visualization of Target Cube (Label: {label}, coordinates: {coordinates})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'DatasetGradCAM_sample_{idx}.png'), dpi=300)
        plt.close()
        
        nib.save(nib.Nifti1Image(volume, np.eye(4)), os.path.join(save_path, f'DatasetGradCAM_sample_{idx}nii'))
        
        print(f"3D visualization saved to {os.path.join(save_path, f'DatasetGradCAM_sample_{idx}')}")
    