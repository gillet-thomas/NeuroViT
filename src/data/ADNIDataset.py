import torch
import pickle
import pandas as pd
import numpy as np
import gc

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset
from torch.nn import functional as F

class ADNIDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['csv_path']
        self.dataset_train_path = config['dataset_train_path']
        self.dataset_val_path = config['dataset_val_path']
        self.selected_groups = ['EMCI', 'CN', 'LMCI', 'AD']
        
        # self.data = self.generate_data()
        # self.train_data, self.val_data = torch.utils.data.random_split(self.data, [0.80, 0.20])
        with open(self.dataset_train_path, 'rb') as f: train_data = pickle.load(f)  # 69720 samples
        with open(self.dataset_val_path, 'rb') as f: val_data = pickle.load(f)      # 17780 samples

        self.data = train_data if mode == 'train' else val_data
        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        
    def generate_data(self):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Group', 'Sex', 'Age'])
        df = df[df['Group'].isin(self.selected_groups)]  # Filter out unwanted groups
        dataset = list(df.itertuples(index=False))  

        # Split into training and validation sets
        train_samples, val_samples = [], []
        train_data, val_data = torch.utils.data.random_split(dataset, [0.90, 0.10])
        print(f"Training set has {len(train_data)} samples and validation set has {len(val_data)} samples.")

        # Process training data
        for row in tqdm(train_data, total=len(train_data)):
            subject, fmri_path, group, sex, age = row.Subject, row.Path_fMRI, row.Group, row.Sex, row.Age
            samples = self.process_subject_data(subject, fmri_path, group, sex, age)
            train_samples.extend(samples)
        
        # Process validation data
        for row in tqdm(val_data, total=len(val_data)):
            subject, fmri_path, group, sex, age = row.Subject, row.Path_fMRI, row.Group, row.Sex, row.Age
            samples = self.process_subject_data(subject, fmri_path, group, sex, age)
            val_samples.extend(samples)
        
        print(f"Processed {len(train_samples)} train samples successfully")
        print(f"Processed {len(val_samples)} test samples successfully")
        
        # Save to pickle files
        pickle.dump(train_samples, open(self.dataset_train_path, 'wb'))
        pickle.dump(val_samples, open(self.dataset_val_path, 'wb'))
        print("Datasets saved!")

    def process_subject_data(self, subject, fmri_path, group, sex, age):
        """Process a single subject's fMRI data and return samples."""
        samples = []
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()
            
            if fmri_data.shape != (64, 64, 48, 140):
                print(f"Error: Expected (64,64,48,140), got {fmri_data.shape} for subject {subject}")
                return samples
                
            for timepoint in range(fmri_data.shape[-1]):
                samples.append((subject, timepoint, fmri_path, group, sex, age))
                
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            
        return samples

    def __getitem__(self, idx):
        subject, timepoint, fmri_path, group, sex, age = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata(dtype=np.float32)
            fmri_data = fmri_data[:, :, :, timepoint]                  # Select timepoint
            mri_tensor = torch.tensor(fmri_data, dtype=torch.float32)  # (64, 64, 48) shape
            mri_tensor = (mri_tensor - mri_tensor.mean()) / mri_tensor.std() # Normalize
            # One-hot encode categorical variables using PyTorch
            group_index = self.selected_groups.index(group)    
            group_encoded = F.one_hot(torch.tensor(group_index), num_classes=len(self.selected_groups))

            sex_classes = ['F', 'M']
            sex_index = sex_classes.index(sex)
            sex_encoded = F.one_hot(torch.tensor(sex_index), num_classes=len(sex_classes))

            age = torch.tensor(age)

            return subject, timepoint, mri_tensor, group_encoded, sex_encoded, age
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
