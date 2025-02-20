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
        self.dataset_path = config['dataset_train_path'] if mode == 'train' else config['dataset_val_path']
        self.selected_groups = ['EMCI', 'CN', 'LMCI', 'AD']
        
        # self.data = self.generate_data(config['dataset_train_path'], config['dataset_val_path'])
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)  # 78820 train samples and 8680 val sample
        
        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        
    def generate_data(self, train_path, val_path):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Group', 'Sex', 'Age'])
        df = df[df['Group'].isin(self.selected_groups)]  # Filter out unwanted groups
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")       # 178
        
        # Randomly shuffle and split subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        train_size = int(0.9 * n_subjects)
        train_subjects = shuffled_subjects[:train_size]
        val_subjects = shuffled_subjects[train_size:]
        
        print(f"Training subjects: {len(train_subjects)}")  # 160
        print(f"Validation subjects: {len(val_subjects)}")  # 18
        
        # Split dataframe based on subjects
        train_df = df[df['Subject'].isin(train_subjects)] 
        val_df = df[df['Subject'].isin(val_subjects)]
        
        print(f"Training rows: {len(train_df)}")            # 572
        print(f"Validation rows: {len(val_df)}")            # 63

        train_samples, val_samples = [], []

         # Process training data
        print("Processing training data...")
        for row in tqdm(train_df.itertuples(index=False), total=len(train_df)):
            subject, fmri_path, group, sex, age = row.Subject, row.Path_fMRI, row.Group, row.Sex, row.Age
            samples = self.process_subject_data(subject, fmri_path, group, sex, age)
            train_samples.extend(samples)
        
        # Process validation data
        print("Processing validation data...")
        for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
            subject, fmri_path, group, sex, age = row.Subject, row.Path_fMRI, row.Group, row.Sex, row.Age
            samples = self.process_subject_data(subject, fmri_path, group, sex, age)
            val_samples.extend(samples)
        
        print(f"Processed {len(train_samples)} train samples")          # 78820
        print(f"Processed {len(val_samples)} validation samples")       # 8680
        
        # Save to pickle files
        with open(train_path, 'wb') as f:
            pickle.dump(train_samples, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_samples, f)
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
