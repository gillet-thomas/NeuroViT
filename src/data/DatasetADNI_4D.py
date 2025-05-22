import torch
import pickle
import pandas as pd
import numpy as np
import os
import nibabel as nib

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset

# ADNI dataset class
class ADNIDataset4D(Dataset):
    def __init__(self, config, mode='train', generate_data=False):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['adni_csv']
        self.dataset_path = config['adni4D_train_path'] if mode == 'train' else config['adni4D_val_path']

        if generate_data:
            self.generate_data()
        
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        # Data filtering
        # self.data = [sample for sample in self.data if sample[3] < 68 or sample[3] > 80]
        # self.data = [sample for sample in self.data if sample[1] in ['AD', 'CN']]
        # self.data = self.data[:int(len(self.data) * 0.1)]
        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self):
        # Load CSV data
        df = pd.read_csv(self.csv_path, usecols=['ID', 'Subject', 'Group', 'Sex', 'Age', 'Path_sMRI_brain', 'Path_fMRI_brain'])
        print(f"Total rows in CSV: {len(df)}")              # 690
        print(f"Total unique subjects: {len(df['Subject'].unique())}")       # 206

        q25 = df['Age'].quantile(0.25)  # 69
        q75 = df['Age'].quantile(0.75)  # 78
        
        young_subjects = df[df['Age'] < q25]['Subject'].unique()
        old_subjects = df[df['Age'] > q75]['Subject'].unique()
        
        # Randomly shuffle and split subjects
        young_train = int(0.9 * len(young_subjects))
        old_train = int(0.9 * len(old_subjects))
        
        young_subjects = np.random.permutation(young_subjects)
        old_subjects = np.random.permutation(old_subjects)

        train_subjects = np.concatenate([young_subjects[:young_train], old_subjects[:old_train]]) # 43 young + 46 old = 89
        val_subjects = np.concatenate([young_subjects[young_train:], old_subjects[old_train:]]) # 5 young + 6 old = 11
        
        print(f"Training subjects: {len(train_subjects)}")  # 89 with young/old groups
        print(f"Validation subjects: {len(val_subjects)}")  # 11 with young/old groups
        
        # Split dataframe based on subjects
        train_df = df[df['Subject'].isin(train_subjects)] 
        train_df = train_df[(train_df['Age'] < q25) | (train_df['Age'] > q75)]     # Double verification because some subjects have diffent ages accross datasamples
        val_df = df[df['Subject'].isin(val_subjects)]
        val_df = val_df[(val_df['Age'] < q25) | (val_df['Age'] > q75)]            # Double verification because some subjects have diffent ages accross datasamples
        print(f"Training samples: {len(train_df)}")            # 274
        print(f"Validation samples: {len(val_df)}")            # 30

        train_list = train_df.values.tolist()
        val_list = val_df.values.tolist()

        # Save to pickle files
        with open(self.config['adni4D_train_path'], 'wb') as f:
            pickle.dump(train_list, f)
        with open(self.config['adni4D_val_path'], 'wb') as f:
            pickle.dump(val_list, f)
        print("Datasets saved!")

    def __getitem__(self, idx):
        id, subject, group, gender, age, sMRI_path, fMRI_path = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = nib.load(fMRI_path)
            fmri_data = fmri_img.dataobj[1:, 10:-9, 1: ,]          # Shape: (91, 109, 91, 140) for (H, W, D, T) -> (90, 90, 90, 140)
            fMRI_tensor = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
            fMRI_tensor = torch.tensor(fMRI_tensor, dtype=torch.float32)      # (90, 90, 90) shape

            # group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['EMCI', 'LMCI'] else 2 if group == 'AD' else -1)     # 0: CN, 1: EMCI/LMCI, 2: AD, -1: unknown
            group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['AD'] else -1)     # 0: CN, 1: AD, -1: unknown
            gender_encoded = torch.tensor(0 if gender == 'F' else 1)
            age = torch.tensor(age)
            age_group = torch.tensor(0 if age < 69 else 1)      # min 56, max 96, median 74. Quartile1 = 69, Quartile3 = 78.

            return subject, fMRI_tensor, group_encoded, gender_encoded, age, age_group
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)