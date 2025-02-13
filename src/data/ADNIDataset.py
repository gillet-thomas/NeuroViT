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
    def __init__(self, config, mode='train', mini=False):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['csv_path']
        self.dataset_path = config['dataset_path'] if not mini else config['mini_dataset_path']
        self.selected_groups = ['EMCI', 'CN', 'LMCI', 'AD']
        
        # Load data using memory mapping
        self.data = self.get_data()
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        self.train_data, self.val_data = torch.utils.data.random_split(self.data, [0.80, 0.20])
        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"{self.dataset_path} initialized: {len(self.data)} {mode} samples/{len(self.data)} total")
    
    def get_data(self):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Group', 'Sex', 'Age'])
        df = df[df['Group'].isin(self.selected_groups)]
        print(f"Found {len(df)} entries in CSV")

        # Create list and store data samples
        samples = []
        for row in tqdm(df.itertuples(), total=len(df)):
            subject, fmri_path, group, sex, age = row.Subject, row.Path_fMRI, row.Group, row.Sex, row.Age

            # Verify fMRI is 64x64x48x140
            try:
                fmri_img = load_img(fmri_path)
                fmri_data = fmri_img.get_fdata()
                if fmri_data.shape != (64, 64, 48, 140):
                    print(f"Error: Expected height and width to be 64x64, got {fmri_data.shape[0]}x{fmri_data.shape[1]} for subject {subject}")
                    continue

                samples.append((subject, fmri_path, group, sex, age))
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
            
        print(f"Processed {len(samples)} samples successfully")
        
        # Save to pickle file
        dataset_path = './src/data/adni_metadata.pkl'
        pickle.dump(samples, open(dataset_path, 'wb'))
        print(f"Saved dataset to {dataset_path}")

        # Save first 10 samples to mini path
        mini_dataset_path = './src/data/adni_metadata_mini.pkl'
        samples = samples[:10]
        pickle.dump(samples, open(mini_dataset_path, 'wb'))
        print(f"Saved dataset to {mini_dataset_path}")
        
        return samples
    
    def __getitem__(self, idx):
        subject, fmri_path, group, sex, age = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata(dtype=np.float32)
            # fmri_data = fmri_data[:, :, :, ::2]                       # Select every 2nd timepoint
            fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)  # (64, 64, 48, 140) shape

            # One-hot encode categorical variables using PyTorch
            group_index = self.selected_groups.index(group)    
            group_encoded = F.one_hot(torch.tensor(group_index), num_classes=len(self.selected_groups))

            sex_classes = ['F', 'M']
            sex_index = sex_classes.index(sex)
            sex_encoded = F.one_hot(torch.tensor(sex_index), num_classes=len(sex_classes))

            age = torch.tensor(age)

            return subject, fmri_tensor, group_encoded, sex_encoded, age
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
