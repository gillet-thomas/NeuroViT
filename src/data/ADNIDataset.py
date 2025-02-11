import torch
import pickle
import pandas as pd
import numpy as np

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
        
        # Load data using memory mapping
        # self.data = self.get_data()
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        # Check if height and width are 64x64
        processed_data = []
        for sample in self.data:
            fmri = sample['fmri']
            if fmri.shape[0] == 64 and fmri.shape[1] == 64:
                processed_data.append(sample)
            else:
                print(f"Error: Expected height and width to be 64x64, got {fmri.shape[0]}x{fmri.shape[1]}")
        self.data = processed_data

        self.train_data, self.val_data = torch.utils.data.random_split(self.data, [0.80, 0.20])
        self.data = self.train_data if mode == 'train' else self.val_data
        print(f"{self.dataset_path} initialized: {len(self.data)} {mode} samples/{len(self.data)} total")
    
    def get_data(self):
        # Load CSV file
        df = pd.read_csv(self.csv_path)
        print(f"Found {len(df)} entries in CSV")

        # Create list and store data samples
        samples = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Load fMRI
                fmri_path = row['Path_fMRI']
                fmri_img = load_img(fmri_path)
                fmri_data = fmri_img.get_fdata()
                fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)  # (64, 64, 48, 140) shape
                
                # Create sample dictionary
                sample = {
                    'subject_id': row['Subject'],
                    'fmri': fmri_tensor,
                    'group': row['Group'],
                    'sex': row['Sex'],
                    'age': row['Age_x'],
                }
                samples.append(sample)
                
            except Exception as e:
                print(f"Error processing subject {row['Subject']}: {e}")
        
        print(f"Processed {len(samples)} samples successfully")
        
        # Save to numpy file instead of pickle for memory mapping
        dataset_path = './src/data/adni_dataset.pkl'
        pickle.dump(samples, open(dataset_path, 'wb'))
        print(f"Saved dataset to {dataset_path}")

        # Save first 10 samples to mini path
        mini_dataset_path = './src/data/adni_dataset_mini.pkl'
        samples = samples[:10]
        pickle.dump(samples, open(mini_dataset_path, 'wb'))
        print(f"Saved dataset to {mini_dataset_path}")
        
        return samples
    
    def __getitem__(self, idx):
        subject, fmri, group, sex, age = self.data[idx].values()    # Types are str, torch.Tensor, str, str, int
        age = torch.tensor(age)
        
        # One-hot encode categorical variables using PyTorch
        group_classes = ['CN', 'AD', 'EMCI', 'LMCI', 'MCI', 'SMC']
        group_index = group_classes.index(group)    
        group_encoded = F.one_hot(torch.tensor(group_index), num_classes=len(group_classes))

        sex_classes = ['F', 'M']
        sex_index = sex_classes.index(sex)
        sex_encoded = F.one_hot(torch.tensor(sex_index), num_classes=len(sex_classes))

        return subject, fmri, group_encoded, sex_encoded, age
    
    def __len__(self):
        return len(self.data)
