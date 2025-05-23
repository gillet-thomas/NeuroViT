import torch
import pickle
import pandas as pd
import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import time 

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset
from monai.transforms import Compose, RandSpatialCrop, ToTensor


# ADNI dataset class
class ADNIDataset(Dataset):
    def __init__(self, config, mode='train', generate_data=False):
        self.mode = mode
        self.config = config
        self.batch_size = config['TRAINING_BATCH_SIZE']
        self.csv_path = config['ADNI_CSV_PATH']
        self.dataset_path = config['ADNI_TRAIN_PATH'] if mode == 'train' else config['ADNI_VAL_PATH']

        if generate_data:
            self.generate_data()

        # self.transform = Compose([
        #     RandSpatialCrop(roi_size=(75, 75, 75), random_center=True, random_size=False),
        #     ToTensor()
        # ])
        
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        # Data filtering
        # self.data = [sample for sample in self.data if sample[5] < 68 or sample[5] > 80]
        # self.data = [sample for sample in self.data if sample[3] in ['AD', 'CN', 'LMCI', 'EMCI']]
        print(f"Dataset initialized: {len(self.data)} {mode} samples") # Train young 18340 & old 20020, val young 1960 & old 2240
        
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
        
        print(f"Training subjects: {len(train_subjects)}")  # 185 - 89 with young/old groups
        print(f"Validation subjects: {len(val_subjects)}")  # 21 - 11 with young/old groups
        
        # Split dataframe based on subjects
        train_df = df[df['Subject'].isin(train_subjects)] 
        train_df = train_df[(train_df['Age'] < q25) | (train_df['Age'] > q75)]     # Double verification because some subjects have diffent ages accross datasamples
        val_df = df[df['Subject'].isin(val_subjects)]
        val_df = val_df[(val_df['Age'] < q25) | (val_df['Age'] > q75)]            # Double verification because some subjects have diffent ages accross datasamples
        
        print(f"Training rows: {len(train_df)}")            # 630 - 302 with young/old groups
        print(f"Validation rows: {len(val_df)}")            # 60 - 32 with young/old groups

        train_samples, val_samples = [], []

        # Process training data
        print("Processing training data...")
        for row in tqdm(train_df.itertuples(index=False), total=len(train_df)):
            subject, group, sex, age, path_fmri = row.Subject, row.Group, row.Sex, row.Age, row.Path_fMRI_brain
            samples = self.process_subject_data(subject, path_fmri, group, sex, age)
            train_samples.extend(samples)
        
        # Process validation data
        print("Processing validation data...")
        for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
            subject, group, sex, age, path_fmri = row.Subject, row.Group, row.Sex, row.Age, row.Path_fMRI_brain
            samples = self.process_subject_data(subject, path_fmri, group, sex, age)
            val_samples.extend(samples)
        
        print(f"Processed {len(train_samples)} train samples")          # 630 - 42280 with young/old groups
        print(f"Processed {len(val_samples)} validation samples")       # 60 - 4480 with young/old groups
        
        # Save to pickle files  
        with open(self.config['ADNI_TRAIN_PATH'], 'wb') as f:
            pickle.dump(train_samples, f)
        with open(self.config['ADNI_VAL_PATH'], 'wb') as f:
            pickle.dump(val_samples, f)
        print("Datasets saved!")

    def generate_folds(self, base_path):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Gender', 'Age', 'Age_Group', 'Pain_Distraction_Group'])
        
        # Get unique subjects and their counts
        unique_subjects = df['Subject'].unique()
        n_subjects = len(unique_subjects)
        print(f"Total unique subjects: {n_subjects}")  # 178
        
        # Randomly shuffle subjects
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        # Implement 5-fold cross-validation
        k_folds = 5
        fold_size = n_subjects // k_folds
        
        # Create directories for each fold if they don't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Process each fold
        for fold in range(k_folds):
            print(f"\nProcessing fold {fold+1}/{k_folds}")
            
            # Calculate start and end indices for validation subjects in this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_subjects
            
            # Split subjects for this fold
            val_subjects = shuffled_subjects[val_start:val_end]
            train_subjects = np.concatenate([
                shuffled_subjects[:val_start],
                shuffled_subjects[val_end:]
            ])
            
            print(f"Training subjects: {len(train_subjects)}")
            print(f"Validation subjects: {len(val_subjects)}")
            
            # Split dataframe based on subjects
            train_df = df[df['Subject'].isin(train_subjects)]
            val_df = df[df['Subject'].isin(val_subjects)]
            
            print(f"Training rows: {len(train_df)}")
            print(f"Validation rows: {len(val_df)}")
            
            train_samples, val_samples = [], []
            
            # Process training data
            print("Processing training data...")
            for row in tqdm(train_df.itertuples(index=False), total=len(train_df)):
                subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
                samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
                train_samples.extend(samples)
            
            # Process validation data
            print("Processing validation data...")
            for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
                subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
                samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
                val_samples.extend(samples)
            
            print(f"Processed {len(train_samples)} train samples")
            print(f"Processed {len(val_samples)} validation samples")
            
            # Create fold directory
            fold_dir = os.path.join(base_path, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Save to pickle files
            train_path = os.path.join(fold_dir, 'train_data.pkl')
            val_path = os.path.join(fold_dir, 'val_data.pkl')
            
            with open(train_path, 'wb') as f:
                pickle.dump(train_samples, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_samples, f)
            
            print(f"Fold {fold+1} datasets saved!")
        
        print("\nAll folds processed and saved successfully!")
    
    def process_subject_data(self, subject, fmri_path, group, gender, age):
        """Process a single subject's fMRI data and return samples."""
        samples = []
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()    # (91, 109, 91, 140)

            for timepoint in range(fmri_data.shape[-1]):
                samples.append((subject, timepoint, fmri_path, group, gender, age))
                
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            
        return samples

    def __getitem__(self, idx):
        subject, timepoint, fmri_path, group, gender, age = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = nib.load(fmri_path)
            fmri_data = fmri_img.dataobj[1:, 10:-9, 1: , timepoint]          # Shape: (91, 109, 91, 146) -> (90, 90, 90)
            mri_tensor = (fmri_data - fmri_data.mean()) / (fmri_data.std() + 1e-8)  # Normalize, add 1e-8 to avoid division by zero
            mri_tensor = torch.tensor(mri_tensor, dtype=torch.float32)      # (90, 90, 90) shape
            
            # if self.transform:
            #     mri_tensor = mri_tensor.unsqueeze(0)
            #     # plt.imsave("mri_tensor0.png", mri_tensor.squeeze(0)[:,:, 45].numpy())
            #     mri_tensor = self.transform(mri_tensor).squeeze
            #     # plt.imsave("mri_tensor1.png", mri_tensor.squeeze(0)[:,:, 45].numpy())
            #     # time.sleep(5)

            group_encoded = torch.tensor(0 if group == 'CN' else 1 if group in ['EMCI', 'LMCI'] else 2 if group == 'AD' else -1)     # 0: CN, 1: EMCI/LMCI, 2: AD, -1: unknown
            gender_encoded = torch.tensor(0 if gender == 'F' else 1)
            age = torch.tensor(age)
            age_group = torch.tensor(0 if age < 69 else 1)      # min 56, max 96, median 74. Quartile1 = 69, Quartile3 = 78.

            # if age < 68 and age > 80:
            #     print("ERROR: age out of bounds")
            return subject, timepoint, mri_tensor, group_encoded, gender_encoded, age, age_group
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
