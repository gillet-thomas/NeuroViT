import torch
import pickle
import pandas as pd
import numpy as np
import gc
import os

from tqdm import tqdm
from nilearn.image import load_img
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms import Resize

class ADNIDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.config = config
        self.batch_size = config['batch_size']
        self.csv_path = config['csv_path']
        self.dataset_path = config['dataset_train_path'] if mode == 'train' else config['dataset_val_path']
        self.selected_groups = ['EMCI', 'CN', 'LMCI', 'AD'] # Not used on marian's dataset
        
        # self.generate_data(config['dataset_train_path'], config['dataset_val_path'])
        # self.generate_folds('./src/data/')
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)  # 78820 train samples and 8680 val sample
        
        print(f"Dataset initialized: {len(self.data)} {mode} samples")
        
    def generate_data(self, train_path, val_path):
        # Load CSV file
        df = pd.read_csv(self.csv_path, usecols=['Subject', 'Path_fMRI', 'Gender', 'Age', 'Age_Group', 'Pain_Distraction_Group'])
        
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
            subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
            samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
            train_samples.extend(samples)
        
        # Process validation data
        print("Processing validation data...")
        for row in tqdm(val_df.itertuples(index=False), total=len(val_df)):
            subject, fmri_path, gender, age, age_group, pain_group = row.Subject, row.Path_fMRI, row.Gender, row.Age, row.Age_Group, row.Pain_Distraction_Group
            samples = self.process_subject_data(subject, fmri_path, gender, age, age_group, pain_group)
            val_samples.extend(samples)
        
        print(f"Processed {len(train_samples)} train samples")          # 78820
        print(f"Processed {len(val_samples)} validation samples")       # 8680
        
        # Save to pickle files
        with open(train_path, 'wb') as f:
            pickle.dump(train_samples, f)
        with open(val_path, 'wb') as f:
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
    
    def process_subject_data(self, subject, fmri_path, gender, age, age_group, pain_group):
        """Process a single subject's fMRI data and return samples."""
        samples = []
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata()

            # if fmri_data.shape != (64, 64, 48, 140):
            #     print(f"Error: Expected (64,64,48,140), got {fmri_data.shape} for subject {subject}")
            #     return samples
                
            for timepoint in range(fmri_data.shape[-1]):
                samples.append((subject, timepoint, fmri_path, gender, age, age_group, pain_group))
                
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            
        return samples

    def __getitem__(self, idx):
        subject, timepoint, fmri_path, gender, age, age_group, pain_group = self.data[idx]    # Types are str, torch.Tensor, str, str, int
        
        try:
            fmri_img = load_img(fmri_path)
            fmri_data = fmri_img.get_fdata(dtype=np.float32)
            fmri_data = fmri_data[:, :, :, timepoint]                  # Select timepoint
            mri_tensor = torch.tensor(fmri_data, dtype=torch.float32)  # (91, 109, 91) shape
            # mri_tensor_expanded = mri_tensor.unsqueeze(0)  # Now shape becomes (1, 91, 109, 91)
            # mri_tensor = F.interpolate(mri_tensor_expanded, size=(91, 91), mode='nearest').squeeze(0)  # Remove the temporary channel dimension
            mri_tensor = mri_tensor[1:, 10:-9, 1:]  # ([90, 90, 91])
            mri_tensor = (mri_tensor - mri_tensor.mean()) / mri_tensor.std() # Normalize

            # Encode gender
            gender_encoded = torch.tensor(0 if gender == 'F' else 1)
            # gender_list = ['M', 'F']   
            # gender_index = gender_list.index(gender)
            # gender_encoded = F.one_hot(torch.tensor(gender_index), num_classes=len(gender_list))

            age = torch.tensor(age)

            age_encoded = torch.tensor(age_group - 1)  # Convert 1, 2 to 0, 1

            pain_group = torch.tensor(pain_group)

            # age_list = [1, 2]
            # age_list = torch.tensor(age_list)
            # age_index = age_list.index(age_group)
            # age_encoded = F.one_hot(torch.tensor(age_index), num_classes=len(age_list))

            return subject, timepoint, mri_tensor, gender_encoded, age, age_encoded, pain_group
        
        except Exception as e:
            print(f"Error loading fMRI for subject {subject}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
