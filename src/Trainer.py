import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import datetime
import pickle

class Trainer():
    def __init__(self, config, model, dataset_train, dataset_val):
        self.config = config
        self.data = dataset_train
        self.val_data = dataset_val
        self.device = config['device']
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")          ## 114 batches of size 64
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")    ## 13 batches of size 64
        # print([name for name, param in self.model.named_parameters() if param.requires_grad])

    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./results/{timestamp}"
        os.mkdir(path)

        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)
            torch.save(self.model.state_dict(), f'{path}/model-e{epoch}.pth')
            print(f"Model saved to .{path}/model-e{epoch}.pth")
    
    def train(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        log_interval = self.config['log_interval']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        for i, (subject, timepoint, mri, group, age, sex) in enumerate(self.dataloader):
            mri, group = mri.to(self.device), group.to(self.device)  ## (batch_size, 64, 64, 48, 140) and (batch_size)

            outputs = self.model(mri)  # output is [batch_size, 4]
            loss = self.criterion(outputs, group.argmax(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == group.argmax(dim=1)).sum().item()
            total += group.size(0)  # returns the batch size

            if i != 0 and i % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i}: train loss {running_loss/log_interval}, train accuracy {correct/total}")
                wandb.log({"epoch": epoch, "batch": i, "train loss": running_loss/log_interval, "train accuracy": correct/total})
                correct, total, running_loss = 0, 0, 0.0

    def validate(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for i, (subject, timepoint, mri, group, age, sex) in enumerate(self.val_dataloader):
                mri, group = mri.to(self.device), group.to(self.device)  ## (batch_size, 64, 64, 48) and (batch_size)
                outputs = self.model(mri)
                loss = self.criterion(outputs, group.argmax(dim=1))
                val_loss += loss.item()
                correct += (outputs.argmax(dim=1) == group.argmax(dim=1)).sum().item()
                total += group.size(0)  # returns the batch size
                
            print(correct, total)
            avg_val_loss = val_loss / len(self.val_dataloader)
            accuracy = correct / total
            print(f"VALIDATION - Epoch {epoch}, Total batch {i}, avg validation loss {avg_val_loss}, val accuracy {accuracy}")
            wandb.log({"epoch": epoch, "val loss": avg_val_loss, "val accuracy": accuracy})
    
    def evaluate_samples(self):
        self.model.eval()  # Set model to evaluation mode
        print("=" * 50)
        print(f"Training set has {len(self.data)} samples and validation set has {len(self.val_data)} samples.")
        print("Training loader has", len(self.dataloader), "batches and validation loader has", len(self.val_dataloader), "batches.")

        # Count number of unique subjects in training set
        with open(self.data.dataset_train_path, 'rb') as f: train_data = pickle.load(f)  # 69720 samples
        print(len(train_data))
        unique_train_subjects = list(set([sample[0] for sample in train_data]))           # 172 unique subjects
        print(f"Unique training subjects: {unique_train_subjects}")

        with open(self.val_data.dataset_val_path, 'rb') as f: val_data = pickle.load(f)  # 17780 samples
        print(len(val_data))
        unique_val_subjects = list(set([sample[0] for sample in val_data]))               # 44 unique subjects
        print(f"Unique validation subjects: {unique_val_subjects}")

        common = list(set(unique_train_subjects) & set(unique_val_subjects))
        print(f"Common subjects: {common}")

        # Create evaluation dataset and dataloader
        evaluation_data = self.val_data
        # evaluation_data = torch.utils.data.Subset(self.val_data, range(100))
        evaluation_dataloader = torch.utils.data.DataLoader(evaluation_data, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

        accuracy, duplicates = 0, 0
        with torch.no_grad():
            for i, (subject, timepoint, mri, group, age, sex) in tqdm(enumerate(evaluation_dataloader), total=len(evaluation_dataloader)):
                subject = subject[0]
                mri = mri.to(self.device)
                predictions = self.model(mri)  # Get model predictions (batch_size, 4)

                prediction = predictions.argmax(dim=1).item()
                actual = group.argmax(dim=1).item()
                # print(f"Predictions of {i}: {self.data.selected_groups[prediction]}/{self.data.selected_groups[actual]}")

                if subject in unique_train_subjects:
                    duplicates += 1
                    # print(f"Duplicate subject found: {subject}")

                accuracy += prediction == actual

        print(f"Accuracy: {accuracy/len(evaluation_dataloader)}%")
        print(f"Duplicates: {duplicates/len(evaluation_dataloader)}%")