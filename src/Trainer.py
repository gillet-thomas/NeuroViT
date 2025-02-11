import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
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
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model total parameters: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M and frozen {(total_params-trainable_params)/1e6:.2f}M)')
        print(f"Number of batches training: {len(self.dataloader)} of size {self.batch_size}")          ## 114 batches of size 64
        print(f"Number of batches validation: {len(self.val_dataloader)} of size {self.batch_size}")    ## 13 batches of size 64
        # print([name for name, param in self.model.named_parameters() if param.requires_grad])

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)
            self.validate(epoch)

        torch.save(self.model.state_dict(), './results/model.pth')
        print("Model saved to ./results/model.pth")

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0

        val_interval = self.config['val_interval']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

        for i, (subject, fmri, group, age, sex) in enumerate(self.dataloader):
            fmri, group = fmri.to(self.device), group.to(self.device)  ## (batch_size, 64, 64, 48, 140) and (batch_size)
            outputs = self.model(fmri)
            print("Model output shape:", outputs.shape)
            loss = self.criterion(outputs, group.argmax(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i != 0 and i % val_interval == 0:
                print(f"Epoch {epoch}, Batch {i}: train loss {running_loss/val_interval}")
                wandb.log({"epoch": epoch, "batch": i, "train loss": running_loss/val_interval})
                running_loss = 0.0


    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, (subject, fmri, group, age, sex) in enumerate(self.val_dataloader):
                fmri, group = fmri.to(self.device), group.to(self.device)  ## (batch_size, 64, 64, 48) and (batch_size)
                
                outputs = self.model(fmri)
                loss = self.criterion(outputs, group.argmax(dim=1))
                
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_dataloader)
            print(f"VALIDATION - Epoch {epoch}, Total batch {i}, avg validation loss {avg_val_loss}")
            wandb.log({"epoch": epoch, "val loss": avg_val_loss})