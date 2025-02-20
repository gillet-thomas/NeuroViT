import sys
import yaml
import wandb
import torch
import argparse
import warnings
import numpy as np

from src.data.ADNIDataset import ADNIDataset
from src.fmriEncoder import fmriEncoder
from src.Trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate fMRI Model")
    parser.add_argument("name", type=str, nargs="?", default=None, help="WandB run name (optional)")        # Optional
    parser.add_argument('--inference', action='store_true', help='Run in inference mode')                   # Training mode is default
    parser.add_argument('--cuda', type=int, default=2, help='CUDA device to use (e.g., 0 for GPU 0)')       # Cuda 2 is default
    parser.add_argument('--wandb', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable Weights and Biases (WandB) tracking (default: True)')

    args = parser.parse_args()
    return args

def get_device(cuda_device):
    return f'cuda:{cuda_device}' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'

def get_config(args):
    config = yaml.safe_load(open("./configs/config.yaml"))
    config["device"] = get_device(args.cuda)
    config["wandb_enabled"] = args.wandb
    return config

def initialize_wandb(config, run_name=None):
    wandb_mode = 'online' if config["wandb_enabled"] else 'disabled'
    wandb.init(project="fMRI2Vec", mode=wandb_mode, config=config, name=run_name)

def set_seeds(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

def get_datasets(config):
    dataset_train = ADNIDataset(config, mode="train")
    dataset_val = ADNIDataset(config, mode="val")
    return dataset_train, dataset_val

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Parse arguments and load config
    args = parse_args()
    config = get_config(args)
    print(f"Running on device: {config['device']}")

    if args.inference is False:
        print("Training mode enabled.")
        initialize_wandb(config, args.name)
        set_seeds(config)
        dataset_train, dataset_val = get_datasets(config)
        model = fmriEncoder(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()
    else:
        print("Training is disabled. Inference only.")
        dataset_train, dataset_val = get_datasets(config)
        model = fmriEncoder(config)
        model.load_state_dict(torch.load('./results/2025-02-19_16-18-01/model-e9.pth', map_location=config["device"], weights_only=True))
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.evaluate_samples()  
        
if __name__ == "__main__":
    main()
