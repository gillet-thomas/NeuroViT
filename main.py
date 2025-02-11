import sys
import yaml
import wandb
import torch
import warnings

from src.fmriEncoder import fmriEncoder
from src.Trainer import Trainer
from src.data.ADNIDataset import ADNIDataset

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    config = yaml.safe_load(open("./configs/config.yaml"))
    device = f'cuda:{config["cuda_device"]}' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    config["device"] = device
    print(f"Device: {device}")
    torch.manual_seed(config["seed"])

    # Initialize wandb
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else None
    wandb_mode = 'online' if config["wandb_enabled"] else 'disabled'
    wandb.init(project="CLIP_MIMIC_CXR", mode=wandb_mode, config=config, name=name)

    if config['training_enabled']:
        dataset_train = ADNIDataset(config, mode="train", mini=True)
        dataset_val = ADNIDataset(config, mode="val", mini=True)
        
        model = fmriEncoder(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()
        