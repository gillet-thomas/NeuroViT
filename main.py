import sys
import yaml
import wandb
import torch
import warnings

from src.data.ADNIDataset import ADNIDataset
from src.fmriEncoder import fmriEncoder
from src.Trainer import Trainer

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    config = yaml.safe_load(open("./configs/config.yaml"))
    device = f'cuda:{config["cuda_device"]}' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    config["device"] = device
    print(f"Device: {device}")

    # Initialize wandb
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else None
    wandb_mode = 'online' if config["wandb_enabled"] else 'disabled'
    wandb.init(project="fMRI2Vec", mode=wandb_mode, config=config, name=name)

    if config['training_enabled']:
        torch.manual_seed(config["seed"])
        dataset_train = ADNIDataset(config, mode="train")
        dataset_val = ADNIDataset(config, mode="val")
        model = fmriEncoder(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()
    else:
        print("Training is disabled. Inference mode enabled.")
        dataset_train = ADNIDataset(config, mode="train")
        dataset_val = ADNIDataset(config, mode="val")
        model = fmriEncoder(config)
        model.load_state_dict(torch.load('./results/2025-02-19_08-46-01/model-e4.pth', map_location=device, weights_only=True))
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.evaluate_samples()
        