import wandb 
from config import CONFIG
from train import train_model

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val/accuracy_top1",
        "goal": "maximize"
    },
    "parameters": {
        # add optimizer 
        # add learning rate scheduler
        # add other hyperparameters
        "initial_lr": {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [8, 16, 32, 64, 128]},
        "augmentation_multiplier": {"values": [2, 5, 10, 25, 50]},
        "hsv_h": {"min": 0.0, "max": 0.5},
        "rotate": {"min": 0, "max": 45}
    }
}

def run_sweep():
    sweep_id= wandb.sweep(sweep=sweep_config, project=CONFIG['project_name'])
    wandb.agent(sweep_id, function=train_model, count=10)  # Adjust count as needed


