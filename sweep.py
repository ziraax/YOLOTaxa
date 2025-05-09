import wandb
from config import CONFIG
from train import train_model

# Define the sweep configuration for hyperparameter tuning
sweep_config = {
    "method": "bayes",  # Bayesian optimization method
    "metric": {
        "name": "val/accuracy_top1",  # Metric to optimize
        "goal": "maximize"  # Goal is to maximize the metric
    },
    "parameters": {
        "initial_lr": {
            "min": 1e-8,
            "max": 1e-3
        },  # Learning rate range
        "batch_size": {
            "values": [8, 16, 32, 64, 128]
        },  # Batch sizes to try
        "optimizer": {
            "values": ["adam", "sgd", "adamw"]  # Trying different optimizers
        },
        "momentum": {
            "min": 0.8,
            "max": 0.99
        },  # For SGD optimizer (if selected)
        "weight_decay": {
            "min": 0.0,
            "max": 0.1
        },  # Weight decay (L2 regularization)
        "epochs": {
            "values": [10, 20, 50, 100]  # Number of epochs for training
        },
        "early_stopping_patience": {
            "values": [10]  # Patience for early stopping (how many epochs without improvement)
        }
    }
}

def run_sweep():
    # Initialize sweep in wandb
    sweep_id = wandb.sweep(sweep=sweep_config, project=CONFIG['project_name'])

    # Run the sweep using the train_model function with a count of 50 trials
    wandb.agent(sweep_id, function=train_model, count=50)  # Adjust count as needed

if __name__ == "__main__":
    run_sweep()
