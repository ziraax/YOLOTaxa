import wandb
import os

from ultralytics import YOLO
from config import CONFIG
from data_preprocessing import log_class_distribution

def train_model():
    # Initialize Weights & Biases
    wandb.init(
        project=CONFIG['project_name'],
        entity=CONFIG['entity'],
        config=CONFIG,
    )

    # Calculate and log class weights
    class_weights = log_class_distribution()
    wandb.config.update({"class_weights": class_weights})

    # Initialize YOLO model
    model = YOLO(CONFIG['model_name'])
    model.set_class_weights(class_weights)

    # Train model 
    results = model.train(
        data=os.path.join(CONFIG['dataset_path'], "dataset.yaml"),
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        device=CONFIG['device'],
        augment=True,
        optimizer="AdamW",
        lr0=CONFIG['initial_lr'],
        weight_decay=CONFIG['weight_decay'],
        label_smoothing=CONFIG['label_smoothing'],
        patience=15,
        project=CONFIG['project_name'],
        name=wandb.run.name,
    )

    # Save and log model
    model_path = f"runs/classify/{wandb.run.name}/weights/best.pt"
    wandb.save(model_path)
    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    return model_path


