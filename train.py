import wandb
import os

from ultralytics import YOLO


from config import CONFIG
from data_preprocessing import log_class_distribution

def train_model():

    ### TO BE USED LATER TO IMPLEMENT A CUSTOM TRAINING LOOP WITH WEIGHTED LOSS
    # Calculate and log class weights
    # class_weights = log_class_distribution()
    # wandb.config.update({"class_weights": class_weights})
    # model.set_class_weights(class_weights) # this line is hypothetical 


    # Initialize YOLO model
    model = YOLO(CONFIG['model_name'])

    # Use sweep parameters

    # Train model 
    results = model.train(
        data="/home/huwalter/WorkingFolderHugoWALTER/YOLOTaxa/DATA/yolo_dataset/",
        epochs=wandb.config.epochs,
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        device=CONFIG['device'],
        augment=False,
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['initial_lr'],
        weight_decay=CONFIG['weight_decay'],
        label_smoothing=CONFIG['label_smoothing'],
        patience=CONFIG['patience'],
        save_period=1, # Save model every epoch
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


