import torch
import albumentations as A
from pathlib import Path

RAW_DATA_ROOT = Path("DATA/DATA_500_INDIV")
PROCESSED_PATH = Path("DATA/processed_dataset")
YOLO_DATASET_PATH = Path("DATA/yolo_dataset")

CONFIG = {

    # Project configuration
    "project_name": "YOLOv8Classification500",

    # Raw data structure 
    "years": ["2022", "2023"],
    "types": ["Macro", "Micro"],

    # Path configuration 
    "raw_data_root": str(RAW_DATA_ROOT),
    "processed_path": str(PROCESSED_PATH),
    "yolo_dataset_path": str(YOLO_DATASET_PATH),

    # Scale bar removal 
    "scalebar_model_path": "ScaleBarModel/best.pt",
    "scalebar_img_size": 416,
    "scalebar_confidence": 0.4,
    "convert_grayscale": True, # convert all images to grayscale since some of them are RGB
    "grayscale_mode": "RGB",

    # Dataset splitting 
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "seed": 42,

    # Augmentation parameters
    "augmentation_threshold": 1000,      # Minimum images required per class
    "augmentation_multiplier": 10,       # Maximum copies per original image
    "class_weights": "balanced",         # [This was incorrect - should be "balanced" or None]

    # Augmentation techniques
    "augmentation": {                   # Albumentations pipeline probabilities
        "HorizontalFlip": 0.5,          # 50% chance of horizontal flip
        "VerticalFlip": 0.2,            # 20% chance of vertical flip
        "Rotate90": 0.3,                # 30% chance of 90° rotation
        "BrightnessContrast": 0.4,      # 40% chance of brightness/contrast adjust
        "HueSaturation": 0.3,           # 30% chance of hue/saturation adjust
    },


    # YOLOv8 Training
    "model_name": "yolov8m-cls.pt",
    "img_size": 224, # Model will automatically resize images to this size
    "batch_size": 32,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "loss": {
        "name": "CE", # Could be FocalLoss, WeightedCrossEntropyLoss, etc. 
        "alpha": "balanced", # For WeightedCE/Focal
        "gamma": 2.0, # Only for FocalLoss
    },

    # Optimization and advanced training parameters
    "optimizer": "Adam",
    "initial_lr": 1e-4,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "patience": 10, # Patience is the number of epochs with no improvement after which training will be stopped
    "early_stopping": True, # If True, training will stop if no improvement is seen for 'patience' epochs

}

    # Augmentation pipeline
CONFIG["augmentation_pipeline"] = A.Compose([
    A.HorizontalFlip(p=CONFIG['augmentation']['HorizontalFlip']),
    A.VerticalFlip(p=CONFIG['augmentation']['VerticalFlip']),
    A.RandomRotate90(p=CONFIG['augmentation']['Rotate90']),
    A.RandomBrightnessContrast(p=CONFIG['augmentation']['BrightnessContrast']),
    A.HueSaturationValue(p=CONFIG['augmentation']['HueSaturation']),
])