import os
import random
import shutil
import time
import wandb

from pathlib import Path
from sklearn.model_selection import train_test_split
from config import CONFIG

def organize_yolo_structure():
    """Convert processed data to YOLO format with proper splits"""
    start_time = time.time()
    print("[INFO] Organizing YOLO structure...")

    processed_path = Path(CONFIG['processed_path'])
    yolo_path = Path(CONFIG['yolo_dataset_path'])
    
    # Create YOLO directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (yolo_path / split).mkdir(parents=True, exist_ok=True)

    # Collect all classes and images
    class_images = {}
    for class_dir in processed_path.glob('*'):
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob('*.*'))
            random.Random(CONFIG['seed']).shuffle(images)
            class_images[class_name] = images

    # Split dataset while preserving class distribution
    all_images = []
    for class_name, images in class_images.items():
        all_images.extend([(str(img), class_name) for img in images])

    # First split: train vs temp (val+test)
    train, temp = train_test_split(
        all_images,
        test_size=1 - CONFIG['train_ratio'],
        stratify=[x[1] for x in all_images],
        random_state=CONFIG['seed']
    )
    
    # Second split: val vs test
    val, test = train_test_split(
        temp,
        test_size=CONFIG['test_ratio']/(CONFIG['val_ratio']+CONFIG['test_ratio']),
        stratify=[x[1] for x in temp],
        random_state=CONFIG['seed']
    )

    # Organize files into YOLO structure
    for split, data in zip(splits, [train, val, test]):
        for img_path, class_name in data:
            dest_dir = yolo_path / split / class_name
            dest_dir.mkdir(exist_ok=True)
            
            # Preserve original filename with split prefix
            dest_path = dest_dir / f"{split}_{Path(img_path).name}"
            if not dest_path.exists():
                if os.name == 'nt':  # Windows
                    shutil.copy(img_path, dest_path)
                else:
                    os.symlink(img_path, dest_path)

    elapsed = time.time() - start_time
    print(f"[INFO] YOLO structure organized in {elapsed:.2f} seconds.")
    wandb.log({"time_organize_yolo_structure_sec": elapsed})
    