import os
import time
import torch
import numpy as np
import albumentations as A
import wandb
import math
import hashlib
import random
import concurrent.futures

from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.utils import compute_class_weight
from scalebar_removal import remove_scale_bars
from data_organizer import organize_yolo_structure
from utils.check_imgs import check_images
from config import CONFIG
from tqdm import tqdm


def create_dataset_yaml():
    """
    Creates a dataset YAML file for YOLO training with paths and class names.
    """
    data_yaml = f"""
    path: {CONFIG['yolo_dataset_path']}
    train: train
    val: val
    test: test
    names: {get_classes()}
    """
    with open(os.path.join(CONFIG['yolo_dataset_path'], "dataset.yaml"), "w") as f:
        f.write(data_yaml)


def get_classes():
    """
    Returns a sorted list of class names from the training directory.
    """
    train_path = os.path.join(CONFIG['yolo_dataset_path'], 'train')
    return sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])


def undersample_overrepresented_classes():
    """
    Undersamples classes that exceed the augmentation threshold (e.g. 1000 images).
    Keeps a random subset of images up to the threshold.
    This must be called BEFORE organize_yolo_structure().
    """
    input_path = CONFIG['processed_path']
    max_per_class = CONFIG['augmentation_threshold']

    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(image_files) > max_per_class:
            print(f"[INFO] Undersampling {class_name}: {len(image_files)} -> {max_per_class}")
            keep = set(random.sample(image_files, max_per_class))
            for img_file in image_files:
                if img_file not in keep:
                    os.remove(os.path.join(class_dir, img_file))




def analyze_class_distribution():
    """
    Analyzes how many images exist per class across train/val/test.
    Returns a dictionary mapping class names to image counts.
    """

    class_counts = defaultdict(int)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(CONFIG['yolo_dataset_path'], split)
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_dir] += count
    return class_counts


def augment_image_task(args):
    """
    Worker task: Applies multiple augmentations to one image and saves results.
    """
    img_path, class_dir, idx, per_image, img_name = args
    try:
        image = np.array(Image.open(img_path).convert("RGB"))
    except Exception as e:
        print(f"Error loading {img_path}: {str(e)}")
        return 0

    successful = 0
    for copy_num in range(per_image):
        seed = int(hashlib.sha256(img_name.encode()).hexdigest()[:8], 16) + copy_num
        random.seed(seed)
        torch.manual_seed(seed)

        augmented = CONFIG['augmentation_pipeline'](image=image)['image']
        aug_img = Image.fromarray(augmented)

        aug_path = os.path.join(class_dir, f"aug_{idx}_{copy_num}_{img_name}")
        aug_img.save(aug_path)
        successful += 1

    return successful


def apply_class_aware_augmentation(before_aug):
    """
    Augments underrepresented classes to reach the augmentation threshold.
    Runs image-level parallel augmentation using multiprocessing.
    Logs total number of augmented images to Weights & Biases.
    """
    class_counts = analyze_class_distribution()
    train_path = os.path.join(CONFIG['yolo_dataset_path'], 'train')

    tasks = []

    for class_name, count in class_counts.items():
        if count < CONFIG['augmentation_threshold']:
            class_dir = os.path.join(train_path, class_name)
            original_images = sorted([
                f for f in os.listdir(class_dir)
                if not f.startswith('aug_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            needed = CONFIG['augmentation_threshold'] - count
            num_originals = len(original_images)

            if num_originals == 0:
                print(f"Skipping {class_name} - no original images")
                continue

            per_image = max(1, math.ceil(needed / num_originals))
            if per_image > CONFIG['augmentation_multiplier']:
                print(f"[WARN] Class '{class_name}' cannot reach the threshold of {CONFIG['augmentation_threshold']} images "
                      f"with only {num_originals} originals and augmentation_multiplier={CONFIG['augmentation_multiplier']}. "
                      f"Will generate {CONFIG['augmentation_multiplier']} augmentations per original.")
                per_image = CONFIG['augmentation_multiplier']

            print(f"Class {class_name}: {count} -> {CONFIG['augmentation_threshold']}")
            print(f"Generating {per_image} augmentations per image for {num_originals} originals")

            for idx, img_name in enumerate(original_images):
                if needed <= 0:
                    break
                img_path = os.path.join(class_dir, img_name)
                tasks.append((img_path, class_dir, idx, per_image, img_name))
                needed -= per_image

    # Parallel processing
    total_augmented = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(augment_image_task, tasks), total=len(tasks), desc="Augmenting images"):
            total_augmented += result

    print(f"\n Augmentation complete. Total augmented images: {total_augmented}")
    wandb.log({"total_augmented_images": total_augmented})
    create_dataset_yaml()

    # Log class distribution before/after
    after_aug = analyze_class_distribution()
    comparison_data = []
    for cls in sorted(set(before_aug) | set(after_aug)):
        comparison_data.append([
            cls,
            before_aug.get(cls, 0),
            after_aug.get(cls, 0)
        ])
    comp_table = wandb.Table(columns=["Class", "Before Aug", "After Aug"], data=comparison_data)
    wandb.log({"class_distribution_comparison": comp_table})





def compute_class_weights(strategy="balanced"):
    """
    Computes class weights using sklearn's compute_class_weight.
    
    Args:
        strategy (str): "balanced" or "uniform". Only "balanced" computes true weights.
    
    Saves:
        CONFIG['loss_weights' ] - a list of float weights matching the class order. 
    """
    class_counts = analyze_class_distribution()
    classes = sorted(class_counts.keys())  # keep ordering consistent with dataset.yaml

    if strategy == "uniform":
        weights = np.ones(len(classes))
    else:
        y = []
        for cls_idx, cls in enumerate(classes):
            y.extend([cls_idx] * class_counts[cls])
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=np.array(y))

    CONFIG['loss_weights'] = weights.tolist()
    print(f"[INFO] Computed class weights: {CONFIG['loss_weights']}")

    # Loging class weights to W&B
    weight_table = wandb.Table(columns=["Class", "Weight"],
                               data=[[cls, float(w)] for cls, w in zip(classes, weights)])
    wandb.log({"class_weights_table": weight_table})


def log_class_distribution():
    """
    Logs class distribution to Weights & Biases and computes class weights.
    """
    class_counts = analyze_class_distribution()

    table = wandb.Table(
        columns=["Class", "Count"],
        data=[[k, v] for k, v in class_counts.items()]
    )

    wandb.log({
        "class_distribution": wandb.plot.bar(table, "Class", "Count", title="Class Distribution"),
        "class_histogram": wandb.plot.histogram(table, "Count", title="Class Distribution Histogram")
    })



def full_preprocessing():
    """
    Full preprocessing pipeline:
    1. Remove scale bars from images.
    2. Undersample overrepresented classes.
    3. Organize dataset into YOLO structure.
    4. Save class distribution before augmenting.
    5. Apply class-aware augmentation.
    6. Log sample images to Weights & Biases.
    7. Compute class weights.
    8. Log class distributions to Weights & Biases.
    """


    print(f"[INFO] Starting full preprocessing pipeline...")
    start = time.time()

    # Step 1: Remove scale bars
    # remove_scale_bars()
    print(f"Scale bar removal complete. Processed images saved to: {CONFIG['processed_path']}")

    # Step 2: Undersample overrepresented classes
    undersample_overrepresented_classes()   

    # Check if images are broken 
    check_images(CONFIG['processed_path'], delete=False)

    # Step 3: Organize YOLO structure
    organize_yolo_structure()

    print(f"YOLO structure organized. Processed images saved to: {CONFIG['yolo_dataset_path']}")

    # Check images again
    check_images(CONFIG['yolo_dataset_path'], delete=False)


    # Step 4: Save class dist before augmenting
    before_aug = analyze_class_distribution()

    # Step 5: Augment data
    apply_class_aware_augmentation(before_aug)


    # Check images again
    print(f"Augmentation complete. Processed images saved to: {CONFIG['yolo_dataset_path']}")
    check_images(CONFIG['yolo_dataset_path'], delete=False)

    # Step 6: Log sample images
    sample_images = []
    for cls in get_classes():
        class_dir = os.path.join(CONFIG['yolo_dataset_path'], 'train', cls)
        img_path = next(Path(class_dir).glob('*'))
        sample_images.append(wandb.Image(str(img_path), caption=cls))
    wandb.log({"dataset_preview": sample_images})

    compute_class_weights(strategy="balanced")

    # Step 6: Log class distributions
    log_class_distribution()

    end = time.time()
    duration = end - start
    print(f"[INFO] Full preprocessing completed in {duration:.2f} seconds.")
    wandb.log({"full_preprocessing_time_sec": duration})
