import wandb
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG
import pandas as pd
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

def evaluate_model(model_path, data_dir):
    # Initialize W&B
    wandb.init(project=CONFIG['project_name'], job_type="evaluation")
    
    try:
        data_dir = Path(data_dir).resolve()
        yaml_path = data_dir / "dataset.yaml"
        
        with open(yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        
        class_names = data_cfg['names']
        test_dir = data_dir / data_cfg['test']
        
        # Map: class name -> class index
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Load model
        model = YOLO(model_path)
        
        y_true, y_pred = [], []
        
        # Loop through val folder
        for class_name in os.listdir(test_dir):
            class_folder = test_dir / class_name
            if not class_folder.is_dir():
                continue
            label_idx = class_to_idx[class_name]
            for img_file in tqdm(list(class_folder.glob("*"))):
                if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    continue
                try:
                    pred = model.predict(str(img_file), verbose=False)[0]
                    pred_label = pred.probs.top1
                    y_true.append(label_idx)
                    y_pred.append(pred_label)
                except Exception as e:
                    print(f"Error on image {img_file}: {e}")
        
        # Classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().reset_index()
        report_df.rename(columns={"index": "class"}, inplace=True)
        wandb.log({"classification_report": wandb.Table(dataframe=report_df)})

        # Bar plot of F1 score per class
        # Bar plot of F1 score per class (sorted)
        f1_sorted_df = report_df[report_df['class'].isin(class_names)].sort_values("f1-score")

        plt.figure(figsize=(12, max(6, len(f1_sorted_df) * 0.3)))
        sns.barplot(data=f1_sorted_df, x='f1-score', y='class', hue='class', dodge=False, legend=False, palette='viridis')
        plt.title("F1 Score per Class (Sorted)")
        plt.xlabel("F1 Score")
        plt.ylabel("Class")
        plt.xlim(0, 1)
        plt.tight_layout()
        wandb.log({"f1_score_per_class": wandb.Image(plt)})
        
        # --- Plot Precision ---
        precision_sorted = report_df.sort_values("precision")
        plt.figure(figsize=(12, max(6, len(precision_sorted) * 0.3)))
        sns.barplot(data=precision_sorted, x='precision', y='class', hue='class', dodge=False, legend=False, palette='mako')
        plt.title("Precision per Class (Sorted)")
        plt.xlabel("Precision")
        plt.ylabel("Class")
        plt.xlim(0, 1)
        plt.tight_layout()
        wandb.log({"precision_per_class": wandb.Image(plt)})

        # --- Plot Recall ---
        recall_sorted = report_df.sort_values("recall")
        plt.figure(figsize=(12, max(6, len(recall_sorted) * 0.3)))
        sns.barplot(data=recall_sorted, x='recall', y='class', hue='class', dodge=False, legend=False, palette='crest')
        plt.title("Recall per Class (Sorted)")
        plt.xlabel("Recall")
        plt.ylabel("Class")
        plt.xlim(0, 1)
        plt.tight_layout()
        wandb.log({"recall_per_class": wandb.Image(plt)})


        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 15))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
    
    except Exception as e:
        wandb.alert(title="Evaluation Failed", text=str(e))
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                      help="Path to model weights or W&B artifact address")
    parser.add_argument("--data", type=str, default=CONFIG['yolo_dataset_path'],
                      help="Path to dataset directory")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data)
