import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from config import CONFIG
from ultralytics import YOLO
from sklearn.metrics import classification_report

def evaluate_model(model_path):
    wandb.init(project=CONFIG['project_name'], job_type="evaluation")
    
    model = YOLO(model_path)
    results = model.val()
    
    # Confusion Matrix
    cm = results.confusion_matrix
    plt.figure(figsize=(15,12))
    sns.heatmap(cm, annot=True, fmt=".2f")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    
    # Classification Report
    report = classification_report(results.labels, results.preds.argmax(axis=1))
    wandb.log({"classification_report": wandb.Html(report)})
    
    # PR Curve
    wandb.log({"pr_curve": wandb.plot.pr_curve(results.labels, results.probs)})
    
    wandb.finish()