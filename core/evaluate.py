import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import DEVICE, CLASS_NAMES, CLASS_ABBREV, NUM_CLASSES, PLOTS_DIR, METRICS_DIR

def evaluate_model(model, loader, model_name="ImprovedResNet50", dataset_split="test"):
    """
    Evaluates the model, calculates classification metrics, and generates visualizations.
    Results are saved dynamically to be consumed by the Streamlit dashboard.
    """
    print(f"\n📊 Evaluating {model_name} on {dataset_split} split...")
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # Calculate classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES, 
        labels=np.arange(NUM_CLASSES),
        output_dict=True,
        zero_division=0
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    
    # Extract Key Metrics
    accuracy = report["accuracy"]
    macro_recall = report["macro avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    
    print(f"Accuracy: {accuracy:.4f} | Macro Recall: {macro_recall:.4f} | Weighted F1: {weighted_f1:.4f}")
    
    # --- Generate Visualizations ---
    _plot_confusion_matrix(cm, f"{model_name}_{dataset_split}_cm")
    _plot_per_class_metrics(report, f"{model_name}_{dataset_split}_recall", metric="recall")
    _plot_per_class_metrics(report, f"{model_name}_{dataset_split}_f1", metric="f1-score")
    
    # --- Save JSON for Dashboard ---
    metrics_data = {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "report": report,
        "cm": cm.tolist()
    }
    
    json_path = os.path.join(METRICS_DIR, f"{model_name}_{dataset_split}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"✅ Metrics and plots saved for {model_name}.")
    
    return metrics_data

def _plot_confusion_matrix(cm, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=False, cmap="Blues", 
        xticklabels=CLASS_ABBREV, 
        yticklabels=CLASS_ABBREV,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save with dark theme aesthetic matching Streamlit if possible or standard
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def _plot_per_class_metrics(report, filename, metric="recall"):
    metric_values = [report[cls][metric] for cls in CLASS_NAMES]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASS_NAMES, metric_values, color='#7c3aed')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9
        )
        
    plt.axhline(y=report["macro avg"][metric], color='#00d4ff', linestyle='--', label=f'Macro Avg ({report["macro avg"][metric]:.2f})')
    
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Per-Class {metric.capitalize()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
