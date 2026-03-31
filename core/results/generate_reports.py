import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
import torch.nn as nn
from phase2.train_utils import get_dataloaders

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "skin-ds")

MODEL_PATHS = {
    "SimpleCNN": os.path.join(BASE_DIR, "phase2", "saved_models", "baseline_simplecnn_best.pth"),
    "ResNet50": os.path.join(BASE_DIR, "phase2", "saved_models", "baseline_resnet50_best.pth")
}

SAVE_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model Definitions
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_resnet(num_classes):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    return report, cm


# -----------------------------
# Plot Functions
# -----------------------------
def plot_confusion(cm, class_names, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


def plot_metric_comparison(metrics_dict, metric_name, filename):
    names = list(metrics_dict.keys())
    values = [metrics_dict[m][metric_name] for m in names]

    plt.figure()
    plt.bar(names, values)
    plt.title(f"{metric_name} Comparison")
    plt.ylabel(metric_name)
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


def plot_per_class(metric_data, class_names, title, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, metric_data)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


def save_table(metrics_dict, filename):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for model_name, m in metrics_dict.items():
        table_data.append([
            model_name,
            round(m["accuracy"], 3),
            round(m["macro_recall"], 3),
            round(m["weighted_f1"], 3)
        ])

    columns = ["Model", "Accuracy", "Macro Recall", "Weighted F1"]

    table = ax.table(cellText=table_data, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("🚀 Generating Reports...")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR, batch_size=32)

    results = {}

    for model_name, path in MODEL_PATHS.items():
        print(f"\n🔍 Loading {model_name}")

        if model_name == "SimpleCNN":
            model = SimpleCNN(len(class_names))
        else:
            model = load_resnet(len(class_names))

        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Determine if checkpoint is state_dict or a dictionary containing it
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k.replace("model.", "", 1)] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(DEVICE)

        report, cm = evaluate(model, test_loader, class_names)

        acc = report["accuracy"]
        macro_recall = report["macro avg"]["recall"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        results[model_name] = {
            "accuracy": acc,
            "macro_recall": macro_recall,
            "weighted_f1": weighted_f1,
            "report": report,
            "cm": cm
        }

        # Confusion matrix
        plot_confusion(cm, class_names,
                       f"{model_name} Confusion Matrix",
                       f"{model_name}_confusion.png")

        # Per-class plots
        recalls = [report[c]["recall"] for c in class_names]
        f1s = [report[c]["f1-score"] for c in class_names]

        plot_per_class(recalls, class_names,
                       f"{model_name} Per-Class Recall",
                       f"{model_name}_recall.png")

        plot_per_class(f1s, class_names,
                       f"{model_name} Per-Class F1",
                       f"{model_name}_f1.png")

    # Comparison charts
    plot_metric_comparison(results, "accuracy", "accuracy_comparison.png")
    plot_metric_comparison(results, "macro_recall", "macro_recall_comparison.png")
    plot_metric_comparison(results, "weighted_f1", "f1_comparison.png")

    # Table
    save_table(results, "model_comparison_table.png")

    print("\n✅ All graphs saved in:", SAVE_DIR)