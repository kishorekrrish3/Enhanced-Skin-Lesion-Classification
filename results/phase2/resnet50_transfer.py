import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from train_utils import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# -------------------------------
# MIXUP FUNCTIONS
# -------------------------------

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -------------------------------
# MODEL
# -------------------------------

def get_model(num_classes):

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model


# -------------------------------
# TRAIN FUNCTION (WITH MIXUP)
# -------------------------------

def train_model(model, train_loader, val_loader, device,
                epochs=50, patience=5):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0

    save_path = "phase2/saved_models/resnet50_best.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # 🔥 APPLY MIXUP HERE
            images, targets_a, targets_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()

            outputs = model(images)

            loss = mixup_criterion(
                criterion,
                outputs,
                targets_a,
                targets_b,
                lam
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print("✅ Validation loss improved — model saved")

        else:
            patience_counter += 1
            print(f"⏸ No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("\n🛑 Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path))

    return model


# -------------------------------
# EVALUATION
# -------------------------------

def evaluate_model(model, loader, device, class_names, split_name="Test"):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    print(f"\n📊 {split_name} Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"\n📉 {split_name} Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    # Key Metrics
    accuracy = report["accuracy"]
    macro_recall = report["macro avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(f"\n✅ {split_name} Key Metrics:")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    return accuracy, macro_recall, weighted_f1


# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "skin-ds")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", DEVICE)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR,
        batch_size=32
    )

    model = get_model(num_classes=len(class_names)).to(DEVICE)

    model = train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        epochs=50,
        patience=5
    )

    # Evaluation
    evaluate_model(model, val_loader, DEVICE, class_names, "Validation")
    evaluate_model(model, test_loader, DEVICE, class_names, "Test")