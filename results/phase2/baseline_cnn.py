import os
import torch
import torch.nn as nn
import torch.optim as optim
from train_utils import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------------
# Simple CNN Model (Baseline A)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -------------------------------
# Training with Early Stopping
# -------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    patience=5,
    save_path="best_model.pth"
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # -------- Training --------
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{max_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # -------- Early Stopping Logic --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), save_path)
            print("✅ Validation loss improved — model saved")

        else:
            epochs_without_improvement += 1
            print(
                f"⏸ No improvement "
                f"({epochs_without_improvement}/{patience})"
            )

        if epochs_without_improvement >= patience:
            print("\n🛑 Early stopping triggered")
            break

    # Load best model before returning
    model.load_state_dict(torch.load(save_path))
    return model


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model, loader, device, class_names, split_name="Test"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    print(f"\n📊 {split_name} Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"\n📉 {split_name} Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    accuracy = report["accuracy"]
    macro_recall = report["macro avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(f"\n✅ {split_name} Key Metrics:")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    return accuracy, macro_recall, weighted_f1


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "skin-ds")
    SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", DEVICE)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR,
        batch_size=32
    )

    model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)

    best_model_path = os.path.join(
        SAVE_DIR, "baseline_simplecnn_best.pth"
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        max_epochs=50,   # high upper bound
        patience=5,      # early stopping patience
        save_path=best_model_path
    )

    evaluate_model(model, val_loader, DEVICE, class_names, split_name="Validation")
    evaluate_model(model, test_loader, DEVICE, class_names, split_name="Test")

    print(f"\n💾 Best model saved at: {best_model_path}")
