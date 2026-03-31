import os
import json
import torch
import torch.optim as optim
from sklearn.metrics import recall_score
import numpy as np

from .config import BATCH_SIZE, DEVICE, MODELS_DIR, METRICS_DIR, CLASS_NAMES

def mixup_data(x, y, alpha=0.3):
    """MixUp Augmentation for Robustness"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for MixUp"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def validate(model, loader, criterion):
    """Validation loop calculating Loss and Macro Recall"""
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    avg_loss = val_loss / len(loader)
    
    # Calculate Macro Recall (Critical metric for imbalanced data)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return avg_loss, macro_recall

def train_model(model, train_loader, val_loader, criterion, model_name="ImprovedResNet50", epochs=50, patience=6):
    """Main Training Loop with Early Stopping on Macro Recall"""
    print(f"\n🚀 Starting Training: {model_name} on {DEVICE}")
    print(f"Epochs: {epochs} | Batch Size: {BATCH_SIZE}")
    
    # Optimizer - focusing mostly on unfrozen layer4 & FC as they have requires_grad=True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_macro_recall = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    history = {"train_loss": [], "val_loss": [], "val_macro_recall": []}
    
    for epoch in range(epochs):
        # ---------------- TRAIN ---------------- #
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Apply MixUp Randomly (50% of the time to maintain some real representation)
            if np.random.rand() > 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # ---------------- VALIDATION ---------------- #
        avg_val_loss, macro_recall = validate(model, val_loader, criterion)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_recall"].append(macro_recall)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Recall: {macro_recall:.4f}")
        
        # ---------------- EARLY STOPPING (Optimized for Recall) ---------------- #
        # We save models primarily based on Macro Recall improvement!
        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            best_val_loss = avg_val_loss  # keep track of best loss corresponding to this
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best Macro Recall improved -> {macro_recall:.4f}. Model saved.")
            
        # Or if recall matched, but loss heavily improved
        elif macro_recall == best_macro_recall and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Loss improved (Recall maintained). Model saved.")
            
        else:
            patience_counter += 1
            print(f"⏸ No improvement ({patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print("\n🛑 Early stopping triggered!")
            break

    # Save training history for dashboard
    with open(os.path.join(METRICS_DIR, f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f)
        
    print(f"\n💾 Training complete. Best model loaded with Macro Recall: {best_macro_recall:.4f}")
    
    # Load and return the best weights
    model.load_state_dict(torch.load(save_path))
    return model
