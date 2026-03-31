import os
import argparse
import torch

from .config import DEVICE, NUM_CLASSES, CLASS_NAMES, MODELS_DIR
from .dataset import get_dataloaders
from .losses import get_criterion
from .models import ImprovedResNet50, SimpleCNN
from .train import train_model
from .evaluate import evaluate_model
from .gan_experiment import generate_mixup_example

def build_parser():
    parser = argparse.ArgumentParser(description="Skin Lesion Classification Pipeline")
    parser.add_argument('--train', action='store_true', help="Train the improved ResNet50 model")
    parser.add_argument('--eval', action='store_true', help="Evaluate existing models")
    parser.add_argument('--baseline', action='store_true', help="Train/eval SimpleCNN baseline instead of ResNet50")
    parser.add_argument('--synth', action='store_true', help="Generate synthetic data comparisons")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Auto-run evaluate if nothing provided
    if not (args.train or args.eval or args.synth):
        print("No args provided. Defaulting to --synth and --eval mode.")
        args.synth = True
        args.eval = True
        
    # Generate Synthetic examples
    if args.synth:
        generate_mixup_example("Squamous cell carcinoma", "scc_synthetic.png")
        generate_mixup_example("Dermatofibroma", "df_synthetic.png")
        
    if args.train or args.eval:
        train_loader, val_loader, test_loader, class_counts = get_dataloaders()
        
        # Decide which model
        if args.baseline:
            model_name = "SimpleCNN"
            model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
            use_focal = False # Baseline didn't use Focal Loss
        else:
            model_name = "ImprovedResNet50"
            model = ImprovedResNet50(num_classes=NUM_CLASSES).to(DEVICE)
            use_focal = True
            
        criterion = get_criterion(class_counts, DEVICE, use_focal=use_focal)
        
        # Train
        if args.train:
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                model_name=model_name,
                epochs=50,
                patience=6
            )
            
        # Eval
        if args.eval:
            # Load best weights if exist
            save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            if os.path.exists(save_path):
                print(f"Loading weights from {save_path}")
                model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            else:
                print(f"⚠️ Warning: No pre-trained weights found at {save_path}. Evaluating initialized weights.")
                
            evaluate_model(model, val_loader, model_name=model_name, dataset_split="val")
            evaluate_model(model, test_loader, model_name=model_name, dataset_split="test")

if __name__ == "__main__":
    main()
