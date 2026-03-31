import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .config import DATA_DIR, CLASS_NAMES, MINORITY_CLASSES, BATCH_SIZE, NUM_WORKERS
from .augmentation import get_val_transform, get_standard_train_transform, get_strong_train_transform

class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, class_names, minority_classes, phase='train'):
        """
        Custom Dataset that applies strong augmentations to minority classes during training.
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.minority_classes = minority_classes
        self.phase = phase
        
        self.image_paths = []
        self.labels = []
        
        # Populate paths and labels
        for idx, cls in enumerate(self.class_names):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[label]
        
        # Read image using cv2 (Default BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Convert BGR to RGB (Required for Albumentations and PyTorch standards)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms based on phase and class logic
        if self.phase == 'train':
            if class_name in self.minority_classes:
                augmented = get_strong_train_transform(image=image)
            else:
                augmented = get_standard_train_transform(image=image)
        else:
            augmented = get_val_transform(image=image)
            
        image = augmented['image']
        
        return image, label

def get_dataloaders():
    """Build train, val, test loaders with Class-Aware Oversampling"""
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # Datasets
    train_dataset = SkinLesionDataset(train_dir, CLASS_NAMES, MINORITY_CLASSES, phase='train')
    val_dataset = SkinLesionDataset(val_dir, CLASS_NAMES, MINORITY_CLASSES, phase='val')
    test_dataset = SkinLesionDataset(test_dir, CLASS_NAMES, MINORITY_CLASSES, phase='test')
    
    # Imbalance Handling: Calculate Weights for Sampler
    targets = np.array(train_dataset.labels)
    class_counts = np.bincount(targets)
    
    # Check for empty classes
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_weights[i] = 1.0 / class_counts[i]
            
    sample_weights = np.array([class_weights[t] for t in targets])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),  # Original length to keep epoch duration stable
        replacement=True
    )
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_counts
