import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import IMG_SIZE, NORM_MEAN, NORM_STD

# -------------------------------
# Valid/Test transforms (NO AUGMENTATION)
# -------------------------------
get_val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2()
])


# -------------------------------
# Standard Training transforms (For Majority Classes)
# -------------------------------
# We keep this realistic to prevent corrupting the abundant normal classes
get_standard_train_transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.3),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2()
])


# -------------------------------
# Strong Training transforms (For Minority Classes)
# -------------------------------
# Here we add aggressive but safe medical augmentations to battle severe class imbalance
get_strong_train_transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-30, 30), p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    
    # Noise and blurring
    A.OneOf([
        A.GaussNoise(std_range=(0.01, 0.02), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),

    # Distortions
    A.OneOf([
        A.ElasticTransform(alpha=30, sigma=30 * 0.05, p=1.0),
        A.GridDistortion(p=1.0),
        A.OpticalDistortion(distort_limit=0.1, p=1.0),
    ], p=0.3),

    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2()
])
