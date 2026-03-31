import os
import torch

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "skin-ds")
RESULTS_DIR = os.path.join(BASE_DIR, "core", "results")

# Sub-directories for results
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for d in [METRICS_DIR, PLOTS_DIR, GRADCAM_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------------
# HARDWARE
# -------------------------------
# RTX 3050 - optimizing for 4GB VRAM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # Reduced from 32 to ensure stable MixUp/CutMix + layer4 fine-tuning fits in 4GB VRAM
NUM_WORKERS = min(4, os.cpu_count() or 1)

# -------------------------------
# DATASET CONSTANTS
# -------------------------------
IMG_SIZE = 224

# ImageNet normalization standard (important since we use pretrained ResNet50)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD",
    "Healthy",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_ABBREV = [
    "AK", "BCC", "BKL", "CP", "CPX", "DF", "HFMD",
    "HLT", "MSL", "MN", "MEL", "MPX", "SCC", "VL"
]

# We consider any class with < 700 samples in the train set as a minority class.
# These will receive stronger augmentation to combat Severe class imbalance.
MINORITY_CLASSES = [
    "Actinic keratoses",
    "Measles",
    "Dermatofibroma",
    "Squamous cell carcinoma",
    "Vascular lesions"
]
