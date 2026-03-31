import os
import shutil

# Resolve paths relative to this script's location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(os.path.join(SCRIPT_DIR, "../dataset/skin-ds/train"))
DEST = os.path.abspath(os.path.join(SCRIPT_DIR, "gan_dataset"))

classes = [
    "Dermatofibroma",
    "Vascular lesions",
    "Squamous cell carcinoma",
    "Actinic keratoses"
]

if not os.path.exists(SOURCE):
    print(f"Error: SOURCE directory not found at {SOURCE}")
    exit(1)

os.makedirs(DEST, exist_ok=True)

for cls in classes:
    src = os.path.join(SOURCE, cls)
    dst = os.path.join(DEST, cls)

    if os.path.exists(src):
        print(f"Copying {cls} to {dst}...")
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        print(f"Warning: Class directory '{cls}' not found in SOURCE.")

print("\nGAN dataset prepared")
