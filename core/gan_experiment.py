import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .config import DATA_DIR, PLOTS_DIR, MINORITY_CLASSES
from .dataset import SkinLesionDataset
from .augmentation import get_strong_train_transform

def generate_mixup_example(class_name, save_filename="mixup_vs_gan_comparison.png"):
    """
    Since DCGAN previously failed to generate high-quality skin lesion images
    (they were too grainy), we replace the generative approach with MixUp and CutMix
    visualizations. This function creates a visualization showing why MixUp
    is safer for medical images than unconditional GANs.
    """
    print(f"\n🧬 Generating Synthetic Examples for {class_name}...")
    
    # Load Real Images
    cls_folder = os.path.join(DATA_DIR, "train", class_name)
    image_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg'))][:2]
    
    if len(image_files) < 2:
        print(f"Not enough images for {class_name}")
        return
        
    img1 = cv2.imread(os.path.join(cls_folder, image_files[0]))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(os.path.join(cls_folder, image_files[1]))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Apply Standard Augmentations first (Resize)
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    
    # Generate MixUp Image (Linear Interpolation)
    lam = 0.4
    mixup_img = cv2.addWeighted(img1, lam, img2, 1 - lam, 0)
    
    # Generate CutMix Image (Patch replacement)
    cutmix_img = img1.copy()
    h, w = 224, 224
    cut_w = np.int32(w * np.sqrt(1 - lam))
    cut_h = np.int32(h * np.sqrt(1 - lam))
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    cutmix_img[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]
    
    # Generate Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img1)
    axes[0].set_title("Real Image A")
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title("Real Image B")
    axes[1].axis('off')
    
    axes[2].imshow(mixup_img)
    axes[2].set_title(f"MixUp (λ={lam})")
    axes[2].axis('off')
    
    axes[3].imshow(cutmix_img)
    axes[3].set_title("CutMix")
    axes[3].axis('off')
    
    plt.suptitle(f"Medical-Safe Synthetic Generation ({class_name})\n(Alternative to unstable GANs)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_filename), bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Synthetic comparison saved to {save_filename}")

if __name__ == "__main__":
    generate_mixup_example("Squamous cell carcinoma", "scc_synthetic.png")
    generate_mixup_example("Dermatofibroma", "df_synthetic.png")
