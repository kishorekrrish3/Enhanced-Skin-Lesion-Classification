import torch
from torchvision.utils import save_image
import os
from train_dcgan_fixed import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100

# 🔥 LIMITED generation
targets = {
    "Dermatofibroma": 200,
    "Squamous cell carcinoma": 200,
    "Vascular lesions": 150
}

for cls, n in targets.items():

    G = Generator().to(device)
    G.load_state_dict(torch.load(f"gan_models_fixed/{cls}.pth"))
    G.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "..", "dataset", "skin-ds", "train", cls)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        noise = torch.randn(1, z_dim, 1, 1).to(device)
        fake = G(noise)

        save_image(fake, f"{save_dir}/gan_{i}.png", normalize=True)

    print(f"Generated {n} images for {cls}")