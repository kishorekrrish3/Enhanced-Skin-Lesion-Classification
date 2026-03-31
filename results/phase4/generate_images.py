import torch
from torchvision.utils import save_image
import os
from train_dcgan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100

target_samples = {
    "Dermatofibroma": 600,
    "Squamous cell carcinoma": 400,
    "Actinic keratoses": 300,
    "Vascular lesions": 600
}

for cls, num_images in target_samples.items():

    print(f"\nGenerating images for {cls}")

    G = Generator().to(device)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gan_models/{cls}_generator.pth")

    G.load_state_dict(torch.load(model_path))

    G.eval()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../dataset/skin-ds/train/{cls}")

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):

        noise = torch.randn(1, z_dim, 1, 1).to(device)

        fake = G(noise)

        save_image(
            fake,
            f"{save_dir}/synthetic_{i}.png",
            normalize=True
        )

    print(f"Generated {num_images} images for {cls}")