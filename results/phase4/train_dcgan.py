import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image

class FlatDirectoryDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 128
batch_size = 64
z_dim = 100
epochs = 100


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_gan(class_name):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "gan_dataset", class_name)
    os.makedirs(os.path.join(script_dir, "gan_models"), exist_ok=True)

    dataset = FlatDirectoryDataset(dataset_path, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for real, _ in loader:

            real = real.to(device)
            batch = real.size(0)

            noise = torch.randn(batch, z_dim, 1, 1).to(device)

            fake = G(noise)

            # train discriminator

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D = criterion(D_real, torch.ones_like(D_real)) + \
                     criterion(D_fake, torch.zeros_like(D_fake))

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train generator

            output = D(fake)

            loss_G = criterion(output, torch.ones_like(output))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"{class_name} Epoch {epoch+1}/{epochs} | D Loss {loss_D:.4f} | G Loss {loss_G:.4f}")

    torch.save(G.state_dict(), os.path.join(script_dir, "gan_models", f"{class_name}_generator.pth"))


if __name__ == "__main__":
    target_classes = [
        "Dermatofibroma",
        "Squamous cell carcinoma",
        "Actinic keratoses",
        "Vascular lesions"
    ]

    for cls in target_classes:
        print(f"\nTraining GAN for {cls}\n")
        train_gan(cls)