import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

class SingleClassDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 128
batch_size = 32
z_dim = 100
epochs = 40   # 🔥 REDUCED


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),  # 🔥 IMPORTANT
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ---------------- GENERATOR ----------------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 64 -> 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ---------------- DISCRIMINATOR ----------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------- TRAIN ----------------

def train_gan(class_name):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "gan_dataset", class_name)
    dataset = SingleClassDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):

        for real, _ in loader:

            real = real.to(device)
            batch = real.size(0)

            # 🔥 LABEL SMOOTHING
            real_labels = torch.ones(batch).to(device) * 0.9
            fake_labels = torch.zeros(batch).to(device)

            # 🔥 ADD NOISE
            real = real + 0.05 * torch.randn_like(real)

            noise = torch.randn(batch, z_dim, 1, 1).to(device)
            fake = G(noise)

            # ----- Train D -----
            D_real = D(real).view(-1)
            D_fake = D(fake.detach()).view(-1)

            loss_D = criterion(D_real, real_labels) + \
                     criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ----- Train G -----
            output = D(fake).view(-1)
            loss_G = criterion(output, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"{class_name} Epoch {epoch+1}/{epochs} | D {loss_D:.3f} | G {loss_G:.3f}")

    os.makedirs("gan_models_fixed", exist_ok=True)
    torch.save(G.state_dict(), f"gan_models_fixed/{class_name}.pth")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    classes = [
        "Dermatofibroma",
        "Squamous cell carcinoma",
        "Vascular lesions"
    ]

    for cls in classes:
        print(f"\nTraining GAN for {cls}\n")
        train_gan(cls)