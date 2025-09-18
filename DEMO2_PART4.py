import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =====================================================
# Dataset loader
# =====================================================
def load_oasis_png(data_dir="OASIS", img_size=128, batch_size=64):
    """
    Load MRI PNG slices from OASIS dataset using torchvision.datasets.ImageFolder.
    Expects subfolders like keras_png_slices_train, validate, test.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "keras_png_slices_train"),
                                         transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "keras_png_slices_validate"),
                                       transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "keras_png_slices_test"),
                                        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# =====================================================
# VAE definition
# =====================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 128 * 16 * 16)
        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec_deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 128, 16, 16)
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        return torch.sigmoid(self.dec_deconv3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =====================================================
# Loss function
# =====================================================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# =====================================================
# Training loop
# =====================================================
def train_vae(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch}, Train loss: {train_loss/len(train_loader.dataset):.4f}")


# =====================================================
# Visualization
# =====================================================
def visualize_reconstruction(model, test_loader, device, out_file="reconstruction.png"):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon, _, _ = model(x)
        # Take first 8 samples
        n = 8
        plt.figure(figsize=(16, 4))
        for i in range(n):
            # Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x[i, 0].cpu(), cmap="gray")
            plt.axis("off")
            # Reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon[i, 0].cpu(), cmap="gray")
            plt.axis("off")
        plt.savefig(out_file)
        plt.show()


def visualize_latent_space(model, test_loader, device, out_file="latent_space.png"):
    model.eval()
    zs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            zs.append(mu.cpu())
    zs = torch.cat(zs, dim=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(zs[:, 0], zs[:, 1], alpha=0.5)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space")
    plt.savefig(out_file)
    plt.show()


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = load_oasis_png("OASIS", img_size=128, batch_size=64)

    model = VAE(latent_dim=2).to(device)
    train_vae(model, train_loader, val_loader, device, epochs=20, lr=1e-3)

    # Visualizations
    visualize_reconstruction(model, test_loader, device)
    visualize_latent_space(model, test_loader, device)
