# THIS

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

INPUT = 784
HIDDEN = 512
BATCH_SIZE = 100


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # TODO initialize layers

        self.L1 = nn.Linear(784, 512)
        self.L2 = nn.Linear(512, 256)
        self.L3 = nn.Linear(256, 32)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, X):
        # TODO execute layers and return result
        X = F.tanh(self.L1(X))
        X = self.drop(X)
        X = F.tanh(self.L2(X))
        mu = (self.L3(X))
        sigma = (self.L3(X))

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # TODO initialize layers
        self.d1 = nn.Linear(32, 256)
        self.d2 = nn.Linear(256, 512)
        self.d3 = nn.Linear(512, 784)

    def forward(self, Z):
        # TODO execute layers and return result
        Z = F.tanh(self.d1(Z))
        Z = F.tanh(self.d2(Z))
        Z = (self.d3(Z))
        Z = F.sigmoid(Z)

        return Z


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    # Computes the hidden representation given the pre-sampled values eps
    def z(self, x):
        code = self.encoder(x)
        # TODO extract mu and sigma2 from the code, and then use them and eps
        # to compute z.
        mu = code[0]
        sigma = code[1]
        e = torch.randn_like(sigma)
        z = e.mul(sigma) + mu

        return z, mu, sigma

    def x(self, z):
        x = self.decoder(z)
        return x


def visualization(X):
    image = X
    image = image.permute(1, 2, 0)
    image = image.detach().numpy()
    plt.imshow(image)
    plt.show()


def train(device, vae, X):
    optimizer = optim.Adam(vae.parameters())  # worked for me

    my_dataset = TensorDataset(X)
    my_dataloader = DataLoader(X, 100, shuffle=True)

    def computeLoss(x):
        # TODO implement your custom loss
        input = x
        input = input.to(device)
        optimizer.zero_grad()

        z, new_mu, new_sigma = vae.z(x)
        recon_x = vae.x(z)

        recon_loss = F.binary_cross_entropy(recon_x, input, reduction='sum')
        KL_loss = 0.5 * torch.sum(torch.exp(new_sigma) + new_mu ** 2 - 1 - new_sigma)
        loss = recon_loss + 0.45 * KL_loss

        return loss, z

    for e in range(50):  # epochs

        print("Epoch {}".format(e + 1))
        loss_tol = 0
        for _, minibatch in enumerate(my_dataloader):
            # Call computeLoss on each minibatch
            loss, z = computeLoss(minibatch)
            loss.backward()
            loss_tol += loss.item()
            optimizer.step()
        loss_tol /= len(X)
        print(loss_tol)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)

    X = torch.from_numpy(np.load("mnist_train_images.npy")).float().to(device)
    train(device, vae, X)
    # Original
    T = X[:16].view(16, 1, 28, 28)
    visualization(make_grid(T.cpu(), nrow=16))
    # reconstructed images using VAE
    eps = torch.randn((16, 32)).to(device)
    z, _, _ = vae.z(X[:16])
    recon_ = vae.x(z)
    img_recon = recon_.view(16, 1, 28, 28)
    visualization(make_grid(img_recon.cpu(), nrow=16))
    # random noise
    xhat = vae.x(eps)
    img = xhat.view(16, 1, 28, 28)
    visualization(make_grid(img.cpu(), nrow=16))