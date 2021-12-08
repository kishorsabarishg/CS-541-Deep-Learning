import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # TODO initialize layers
        self.L1 = nn.Linear(100, 128)
        self.L1_ = nn.Linear(128, 256)
        self.L2 = nn.Linear(256, 512)
        self.L3 = nn.Linear(512, 600)
        self.L4 = nn.Linear(600, 784)
        self.bn0 = nn.BatchNorm1d(num_features=128)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=600)

    def forward(self, Z):
        # TODO execute layers and return result
        Z = F.relu(self.L1(Z))
        Z = self.bn0(Z)
        Z = F.relu(self.L1_(Z))
        Z = self.bn1(Z)
        Z = F.relu(self.L2(Z))
        Z = self.bn2(Z)
        Z = F.relu(self.L3(Z))
        Z = self.bn3(Z)
        Z = F.sigmoid(self.L4(Z))

        return Z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO initialize layers
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.drop = nn.Dropout(p=0.5)
        self.bn0 = nn.BatchNorm1d(num_features=1024)
        self.bn1 = nn.BatchNorm1d(num_features=512)

    def forward(self, X):
        # TODO execute layers and return result
        X = self.drop(X)
        X = F.relu(self.fc1(X))
        # X = self.bn0(X)
        X = F.relu(self.fc2(X))
        # X = self.bn1(X)
        X = F.sigmoid(self.fc3(X))

        return X


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def generate(self, x):
        z = self.generator(x)
        return z

    def discriminate(self, z):
        return self.discriminator(z)


BATCH_SIZE = 100


def visualization(X):
    image = X
    image = image.permute(1, 2, 0)
    image = image.detach().numpy()
    plt.imshow(image)
    plt.show()


def train(device, gan, X):
    optimizerG = optim.Adam(gan.generator.parameters(), lr=5e-5)  # worked for me
    optimizerD = optim.Adam(gan.discriminator.parameters(), lr=1e-4)  # worked for me
    criterion = nn.BCELoss()
    my_dataset = TensorDataset(X)
    my_dataloader = DataLoader(X, 100, shuffle=True)
    for it in range(200):  # epochs
        count = 1
        if it % 10 == 0:
            eps = torch.randn((100, 100)).to(device)
            z = gan.generator(eps)
            # xhat = gan.x(z)
            img = z.view(100, 1, 28, 28).data
            visualization(make_grid(img.cpu(), nrow=20))

        print("Epoch :: ", it + 1)
        count = 0
        dis_loss = 0
        gen_loss = 0

        for _, minibatch in enumerate(my_dataloader):
            count += 1
            if it % 5 != 0:  # Skip training discriminator every ith iteration
                gan.discriminator.zero_grad()
                ############# DISCRIMINATOR #############
                realX = minibatch
                realY = torch.ones(BATCH_SIZE, 1)
                realX = realX.to(device)
                discrimintor_rop = gan.discriminator(realX)
                realY = realY.to(device)
                discriminator_rl = criterion(discrimintor_rop, realY)
                z = torch.randn(BATCH_SIZE, 100).to(device)
                fakeX = gan.generator(z)
                discriminator_fop = gan.discriminator(fakeX)
                fakeY = torch.zeros(BATCH_SIZE, 1).to(device)
                discriminator_fl = criterion(discriminator_fop, fakeY)
                Total_D_Loss = discriminator_rl + discriminator_fl
                Total_D_Loss.backward()
                optimizerD.step()
                dis_loss += Total_D_Loss.item()

            # TODO Train generator
            gan.generator.zero_grad()
            genX = torch.randn(BATCH_SIZE, 100).to(device)
            generator_op = gan.generator(genX)
            genY = torch.ones(BATCH_SIZE, 1).to(device)
            discriminator_op = gan.discriminator(generator_op)
            generator_Loss = criterion(discriminator_op, genY)
            generator_Loss.backward()
            optimizerG.step()
            gen_loss += generator_Loss.item()
            count += 1

        '''for _, minibatch in enumerate(my_dataloader): #UNCOMMENT FOR DROPPING MINI BATCHES WHILE TRAINING DISCRIMINATOR INSTEAD OF SKIPPING DISCRIMINATOR TRAINING EVERY FEW EPOCHS
          count+=1
          if count % 500 != 0: # Skip training discriminator every ith iteration
            gan.discriminator.zero_grad()
            ############# DISCRIMINATOR #############
            realX = minibatch
            realY = torch.ones(BATCH_SIZE, 1)
            realX = realX.to(device)
            discrimintor_rop = gan.discriminator(realX)
            realY = realY.to(device)
            discriminator_rl = criterion(discrimintor_rop, realY)
            z = torch.randn(BATCH_SIZE, 100).to(device)
            fakeX = gan.generator(z)
            discriminator_fop = gan.discriminator(fakeX)
            fakeY = torch.zeros(BATCH_SIZE, 1).to(device)
            discriminator_fl = criterion(discriminator_fop, fakeY)
            Total_D_Loss = discriminator_rl + discriminator_fl
            Total_D_Loss.backward()
            optimizerD.step()
            dis_loss += Total_D_Loss.item()
  
          # TODO Train generator
          gan.generator.zero_grad()        
          genX = torch.randn(BATCH_SIZE, 100).to(device)
          generator_op = gan.generator(genX)
          genY = torch.ones(BATCH_SIZE, 1).to(device)        
          discriminator_op = gan.discriminator(generator_op)
          generator_Loss = criterion(discriminator_op, genY)
          generator_Loss.backward()
          optimizerG.step()
          gen_loss += generator_Loss.item()
          count+=1'''

        print("Generator_Loss=", gen_loss / 55000, "Discriminator_Loss", dis_loss / 55000)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = GAN().to(device)
    X = torch.from_numpy(np.load("mnist_train_images.npy")).float().to(device)
    train(device, gan, X)

    eps = torch.randn((100, 100)).to(device)
    z = gan.generator(eps)
    img = z.view(100, 1, 28, 28)
    visualization(make_grid(img.cpu(), nrow=16))