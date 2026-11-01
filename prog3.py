# Experiment 3: Implement a DeepFake application using GAN

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
# torch	Deep learning library, nn	Creates neural network layers, datasets	Loads MNIST dataset, transforms	Changes image format, save_image	Saves generated images

z, epochs = 64, 50          # generate 64 random numbers, train 50 times

data = torch.utils.data.DataLoader(
    datasets.MNIST(".", download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])), batch_size=64, shuffle=True)

G = nn.Sequential(
        nn.Linear(z, 256), nn.ReLU(),
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 784), nn.Tanh()
    )

D = nn.Sequential(
        nn.Linear(784, 512), nn.LeakyReLU(0.2),
        nn.Linear(512, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 1), nn.Sigmoid()
    )

og = torch.optim.Adam(G.parameters(), lr=0.0002)  # Optimizer for the Generator.
od = torch.optim.Adam(D.parameters(), lr=0.0002)  # update the neural network's weights.  lr = Learning rate.
loss = nn.BCELoss()     # Binary Cross Entropy loss

for ep in range(epochs):        # Repeats training for 50 epochs.
    for real, _ in data:
        b = len(real)
        real = real.view(b, 784)            # Reshapes each 28 × 28 image into a row of 784 values.
        ones, zeros = torch.ones(b, 1), torch.zeros(b, 1)

        fake = G(torch.randn(b, z)).detach()
        ld = loss(D(real), ones) + loss(D(fake), zeros)
        od.zero_grad()          # Clears old gradients
        ld.backward()           # Calculates gradients
        od.step()               # Updates the Discriminator's weights using Adam.

        fake = G(torch.randn(b, z))
        lg = loss(D(fake), ones)
        og.zero_grad()
        lg.backward()
        og.step()

    print("Epoch:", ep+1, "D:", ld.item(), "G:", lg.item())

    if (ep+1) % 10 == 0:
        save_image(G(torch.randn(64, z)).view(-1, 1, 28, 28),
                   f"gan{ep+1}.png", nrow=8, normalize=True)
