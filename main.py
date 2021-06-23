import torch
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
from model import generator_net, discriminator_net

IN_VECTOR = 100
NO_FEATURE_MAPS = 64
OUT_CHANNEL = 3
BATCH_SIZE = 128 
NUM_EPOCHS = 100

lr = 3e-4
beta1 = 0.5

root = 'cat'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

gan_dataset = datasets.ImageFolder(root, transform=transform)

dataloader = DataLoader(gan_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCELoss()

test = torch.randn(1, IN_VECTOR, 1, 1)

generator = generator_net(IN_VECTOR, OUT_CHANNEL, feature_maps=NO_FEATURE_MAPS)
discriminator = discriminator_net(OUT_CHANNEL, NO_FEATURE_MAPS)

optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

G_loss = []
D_loss = []

for epoch in range(NUM_EPOCHS):

    for i, data in enumerate(dataloader, 0):

        discriminator.zero_grad()
        noise = torch.randn((len(data[0]), IN_VECTOR, 1 ,1))

        # first g.d. with real data
        output = discriminator(data[0]).view(-1)
        label = torch.ones(size=(len(data[0]),), dtype=torch.float, requires_grad=False)

        d_real_loss = criterion(output, label)
        d_real_loss.backward()
        real_loss = d_real_loss.mean().detach().item()

        # now g.d. with fake data
        generated = generator(noise)
        output = discriminator(generated.detach()).view(-1)
        label = torch.zeros_like(label)

        d_fake_loss = criterion(output, label)
        d_fake_loss.backward()
        fake_loss = d_fake_loss.mean().detach().item()


        d_loss = d_real_loss + d_fake_loss
        optimizerD.step()
        D_loss.append(d_loss)

        # now generator
        generator.zero_grad()

        output = discriminator(generated).view(-1)
        label = torch.ones_like(label)
        discriminator.zero_grad()

        g_loss = criterion(output, label)
        g_loss.backward()
        G_loss.append(g_loss)

        optimizerG.step()

        if i % 50 == 0:
            r_gloss = g_loss.mean().item()
            r_dloss = d_loss.mean().item()

            print(f"Epoch: {epoch}/{NUM_EPOCHS}\t batch number: {i}/{len(dataloader)} \
                  D_loss = {r_dloss:.2f}\t G_loss = {r_gloss:.2f}\t real_loss = {real_loss:.2f}\t fake_loss = {fake_loss:.2f}")
    
    with torch.no_grad():
        fake = generator(test).detach().squeeze()
        output = transforms.ToPILImage()(fake)
        output.save('Generated_img_'+str(epoch)+'.jpg')